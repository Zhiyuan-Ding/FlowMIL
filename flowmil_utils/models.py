import tensorflow as tf
import numpy as np
import warnings
from flowmil_utils.losses import *
# import tensorflow_probability as tfp
import h5py
import os

'''
models code basis for FlowMIL, this script includes

'''
# optional periodic layers from "Automated Deep Learning-Based Diagnosis and Molecular Characterization of Acute Myeloid Leukemia Using Flow Cytometry"
# have not observed advantages compared to MLP so far
class PeriodicLayer(tf.keras.layers.Layer):
    def __init__(self, units, sigma=1.0, **kwargs):
        """Initializes the Periodic Layer with a specified number of units and standard deviation for initialization.
        Args:
            units (int): The dimensionality of the output space.
            sigma (float, optional): Standard deviation for the initialization of trainable parameters. Defaults to 1.0.
        """
        super(PeriodicLayer, self).__init__(**kwargs)
        self.units = units
        self.sigma = sigma

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # ci are trainable parameters initialized from N(0,σ)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units),
                                    #   initializer=tf.random_normal_initializer(mean=0., stddev=self.sigma, seed=42),
                                      initializer=tf.ones_initializer(),
                                      trainable=False)
    def call(self,inputs):
        v = 2 * np.pi * tf.multiply(inputs[:, :, tf.newaxis], self.kernel[tf.newaxis, :, :])
        return tf.concat([tf.sin(v), tf.cos(v)], axis=-1)
    def get_config(self):
        return super().get_config().update({'units': self.units, 'sigma': self.sigma})

# MLP
class MLP(tf.keras.Model):
    def __init__(self, input_spec, model_spec, inner_activation=tf.keras.activations.softplus, output_activation=None, output_name=None,seed=42,bn=False):
        '''
        input_spec: input specfication,
        model_spec: a list of float between [0,1] and int, where float represent a dropout layer and int represent linear layer with the channel number of this number
        inner_activation: inner activation function after linear layers except the last one
        output_activation: output activation function
        bn: whether or not use batch normalization
        '''
        self.config = {'input_spec':input_spec, 'model_spec':model_spec, 'inner_activation':inner_activation, 'output_activation':output_activation, 'output_name':output_name, 'seed':seed, 'bn':bn}
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        model_tensors = [tf.keras.layers.Input(type_spec=input_spec)]
        # inner layers with inner activation and optional dropout is model_spec < 1
        for i in range(len(model_spec) - 1):
            if model_spec[i] < 1:
                model_tensors.append(tf.keras.layers.Dropout(rate=model_spec[i])(model_tensors[-1]))
            else:
                model_tensors.append(tf.keras.layers.Dense(units=model_spec[i], activation=inner_activation,kernel_initializer=initializer)(model_tensors[-1]))
                if bn:
                    model_tensors.append(tf.keras.layers.BatchNormalization()(model_tensors[-1]))
        # output activation can be set different from inner layers
        model_tensors.append(tf.keras.layers.Dense(units=model_spec[-1], activation=output_activation, name=output_name,kernel_initializer=initializer)(model_tensors[-1]))

        # build keras model
        super(MLP, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1]])
    def get_config(self):
        cfg = self.config
        return cfg  

# optional MLP with periodic layers as the first layer of encoder
class MLP_PLR(tf.keras.Model):
    def __init__(self, input_spec, plr_spec, mlp_spec,seed=42):
        """
        plr_spec: [periodic_units, choice_units]
        mlp_spec: list of units in MLP
        """
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        model_tensors = [tf.keras.layers.Input(type_spec=input_spec)]

        # periodic activation
        model_tensors.append(PeriodicLayer(units=plr_spec[0])(model_tensors[-1]))

        # choice of activation
        model_tensors.append(tf.keras.layers.Dense(units=plr_spec[1], activation=tf.keras.activations.relu,kernel_initializer=initializer)(model_tensors[-1]))

        # flatten
        model_tensors.append(tf.keras.layers.Flatten()(model_tensors[-1]))

        # MLP block
        model_tensors.append(MLP(input_spec=tf.TensorSpec(shape=model_tensors[-1].shape, dtype=model_tensors[-1].dtype),
                                 model_spec=mlp_spec, output_activation=tf.keras.activations.softplus)(model_tensors[-1]))

        # build keras model
        super(MLP_PLR, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1]])
    def get_config(self):
        cfg = super().get_config()
        return cfg
    
# mapping function to wrap a function fn on regular tensor to ragged tensor    
class MapTensorFunctionRagged(tf.keras.layers.Layer):
    def __init__(self, fn, fn_output_signature=None, fn_map=False, **kwargs):
        """
        Args:
            fn: A callable mapping a tensor to a tensor.
            fn_output_signature: The output signature of fn. If None, the output signature will be inferred.
            fn_map: If True, use tf.map_fn instead of tf.ragged.map_flat_values.
            fn_map is true for operations within the bag and false for operations across bags.
        """
        super(MapTensorFunctionRagged, self).__init__(**kwargs)
        self.fn, self.fn_output_signature, self.fn_map = fn, fn_output_signature, fn_map
    def get_config(self):
        return super().get_config().update({'fn': self.fn, 'fn_output_signature': self.fn_output_signature, 'fn_map': self.fn_map})
    
    @tf.function
    def call(self, inputs):
        if self.fn_map:
            return tf.map_fn(self.fn, inputs, fn_output_signature=self.fn_output_signature)
        else:
            return tf.ragged.map_flat_values(self.fn, inputs)

# attention based aggr layer for ragged tensor
class AttentionBasedAggregation(tf.keras.layers.Layer):
    '''
    pre_normalization: if true, the attention normalization is done before weighted average, else after it
    att_normalization: the attention normalization strategy, for attention a_{i,j} from instance i, attention head j
    if pre_normalization:
        attention can be normalized 
            along the instance dim (a_{i,j}' = a_{i,j} / \sum_i a_{i,j}),
            along both instance dim and attention dim (a_{i,j}' = a_{i,j} / \sum_{i,j} a_{i,j}),
            divided simply by the number of instance
        then aggregated feature is computed as f_{sample}' = \sum_i a_{i,j}'f_{i,j}
    else:
        weighted sum feature is first computed as f_{sample} = \sum_i a_{i,j}f{i,j}
        attention sum is computed as:
            along the instance dim (a_{sum} =  \sum_i a_{i,j}),
            along both instance dim and attention dim a_{sum}' = a_{sum} / sum_j a_{sum}
            divided simply by the number of instance a_{num} = number of instance
        then f_{sample}' =  f_{sample} / a
    inputs is a list of ragged tensor, with first ragged feature, second attention
    '''
    def __init__(self, att_normalization=None, pre_normalization = False,**kwargs):
        super().__init__(**kwargs)
        if att_normalization is not None:
            self.att_normalization = att_normalization
        else:
            self.att_normalization = 'normalize_along_instance'
        self.pre_normalization = pre_normalization
    def call(self, inputs):
        
        if self.pre_normalization:
            if self.att_normalization == 'normalize_along_instance':
                instance_att = inputs[1] / tf.reduce_sum(inputs[1],axis = 1,keepdims = True)
            elif self.att_normalization == 'normalize_along_both':
                instance_att = inputs[1] / tf.reduce_sum(inputs[1],axis = 1,keepdims = True)
                instance_att = instance_att / tf.reduce_sum(instance_att,axis = 2,keepdims = True)
            elif self.att_normalization == 'normalize_based_on_instance':
                instance_att = inputs[1] / tf.reduce_sum(tf.ones_like(inputs[1])[:,:,0:1],axis = 1,keepdims = True)
            instance_att = tf.where(tf.math.is_nan(instance_att), tf.zeros_like(instance_att), instance_att)
            features_weighted_sum = tf.reduce_sum(inputs[0][:,:,tf.newaxis,:] * instance_att[:,:,:,tf.newaxis],axis=1)
            return features_weighted_sum, tf.reduce_sum(instance_att,axis=1)
        else:
            # inputs[0] of shape (batch_size, num_instances, num_features), inputs[1] of shape (batch_size, num_instances,nheads)
            # (n_element,1,n_feature) * (n_element,n_head,1) -> (n_element,n_head,n_feature)
            features_weighted_sum = inputs[0].flat_values[:, tf.newaxis, :] * inputs[1].flat_values[:, :, tf.newaxis]
            
            # (n_element,n_head,n_feature) -> (batch_size, n_head, n_features)
            features_weighted_sum = tf.reduce_sum(tf.RaggedTensor.from_row_splits(features_weighted_sum, inputs[0].row_splits), axis=1)

            # (batch_size, num_instances, nheads) -> (batch_size, nheads)
            if self.att_normalization == 'normalize_along_instance':
                weights_sum = tf.reduce_sum(inputs[1], axis=1)
            elif self.att_normalization == 'normalize_along_both':
                weights_sum = tf.reduce_sum(inputs[1], axis=1)
                weights_sum = weights_sum / tf.reduce_sum(weights_sum,axis = 1,keepdims = True)
            elif self.att_normalization == 'normalize_based_on_instance':
                 weights_sum = tf.reduce_sum(tf.ones_like(inputs[1]), axis=1)
            # weighted averages of input features via normalization of weighted sum
            # (batch_size, nheads, nfeatures) / (batch_size, nheads, 1) -> (batch_size, nheads, nfeatures)
            features_weighted_avg = tf.math.divide_no_nan(features_weighted_sum , weights_sum[:, :, tf.newaxis])
            features_weighted_avg = tf.where(tf.math.is_nan(features_weighted_avg), tf.zeros_like(features_weighted_avg)+1e-5, features_weighted_avg)
            # return weighted average and sum of weights
            return features_weighted_avg, weights_sum
    def get_config(self):
        cfg = super().get_config()
        return cfg  

# optional ABMIL with periodic layers
class ABMILM_AML(tf.keras.Model):
    def __init__(self, input_spec, plr_spec, mlp_spec,instance_weight_spec,bag_spec,output_spec, **kwargs):
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        # list to hold model layers with first layer being input
        model_tensors = [tf.keras.layers.Input( shape = input_spec, dtype = tf.float32, ragged=True)]

        # instance-level MLP-PLR
        model_tensors.append(MapTensorFunctionRagged(MLP_PLR(input_spec=tf.TensorSpec(shape=model_tensors[-1].shape[1:], dtype=tf.float32),
                                                               plr_spec=plr_spec, mlp_spec=mlp_spec))(model_tensors[-1]))

        # instance-level attention weights
        #warning if instance_weight_spec[-1] != 1:
        if instance_weight_spec[-1] != 1:
            warnings.warn("The last layer of instance_weight_spec should be 1")
        #(batch_size, num_instances, 1)
        instance_weights = MLP(input_spec=tf.TensorSpec(shape=model_tensors[-1].shape, dtype=tf.float32),
                                 model_spec=instance_weight_spec, output_activation=tf.keras.activations.sigmoid,
                                 output_name='instance_attention')(model_tensors[-1])
        
        # attention-based aggregation
        #(batch_size, nheads, nfeatures), (batch_size, nheads)
        model_tensors.append(AttentionBasedAggregation()([model_tensors[-1], instance_weights]))
        
        # bag-level MLP
        model_tensors.append(MLP(input_spec=tf.TensorSpec(shape=model_tensors[-1][0].shape, dtype=tf.float32),
                                 model_spec=bag_spec, output_activation=tf.keras.activations.sigmoid)(model_tensors[-1][0]))
        model_tensors.append(tf.keras.layers.Dense(units=output_spec, activation=None,name = 'sample_prediction',kernel_initializer=initializer)(tf.squeeze(model_tensors[-1],axis = 1)))

        # build keras model
        super(ABMILM_AML, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1]])
    def get_config(self):
        cfg = super().get_config()
        return cfg
    #init weights for reproducibility
    def weights_init(self,seed=42):
        #initialize all learnable params
        np.random.seed(seed)

# optional clusterformer cross-attention for regular tensor
class RecurrentCrossAttention(tf.keras.layers.Layer):
    '''
    out = softmax(Q_c * K_f^T / sqrt (d_K) )V_f
    Q_c : (attention_head_num,n_head,feature_num) attention_head_num = num of clusters
    K_f : (instance_num,n_head,feature_num)
    V_f : (instance_num,n_head,feature_num)
    '''
    def __init__(self, att_spec, instance_spec,model_spec, inner_activation=tf.keras.activations.softmax, output_activation=None, output_name=None,seed=42,EM_steps = 1):
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        # list to hold model layers with first layer being input
        model_tensors = [[tf.keras.layers.Input(type_spec=att_spec),tf.keras.layers.Input(type_spec=instance_spec),tf.keras.layers.Input(type_spec=instance_spec)]]
        centroid = model_tensors[0][0]
        for _ in range(EM_steps):
            # (attention_head_num,n_head,feature_num) * (instance_num,n_head,feature_num) -> (attention_head_num,n_head,instance_num)
            weights = tf.nn.softmax(tf.reduce_sum(centroid[:,tf.newaxis,:,:] * model_tensors[-1][1][tf.newaxis,...],axis=-1) / (instance_spec[-1])**(0.5),axis = 0)
            # (attention_head_num,n_head,instance_num) * (instance_num,n_head,feature_num) -> (attention_head_num,n_head,feature_num)
            centroid = tf.reduce_sum(weights[:,:,tf.newaxis,:] * model_tensors[-1][2][tf.newaxis,...],axis=-2)
        model_tensors.append(centroid)
        super(RecurrentCrossAttention, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1]])
    def get_config(self):
        cfg = super().get_config()
        return cfg 

# MIL framework for flow cytometry learning
class FlowMIL(tf.keras.Model):
    '''
    init args:
    input_spec: input specification
    instance_weight_spec: network specification for instance(cellular) level network
    bag_spec: sample level network specification
    output_spec: number of classes in output
    instance_mlps_spec: layer specification for instance mlp
    block_settings: list of MIL framework setting, with format [instance encoder type, instance attention & aggregation strategy, bag-level network type]
    final_act: final activation function for bag(sample) level network
    instance_final_act: final activation function for instance level attention network
    instance_att_act: activation function for instance level hidden layers
    instance_supervision: optional, whether cellular supervised target is used
    pre_normalization: whether normalize the instance attention or after the weighted sum, refer to AttentionBasedAggregation
    att_normalization: instance attention normalization strategy, refer to AttentionBasedAggregation
    att_regularization_method: optional attention regularization target
    ashbin: number of attention heads not influced by the sample-level prediction directly
    '''
    def __init__(self, 
                input_spec: tuple = None,
                instance_weight_spec: list = None,
                bag_spec: list = None,
                output_spec: int = 2,
                instance_mlp_spec: list = None,
                block_settings: list = None,
                final_act: str = 'softmax',
                instance_final_act: str = 'softplus',
                instance_att_act: str = 'sigmoid',
                instance_supervision: bool = False,
                pre_normalization: bool =False,
                att_normalization: str = 'normalize_along_instance',
                att_regularization_method: str = None,
                ashbin: int = 0,
                seed=42,**kwargs):
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.config = {'input_spec':input_spec,   'instance_weight_spec':instance_weight_spec, 'bag_spec':bag_spec, 'output_spec':output_spec, 'instance_mlp_spec':instance_mlp_spec,
                       'block_settings':block_settings, 'final_act':final_act, 'instance_final_act':instance_final_act}
        model_tensors = [tf.keras.layers.Input( shape = input_spec, dtype = tf.float32, ragged=True)]
        
        #part 1 instance encoder
        instance_encoder_type_list = ['MLP','none']
        instance_encoder_type = block_settings[0]
        assert instance_encoder_type in instance_encoder_type_list, 'input instance encoder type is not supported'
        if instance_encoder_type == 'MLP':
            model_tensors.append(MapTensorFunctionRagged(MLP(input_spec=tf.TensorSpec(shape=model_tensors[-1].shape[1:], dtype=tf.float32),
                                                                    model_spec=instance_mlp_spec, output_activation=instance_final_act,bn=True,seed=seed))(model_tensors[-1]))
        elif instance_encoder_type == 'none':
            pass
        
        #part 2 instance attention and instance aggregation
        attention_weighted_average_list = ['mean','attention_dependent','attention_independent','sum']
        attention_computation_types = ['normalize_along_instance','normalize_along_both','normalize_based_on_instance']
        attention_weighted_average = block_settings[1]
        assert attention_weighted_average in attention_weighted_average_list, 'provided aggregation strategy is not supported'
        assert (att_normalization in attention_computation_types) + (att_normalization is None), 'provided attention normalization strategy is not supported'

        if attention_weighted_average == 'mean':
            model_tensors.append([tf.reduce_mean(model_tensors[-1][:,:,ashbin:],axis = 1,keepdims=True),tf.ones_like(model_tensors[-1])[...,0:1]])
        elif attention_weighted_average == 'attention_dependent':
            #(batch_size, num_instances, n_head)
            instance_weights = MapTensorFunctionRagged(MLP(input_spec=tf.TensorSpec(shape=model_tensors[-1].shape[1:], dtype=tf.float32),
                        model_spec=instance_weight_spec, output_activation=instance_att_act,
                        output_name='instance_attention',bn=True))(model_tensors[-1])[:,:,ashbin:]
            instance_weights = tf.where(tf.math.is_nan(instance_weights), tf.zeros_like(instance_weights), instance_weights)
            model_tensors.append(AttentionBasedAggregation(att_normalization = att_normalization,pre_normalization = pre_normalization)([model_tensors[-1], instance_weights]))
        elif attention_weighted_average == 'attention_independent':
            #(batch_size, num_instances, n_head)
            instance_weights = MapTensorFunctionRagged(MLP(input_spec=tf.TensorSpec(shape=model_tensors[0].shape[1:], dtype=tf.float32),
                        model_spec=instance_weight_spec, output_activation=instance_att_act,
                        output_name='instance_attention',bn=True,seed=seed))(model_tensors[0])[:,:,ashbin:]
            model_tensors.append(AttentionBasedAggregation(att_normalization = att_normalization, pre_normalization = pre_normalization)([model_tensors[-1], instance_weights]))
        elif attention_weighted_average == 'sum':
            model_tensors.append([tf.reduce_sum(model_tensors[-1][:,:,ashbin:],axis = 1,keepdims=True),tf.ones_like(model_tensors[-1])[...,0:1]])
        
        #part2.5 att regularization
        if att_regularization_method == 'inv_f_stat':
            assert attention_weighted_average in ['attention_dependent','attention_independent']
            att_target = inv_f_stat(model_tensors[0],instance_weights)
        elif att_regularization_method == 'centroid_dist_euclidean':
            assert attention_weighted_average in ['attention_dependent','attention_independent']
            att_target = centroid_dist_euclidean(model_tensors[0],instance_weights)
        elif att_regularization_method == 'centroid_dist_correlation':
            assert attention_weighted_average in ['attention_dependent','attention_independent']
            att_target = centroid_dist_correlation(model_tensors[0],instance_weights)
        elif att_regularization_method == 'batch_global_var':
            att_target = batch_global_var(model_tensors[0],instance_weights)
        elif att_regularization_method == 'centroid_dist_unnormalized_euclidean':
            att_target = centroid_dist_unnormalized_euclidean(model_tensors[0],instance_weights)
        else:
            att_target = None
        
        # part 3 bag encoder
        bag_encoder_type_list = ['none','MLP_single_head','MLP_multiple_head','bn','ratio', 'class_specific_mlp']
        bag_encoder_type = block_settings[2]
        assert bag_encoder_type in bag_encoder_type_list, 'provided bag encoder type is not supported'
        if bag_encoder_type == 'none':
            if len(model_tensors[-1][0].shape) == 3:
                if model_tensors[-1][0].shape[-1] == 1:
                    output_feature = tf.squeeze(model_tensors[-1][0],axis = 2)
                else:
                    output_feature = tf.squeeze(model_tensors[-1][0],axis = 1)
            model_tensors.append(tf.identity(output_feature, name="sample_prediction"))
                 
        if bag_encoder_type == 'bn':
            if len(model_tensors[-1][0].shape) == 3:
                if model_tensors[-1][0].shape[-1] == 1:
                    output_feature = tf.squeeze(model_tensors[-1][0],axis = 2)
                else:
                    output_feature = tf.squeeze(model_tensors[-1][0],axis = 1)
            else:
                output_feature = model_tensors[-1][0]
            output_feature = tf.keras.layers.BatchNormalization()(output_feature)
            model_tensors.append(tf.identity(output_feature, name="sample_prediction"))

        elif bag_encoder_type == 'MLP_single_head':
            output_feature = MLP(input_spec=tf.TensorSpec(shape=tf.squeeze(model_tensors[-1][0],1).shape, dtype=tf.float32),
                            model_spec=bag_spec, output_activation=final_act,bn=True)(tf.squeeze(model_tensors[-1][0],1))
            model_tensors.append(tf.identity(output_feature, name="sample_prediction"))
        
        elif bag_encoder_type == 'MLP_multiple_head':
            # bag-level MLP
            if len(model_tensors[-1][0].shape) == 3:
                total_dim = model_tensors[-1][0].shape[1]*model_tensors[-1][0].shape[2]
                output_feature = tf.reshape(model_tensors[-1][0],[-1,total_dim])
            else:
                output_feature = model_tensors[-1][0]
            tf.debugging.check_numerics(output_feature ,message='output_featureis nan')
            model_tensors.append(MLP(input_spec=tf.TensorSpec(shape=output_feature.shape, dtype=tf.float32),
                        model_spec=bag_spec, output_activation=final_act,bn=True)(output_feature))
            model_tensors.append(tf.identity(model_tensors[-1], name="sample_prediction"))

        elif bag_encoder_type == 'ratio':
            if len(model_tensors[-1][0].shape) == 3:
                if model_tensors[-1][0].shape[-1] == 1:
                    output_feature = tf.squeeze(model_tensors[-1][0],axis = 2)
                else:
                    output_feature = tf.squeeze(model_tensors[-1][0],axis = 1)
            else:
                output_feature = model_tensors[-1][0]
            output_feature = output_feature / tf.reduce_sum(output_feature,axis = -1,keepdims = True)
            model_tensors.append(tf.identity(output_feature, name="sample_prediction"))            

        elif bag_encoder_type == 'class_specific_mlp':
            output_feature = model_tensors[-1][0]
            projected_feature = [tf.keras.layers.Dense(units=1, activation=None,kernel_initializer=initializer,use_bias = False)(output_feature[...,i:i+1,:]) for i in range(output_feature.shape[1])]
            projected_feature = tf.squeeze(tf.concat(projected_feature,axis = -1),axis=1)

            if final_act == 'softmax':
                projected_feature = tf.nn.softmax(projected_feature,axis = -1)
            elif final_act == 'sigmoid':
                projected_feature = tf.nn.sigmoid(projected_feature)
            elif final_act == 'none':
                projected_feature = projected_feature
            model_tensors.append(tf.identity(projected_feature, name="sample_prediction"))
        
        # build keras model
        if instance_supervision:
            super(FlowMIL, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1],model_tensors[1]])
            # self.compile = self.custom_compile
            self.train_step = self.custom_train_step
            self.test_step = self.custom_test_step
            self.grouping_variables()
        
        elif att_regularization_method is not None:
            att_target = tf.identity(att_target, name="attention target")
            super(FlowMIL, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1],att_target])

        else:
            super(FlowMIL, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1]])

    def get_config(self):
        #save the model settings in config
        cfg = self.config
        return cfg
    def weights_init(self,seed=42):
        np.random.seed(seed)
    
    def custom_compile(self, loss, optimizer, loss_weights=None):
        self.loss = loss
        self.optimizers = optimizer
        self.loss_weights = loss_weights
        self.loss_trackers = [tf.keras.metrics.Mean(name='weighted_loss'),
                                    tf.keras.metrics.Mean(name='loss_bag'),
                                    tf.keras.metrics.Mean(name='loss_ins')]

    def grouping_variables(self):
        self.inst_variables = self.get_layer('map_tensor_function_ragged').trainable_variables
        self.bag_variables = [x for x in self.trainable_variables if x.name not in [y.name for y in self.inst_variables]]
        
    def custom_train_step(self,data):
        '''stop the gradient from bag classifier for the instance classifier'''
        feature, (y,y_ins) = data
        with tf.GradientTape(persistent=True) as tape:
            y_pred,y_ins_pred = self(feature, training=True)
            bag_loss = self.loss[0](y, y_pred)
            ins_loss = self.loss[1](y_ins, y_ins_pred)
        bag_gradients = tape.gradient(bag_loss, self.bag_variables)
        ins_gradients = tape.gradient(ins_loss, self.inst_variables)    
        self.optimizers[0].apply_gradients(zip(bag_gradients, self.bag_variables))
        self.optimizers[1].apply_gradients(zip(ins_gradients, self.inst_variables))
        self.loss_trackers[0].update_state(self.loss_weights[0]*bag_loss + self.loss_weights[1]*ins_loss)
        self.loss_trackers[1].update_state(bag_loss)
        self.loss_trackers[2].update_state(ins_loss)
        return {'weighted_loss':self.loss_trackers[0].result(),
                'loss_bag':self.loss_trackers[1].result(),'loss_ins':self.loss_trackers[2].result()}
    
    def custom_test_step(self,data):
        feature, (y,y_ins) = data
        y_pred,y_ins_pred = self(feature, training=False)
        bag_loss = self.loss[0](y, y_pred)
        ins_loss = self.loss[1](y_ins, y_ins_pred)
        self.loss_trackers[0].update_state(self.loss_weights[0]*bag_loss + self.loss_weights[1]*ins_loss)        
        self.loss_trackers[1].update_state(bag_loss)
        self.loss_trackers[2].update_state(ins_loss)
        return {'weighted_loss':self.loss_trackers[0].result(),
                'loss_bag':self.loss_trackers[1].result(),'loss_ins':self.loss_trackers[2].result()}

    def inst_classifier_loading(self,args):        
        self.get_layer('map_tensor_function_ragged').fn.load_weights(args.model_loading_dir+'/model')
        self.get_layer('map_tensor_function_ragged').fn.trainable = False

# multitube MIL framework for flow cytometry learning
class FlowMIL_multi_tubes(tf.keras.Model):
    '''
    init args:
    input_spec: input specification, we assume this is the same across tubes in current version of code
    instance_weights_spec: network specification for instance(cellular) level network
    bag_spec: sample level network specification
    output_spec: number of classes in output
    instance_mlps_spec: layer specification for instance mlp
    block_settings: list of MIL framework setting, with format [instance encoder type, instance attention & aggregation strategy, bag-level network type]
    final_act: final activation function for bag(sample) level network
    instance_final_act: final activation function for instance level attention network
    instance_att_act: activation function for instance level hidden layers
    pre_normalization: whether normalize the instance attention or after the weighted sum, refer to AttentionBasedAggregation
    att_normalization: instance attention normalization strategy, refer to AttentionBasedAggregation
    att_regularization_method: optional attention regularization target
    ashbin: number of attention heads not influced by the sample-level prediction directly
    across_tube_aggr: methods to aggregate tube-level feature
    '''
    def __init__(self, 
                input_spec: tuple = None,
                instance_weights_spec: list[list] = None,
                bag_spec: list = None,
                output_spec: int = 2,
                instance_mlps_spec: list[list] = None,
                block_settings: list = None,
                final_act: str = 'softmax',
                instance_final_act: str = 'softplus',
                instance_att_act: str = 'sigmoid',
                pre_normalization: bool =False,
                att_normalization: str = 'normalize_along_instance',
                att_regularization_method: str = None,
                ashbin: int = 0,
                across_tube_aggr: str = 'mean',
                seed=42,**kwargs):
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.config = {'input_spec':input_spec,   'instance_weight_spec':instance_weights_spec, 'bag_spec':bag_spec, 'output_spec':output_spec, 'instance_mlp_spec':instance_mlps_spec,
                       'block_settings':block_settings, 'final_act':final_act, 'instance_final_act':instance_final_act}
        
        n_tubes = len(instance_mlps_spec)
        model_tensors = [[tf.keras.layers.Input( shape = input_spec, dtype = tf.float32, ragged=True) for i in range(n_tubes)]]
        
        #part 1 instance encoder
        instance_encoder_type_list = ['MLP','none']
        
        instance_encoder_type = block_settings[0]
        if instance_encoder_type == 'MLP':
            multi_tube_inst_features = []
            for _ in range(n_tubes):
                multi_tube_inst_features.append(MapTensorFunctionRagged(MLP(input_spec=tf.TensorSpec(shape=model_tensors[-1][_].shape[1:], dtype=tf.float32),
                                                                        model_spec=instance_mlps_spec[_], output_activation=instance_final_act,bn=True,seed=seed))(model_tensors[-1][_]))
            model_tensors.append(multi_tube_inst_features)
        elif instance_encoder_type == 'none':
            pass

        #part 2 instance attention and instance aggregation
        attention_weighted_average_list = ['mean','attention_dependent','attention_independent','sum']
        attention_computation_types = ['normalize_along_instance','normalize_along_both','normalize_based_on_instance']
        attention_weighted_average = block_settings[1]
        assert attention_weighted_average in attention_weighted_average_list, 'provided aggregation strategy is not supported'
        assert (att_normalization in attention_computation_types) + (att_normalization is None), 'provided attention normalization strategy is not sup'

        if attention_weighted_average == 'mean':
            avg_features = []
            for _ in range(n_tubes):
                avg_features.append([tf.reduce_mean(model_tensors[-1][_][:,:,ashbin:],axis = 1,keepdims=True),tf.ones_like(model_tensors[-1][_])[...,0:1]])
            model_tensors.append(avg_features)
        elif attention_weighted_average == 'attention_dependent':
            #(batch_size, num_instances, n_head)
            instance_weights_list = []
            for _ in range(n_tubes):
                instance_weights = MapTensorFunctionRagged(MLP(input_spec=tf.TensorSpec(shape=model_tensors[-1][_].shape[1:], dtype=tf.float32),
                            model_spec=instance_weights_spec[_], output_activation=instance_att_act,
                            output_name='instance_attention',bn=True))(model_tensors[-1][_])[:,:,ashbin:]
                instance_weights = tf.where(tf.math.is_nan(instance_weights), tf.zeros_like(instance_weights), instance_weights)
                instance_weights_list.append(AttentionBasedAggregation(att_normalization = att_normalization,pre_normalization = pre_normalization)([model_tensors[-1][_], instance_weights]))
            model_tensors.append(instance_weights_list)
        elif attention_weighted_average == 'attention_independent':
            #(batch_size, num_instances, n_head)
            instance_weights_list = []
            for _ in range(n_tubes):
                instance_weights = MapTensorFunctionRagged(MLP(input_spec=tf.TensorSpec(shape=model_tensors[0][_].shape[1:], dtype=tf.float32),
                            model_spec=instance_weights_spec[_], output_activation=instance_att_act,
                            output_name='instance_attention',bn=True,seed=seed))(model_tensors[0][_])[:,:,ashbin:]
                instance_weights_list.append(AttentionBasedAggregation(att_normalization = att_normalization, pre_normalization = pre_normalization)([model_tensors[-1][_], instance_weights]))
            model_tensors.append(instance_weights_list)
        elif attention_weighted_average == 'sum':
            sum_features = []
            for _ in range(n_tubes):
                sum_features.append([tf.reduce_sum(model_tensors[-1][_][:,:,ashbin:],axis = 1,keepdims=True),tf.ones_like(model_tensors[-1][_])[...,0:1]])
            model_tensors.append(sum_features)
            
        # part 2.6 aggregation across tubes
        across_tube_aggr_list = ['mean']
        assert across_tube_aggr in across_tube_aggr_list, 'across tube aggregation strategy is not supported'
        if across_tube_aggr == 'mean':
            tube_level_features = tf.concat([x[0] for x in model_tensors[-1]],axis = 1)
            cross_tube_feature = tf.reduce_mean(tube_level_features,axis=1)
            model_tensors.append(cross_tube_feature)

        # part 3 bag encoder
        bag_encoder_type_list = ['none','MLP_single_head','MLP_multiple_head','ratio', 'class_specific_mlp']
        bag_encoder_type = block_settings[2]
        assert bag_encoder_type in bag_encoder_type_list, 'provided bag encoder type is not supported'
        if bag_encoder_type == 'none':
            output_feature = cross_tube_feature
            model_tensors.append(tf.identity(output_feature, name="sample_prediction"))               
        elif bag_encoder_type == 'MLP_single_head':
            model_tensors.append(MLP(input_spec=tf.TensorSpec(shape=cross_tube_feature.shape, dtype=tf.float32),
                        model_spec=bag_spec, output_activation=final_act,bn=True)(cross_tube_feature))
        elif bag_encoder_type == 'MLP_multiple_head':

            total_dim = cross_tube_feature.shape[1]*cross_tube_feature.shape[2]
            output_feature = tf.reshape(cross_tube_feature,[-1,total_dim])
            tf.debugging.check_numerics(output_feature ,message='output_featureis nan')
            model_tensors.append(MLP(input_spec=tf.TensorSpec(shape=output_feature.shape, dtype=tf.float32),
                        model_spec=bag_spec, output_activation=final_act,bn=True)(output_feature))
            model_tensors.append(tf.identity(model_tensors[-1], name="sample_prediction"))
        elif bag_encoder_type == 'ratio':
            output_feature = cross_tube_feature
            output_feature = output_feature / tf.reduce_sum(output_feature,axis = -1,keepdims = True)
            model_tensors.append(tf.identity(output_feature, name="sample_prediction"))
        elif bag_encoder_type == 'class_specific_mlp':
            output_feature = cross_tube_feature
            projected_feature = [tf.keras.layers.Dense(units=1, activation=None,kernel_initializer=initializer,use_bias = False)(output_feature[...,i:i+1,:]) for i in range(output_feature.shape[1])]
            projected_feature = tf.squeeze(tf.concat(projected_feature,axis = -1),axis=1)

            if final_act == 'softmax':
                projected_feature = tf.nn.softmax(projected_feature,axis = -1)
            elif final_act == 'sigmoid':
                projected_feature = tf.nn.sigmoid(projected_feature)
            elif final_act == 'none':
                projected_feature = projected_feature
            model_tensors.append(tf.identity(projected_feature, name="sample_prediction"))
        # build keras model
        if att_regularization_method is not None:
            att_target = tf.identity(att_target, name="attention target")
            super(FlowMIL_multi_tubes, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1],att_target])

        else:
            super(FlowMIL_multi_tubes, self).__init__(inputs=[model_tensors[0]], outputs=[model_tensors[-1]])

    def get_config(self):
        cfg = self.config
        return cfg
    def weights_init(self,seed=42):
        np.random.seed(seed)
    
    def custom_compile(self, loss, optimizer, loss_weights=None):
        self.loss = loss
        self.optimizers = optimizer
        self.loss_weights = loss_weights
        self.loss_trackers = [tf.keras.metrics.Mean(name='weighted_loss'),
                                    tf.keras.metrics.Mean(name='loss_bag'),
                                    tf.keras.metrics.Mean(name='loss_ins')]

    def grouping_variables(self):
        self.inst_variables = self.get_layer('map_tensor_function_ragged').trainable_variables
        self.bag_variables = [x for x in self.trainable_variables if x.name not in [y.name for y in self.inst_variables]]
        
    def custom_train_step(self,data):
        feature, (y,y_ins) = data
        with tf.GradientTape(persistent=True) as tape:
            y_pred,y_ins_pred = self(feature, training=True)
            bag_loss = self.loss[0](y, y_pred)
            ins_loss = self.loss[1](y_ins, y_ins_pred)
        bag_gradients = tape.gradient(bag_loss, self.bag_variables)
        ins_gradients = tape.gradient(ins_loss, self.inst_variables)    
        self.optimizers[0].apply_gradients(zip(bag_gradients, self.bag_variables))
        self.optimizers[1].apply_gradients(zip(ins_gradients, self.inst_variables))
        self.loss_trackers[0].update_state(self.loss_weights[0]*bag_loss + self.loss_weights[1]*ins_loss)
        self.loss_trackers[1].update_state(bag_loss)
        self.loss_trackers[2].update_state(ins_loss)
        return {'weighted_loss':self.loss_trackers[0].result(),
                'loss_bag':self.loss_trackers[1].result(),'loss_ins':self.loss_trackers[2].result()}
    def custom_test_step(self,data):
        feature, (y,y_ins) = data
        y_pred,y_ins_pred = self(feature, training=False)
        bag_loss = self.loss[0](y, y_pred)
        ins_loss = self.loss[1](y_ins, y_ins_pred)
        self.loss_trackers[0].update_state(self.loss_weights[0]*bag_loss + self.loss_weights[1]*ins_loss)        
        self.loss_trackers[1].update_state(bag_loss)
        self.loss_trackers[2].update_state(ins_loss)
        return {'weighted_loss':self.loss_trackers[0].result(),
                'loss_bag':self.loss_trackers[1].result(),'loss_ins':self.loss_trackers[2].result()}
    def inst_classifier_loading(self,args):        
        self.get_layer('map_tensor_function_ragged').fn.load_weights(args.model_loading_dir+'/model')
        self.get_layer('map_tensor_function_ragged').fn.trainable = False

# func to grab layers for each layers of FlowMIL     
def grab_mid_layers(model,layer_name: list[str]):
    """
    used when tf.keras.Model refused to work
    """
    layers = []
    name_list = [layer.name for layer in model.layers]
    layer_ind = name_list.index(layer_name)
    for lind in range(layer_ind+1):
        layers.append(model.get_layer(name_list[lind]))
    return layers

# func to grab layers for each layers of FlowMIL_multi_tube
def grad_mid_layers_multi_tube(model,layer_name: list[str],n_tube: int):
    tube_layers = []
    for nt in range(n_tube):
        current_tube_layers = []
        if nt == 0:
            current_tube_layers.append(model.get_layer(layer_name))
        else:
            current_tube_layers.append(model.get_layer(layer_name+'_'+str(nt)))
        tube_layers.append(current_tube_layers)
    return tube_layers

# func to grab the outputs from the layer, single tube
def grab_mid_outputs(layers,data):    
    if isinstance(data,tf.RaggedTensor):
        outputs = [data]
        for layer in layers:
            outputs.append(layer(outputs[-1]))
    if isinstance(data,tf.data.Dataset):
        outputs = []
        for d in data:
            out = [d]
            for layer in layers:
                out.append(layer(out[-1]))
            outputs.append(out)
    if isinstance(data,tf.Tensor):
        outputs = [data]
        for layer in layers:
            outputs.append(layer(outputs[-1]))
    return outputs

# optional, pretraing for instance encoder with SCARF, used in "Automated Deep Learning-Based Diagnosis and Molecular Characterization of Acute Myeloid Leukemia Using Flow Cytometry"
class SCARF_pretraining(tf.keras.Model):
    def __init__(self, input_spec, proj_spec, plr_spec, mlp1_spec,mlp2_spec, 
                 feature_low,feature_high, corruption_rate = 0.5,input_dim = 4, **kwargs):
        super(SCARF_pretraining, self).__init__(**kwargs)
        self.corruption_len = int(corruption_rate * input_dim)
        self.encoder = MLP_PLR(input_spec=tf.TensorSpec(shape=input_spec, dtype=tf.float32),
                                                        plr_spec=plr_spec, mlp_spec=mlp1_spec)
        self.encoder_proj = MLP(input_spec=tf.TensorSpec(shape=proj_spec, dtype=tf.float32),
                                                        model_spec=mlp2_spec)
        self.feature_low = feature_low
        self.feature_high = feature_high
        self.corruption_rate = corruption_rate
    def call(self, model_input):
        # if ragged, transform model_input to flattened tensor
        if isinstance(model_input, tf.RaggedTensor):
            model_input = model_input.flat_values
        #set model input dtype tf.float32
        model_input = tf.cast(model_input,tf.float32)
        n_instances, n_features = model_input.shape[0], model_input.shape[1]
        corruption_mask = tf.random.uniform(shape = tf.shape(model_input), minval = 0, maxval = 1)
        #generate an ind of size (n_instances,n_features) with values in range(n_features)
        channel_index = tf.tile(tf.range(n_features)[tf.newaxis,:],[tf.shape(model_input)[0],1])
        corruption_mask = tf.argsort(corruption_mask,axis = 1,direction = 'DESCENDING')[:,:self.corruption_len]
        #generate random metric with the same size as model_input and in the range of [feature_low,feature_high]
        # random_metric = tf.random.uniform(shape = model_input.shape, minval = self.feature_low, maxval = self.feature_high)
        random_metric = tf.random.uniform(shape = tf.shape(model_input), minval = self.feature_low, maxval = self.feature_high)
        #generate corrupted input by replacing feature in pos channel_index with random_metric
        batch_indices = tf.repeat(tf.range(tf.shape(model_input)[0])[:, tf.newaxis,tf.newaxis], self.corruption_len, axis=1)
        # Expand dimensions of corruption_mask to use in gather_nd
        expanded_corruption_mask = tf.expand_dims(corruption_mask, 2)
        # Concatenate batch indices with corruption_mask to get the indices for gather_nd
        indices_for_gather = tf.concat([batch_indices, expanded_corruption_mask], axis=2)
        # Gather values from random_metric at specified indices
        random_values_to_replace = tf.gather_nd(random_metric, indices_for_gather)
        expanded_random_values = tf.scatter_nd(indices_for_gather, random_values_to_replace, tf.shape(model_input))
        # Create a mask of zeros with the same shape as model_input
        mask = tf.scatter_nd(indices_for_gather, tf.ones_like(random_values_to_replace), tf.shape(model_input))
        # Replace the selected features in model_input with random_metric
        corrupted_input = model_input * (1 - mask) + expanded_random_values * mask

        #encode model_input and corrupted_input
        encoded_input = self.encoder(model_input)
        encoded_input_proj = self.encoder_proj(encoded_input)
        encoded_corrupted_input = self.encoder(corrupted_input)
        encoded_corrupted_input_proj = self.encoder_proj(encoded_corrupted_input)
        
        return encoded_input_proj, encoded_corrupted_input_proj
    def get_config(self):
        cfg = super().get_config()
        return cfg
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            encoded_input_proj, encoded_corrupted_input_proj = self.call(data)
            loss = contrastive_loss(encoded_input_proj, encoded_corrupted_input_proj)
            loss_val = tf.reduce_mean(loss)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        #only return loss for now
        return {'loss': loss_val}
    
    def validation_step(self, data):
        encoded_input_proj, encoded_corrupted_input_proj = self.call(data)
        loss = contrastive_loss(encoded_input_proj, encoded_corrupted_input_proj)
        return {'val_loss': tf.reduce_mean(loss).numpy()}

# regularization for the centroid distances
@tf.function
def mean_regularizer(w):
    dist = tf.reduce_sum(tf.square(w[:,tf.newaxis,:] - w[:,:,tf.newaxis]),axis = 0)
    square_norm = tf.reduce_sum(tf.square(w)) * (w.shape[1]-1)
    return square_norm / tf.reduce_sum(dist*(tf.ones_like(dist)-tf.eye(w.shape[1]))) *100

# regularization for the mixture components
@tf.function
def component_regularizer(w):
    sum_to_1 = (1-tf.reduce_sum(w))**2
    non_neg = tf.reduce_sum(tf.nn.relu(-w))**2
    return (sum_to_1 + non_neg)*1000

# use RBF layers to generate instance logits
class GaussianRadialBasisLayer(tf.keras.layers.Layer):
    def __init__(self, units,use_chol=True,return_kld=False,initialization_info=None,trainable=True,phase='clustering', **kwargs):
        super(GaussianRadialBasisLayer, self).__init__(**kwargs)
        self.units = units
        self.ln2pi_2 = tf.constant(np.log(2 * np.pi) / 2, dtype=tf.float32)
        self.use_chol,self.return_kld = use_chol,return_kld
        self.initialization_info = initialization_info
        self.trainable = trainable
        self.phase = phase
    def build(self, input_shape):
        if isinstance(self.units, int):
            if self.initialization_info is not None:
                self.l = self.add_weight(name='means', shape=(input_shape[-1], self.units), dtype=tf.float32, trainable=self.trainable , initializer=tf.keras.initializers.Constant(value = self.initialization_info['mean'].T))
                if not self.use_chol:
                    self.s = self.add_weight(name='stds_log', shape=(input_shape[-1], self.units), dtype=tf.float32, trainable=self.trainable , initializer=tf.keras.initializers.Constant(value = self.initialization_info['stds_log'].T))
                else:
                    self.s = self.add_weight(name='chol', shape=(input_shape[-1], self.units), dtype=tf.float32, trainable=self.trainable , initializer=tf.keras.initializers.Constant(value = self.initialization_info['std'].T))
            else:
                self.l = self.add_weight(name='means', shape=(input_shape[-1], self.units), dtype=tf.float32, trainable=self.trainable, initializer=tf.keras.initializers.GlorotUniform(),regularizer = None)
                if self.use_chol==False:
                    self.s = self.add_weight(name='stds_log', shape=(input_shape[-1], self.units), dtype=tf.float32, trainable=self.trainable , initializer=tf.keras.initializers.GlorotUniform(),regularizer = None)
                else:
                    # in linear space instead of log space
                    self.s = self.add_weight(name='chol', shape=(input_shape[-1], self.units), dtype=tf.float32, trainable=self.trainable , initializer=tf.keras.initializers.GlorotUniform(seed=321),regularizer = None)
        elif isinstance(self.units, list):
            assert len(self.units) == 2
            self.l = self.add_weight(name='means', initializer=tf.keras.initializers.Constant(self.units[0]) , dtype=tf.float32, trainable=self.trainable )
            self.s = self.add_weight(name='stds_log', initializer=tf.keras.initializers.Constant(self.units[1]) , dtype=tf.float32, trainable=self.trainable )
            
    def call(self, inputs):
        if self.phase == 'clustering':
            inputs = tf.reshape(inputs,[-1,inputs.shape[-1]])
            if self.use_chol:
                gmm_embedding_log = tf.square((inputs[..., tf.newaxis] - self.l[tf.newaxis, ...]) / self.s[tf.newaxis, ...])
                gmm_embedding_log = (-0.5 * tf.reduce_sum(gmm_embedding_log, axis=-2)) - 0.5 * tf.reduce_sum(tf.math.log(tf.square(self.s)), axis=0, keepdims=True) - (inputs.shape[-1] * self.ln2pi_2)
                if self.return_kld:
                    kld = self.compute_kld()
                    return gmm_embedding_log,kld
                else:
                    return gmm_embedding_log
            else:
                gmm_embedding_log = tf.square((inputs[..., tf.newaxis] - self.l[tf.newaxis, ...]) / tf.math.exp(self.s)[tf.newaxis, ...])
                gmm_embedding_log = (-0.5 * tf.reduce_sum(gmm_embedding_log, axis=-2)) - tf.reduce_sum(self.s, axis=0, keepdims=True) - (inputs.shape[-1] * self.ln2pi_2)
                return gmm_embedding_log
        elif self.phase == 'classification':
            if self.use_chol:
                #inputs (bs,ins,f,1) l (1,1,f,nh), gmm_embedding_log_1 (bs,ins,f,nh)
                gmm_embedding_log = tf.square((inputs[..., tf.newaxis] - self.l[tf.newaxis,tf.newaxis, ...]) / self.s[tf.newaxis,tf.newaxis, ...])
                gmm_embedding_log = (-0.5 * tf.reduce_sum(gmm_embedding_log, axis=-2)) - 0.5 * tf.reduce_sum(tf.math.log(tf.square(self.s)), axis=0, keepdims=True) - (inputs.shape[-1] * self.ln2pi_2)

            else:
                # gmm_embedding_log = inputs[..., tf.newaxis] 
                gmm_embedding_log = tf.square((inputs[..., tf.newaxis] - self.l[tf.newaxis,tf.newaxis, ...]) / tf.math.exp(self.s)[tf.newaxis,tf.newaxis, ...])
                gmm_embedding_log = (-0.5 * tf.reduce_sum(gmm_embedding_log, axis=-2)) - tf.reduce_sum(self.s, axis=0, keepdims=True)[tf.newaxis,...] - (inputs.shape[-1] * self.ln2pi_2)
                
            return gmm_embedding_log
    def compute_kld(self):
        #compute kld for regularization
        assert self.use_chol
        if self.use_chol:
            mu = tf.expand_dims(self.l, axis=2)  # (feature, n_component, 1)
            sigma_diag = tf.expand_dims(self.s, axis=2)  # (feature, n_component, 1)

            mu2 = tf.transpose(mu, perm=[0, 2, 1])  # (feature, 1, n_component)
            sigma_diag2 = tf.transpose(sigma_diag, perm=[0, 2, 1])  # (feature, 1, n_component)

            # Compute all pairwise differences for means and std devs
            diff_mu = mu - mu2  # (feature, n_component, n_component)
            ratio_sigma_diag = sigma_diag / sigma_diag2  # (feature, n_component, n_component)

            # Compute the determinant terms for all pairs
            log_det_sigma = tf.math.log(sigma_diag) - tf.math.log(sigma_diag2)
            log_det_ratio = tf.reduce_sum(log_det_sigma, axis=0)  

            # Compute the trace terms for all pairs
            trace_term = tf.reduce_sum(ratio_sigma_diag, axis=0)  

            # Compute the quadratic terms for all pairs
            quadratic_term = tf.reduce_sum(diff_mu * (1 / sigma_diag2) * diff_mu, axis=0)  
            # Dimensionality term
            p = tf.cast(tf.shape(self.l)[0], dtype=tf.float32)

            # Sum the KLD components for all pairs, excluding the diagonal
            kld_matrix = 0.5 * (log_det_ratio + trace_term + quadratic_term - p)
            mask = 1 - tf.eye(num_rows=tf.shape(kld_matrix)[0], dtype=tf.float32)  
            kld_sum = tf.reduce_sum(kld_matrix * mask)
            return -kld_sum
   
# use MLP layers to generate instance logits
class GaussianMLPBasisLayer(tf.keras.layers.Layer):
    def __init__(self,input_spec,instance_mlp_spec,instance_final_act,seed=42,phase = 'clustering'):
        super(GaussianMLPBasisLayer, self).__init__()
        self.instance_att_encoder = MapTensorFunctionRagged(MLP(input_spec=tf.TensorSpec(shape=input_spec, dtype=tf.float32),
                                                model_spec=instance_mlp_spec, output_activation=instance_final_act,bn=False,seed=seed))
        self.phase = phase
    def call(self,input):
        instance_att = self.instance_att_encoder(input)
        
        if self.phase == 'clustering':
            input_feature_flatten = tf.reshape(input,[-1,input.shape[-1]])
            instance_att_flatten = tf.reshape(instance_att,[-1,instance_att.shape[-1]])
            features_weighted_sum = tf.divide(tf.reduce_sum(input_feature_flatten[:,tf.newaxis,:] * instance_att_flatten[:,:,tf.newaxis],axis=0),tf.reduce_sum(instance_att_flatten,axis=0)[:,tf.newaxis])
            # (ins, f) , (ins, nhead) -> (nhead, f)
            features_var = tf.divide(tf.reduce_sum(tf.square(input_feature_flatten[:,tf.newaxis,:]-features_weighted_sum[tf.newaxis,:,:]) * instance_att_flatten[:,:,tf.newaxis],axis = 0), tf.reduce_sum(instance_att_flatten,axis=0)[:,tf.newaxis])
            gmm_embedding_log = self.gmm_embedding_log_func(input_feature_flatten,features_weighted_sum,features_var)
            return gmm_embedding_log, features_var
        elif self.phase == 'classification':
            # (bs,ins,f), (bs,ins,att) -> (att,f)
            # (bs,ins,1,f)*(bs,ins,att,1)->(bs,ins,att,f)->(att,f)
            features_weighted_sum = tf.divide(tf.reduce_sum(tf.reduce_sum(input[...,tf.newaxis,:] * instance_att[...,tf.newaxis],axis=1),axis = 0),tf.reduce_sum(tf.reduce_sum(instance_att,axis=1),axis = 0)[...,tf.newaxis])
            # (bs,ins,f),(att,f),(bs,ins,att)->(bs,ins,att,f) ->(att,f)
            # (bs,ins,att,f)
            features_var = tf.divide(tf.reduce_sum(tf.reduce_sum(tf.square(input[:,:,tf.newaxis,:]-features_weighted_sum[tf.newaxis,tf.newaxis,:,:]) * instance_att[:,:,:,tf.newaxis],axis = 1),axis = 0), tf.reduce_sum(tf.reduce_sum(instance_att,axis=1),axis=0)[:,tf.newaxis])
            gmm_embedding_log = self.gmm_embedding_log_func(input,features_weighted_sum,features_var)
            return gmm_embedding_log, features_var            
    def mean_var_computation(self,input):
        instance_att = self.instance_att_encoder(input)
        input_feature_flatten = tf.reshape(input,[-1,input.shape[-1]])
        instance_att = tf.reshape(instance_att,[-1,instance_att.shape[-1]])
        #(ins, f) , (ins, nhead) -> (nhead, f)
        features_weighted_sum = tf.reduce_sum(input_feature_flatten[:,tf.newaxis,:] * instance_att[:,:,tf.newaxis],axis=0)/tf.reduce_sum(instance_att,axis=0)[:,tf.newaxis]
        # (ins, f) , (ins, nhead) -> (nhead, f, f)
        features_var = tf.reduce_sum(tf.square(input_feature_flatten[:,tf.newaxis,:]-features_weighted_sum[tf.newaxis,:,:]) * instance_att[:,:,tf.newaxis],axis = 0)/tf.reduce_sum(instance_att,axis=0)[:,tf.newaxis]
        return features_weighted_sum,features_var
    def gmm_embedding_log_func(self,features,means,vars):
        if len(features.shape)==2:
            ln2pi_2 = tf.constant(np.log(2 * np.pi) / 2, dtype=tf.float32)
            # (ins,f), (f,nh)
            gmm_embedding_log = tf.square(features[:,tf.newaxis,:] - means[tf.newaxis, :,:]) / vars[tf.newaxis, ...]
            gmm_embedding_log = (-0.5 * tf.reduce_sum(gmm_embedding_log, axis=-1)) - 0.5 * tf.reduce_sum(tf.math.log(vars), axis=-1) - (features.shape[-1] * ln2pi_2)
        else:
            ln2pi_2 = tf.constant(np.log(2 * np.pi) / 2, dtype=tf.float32)
            #features (bs,ins,f), mean (f,nh)
            # (bs,ins,1,f), (1,1,f,nh)
            gmm_embedding_log = tf.divide(tf.square(features[:,:,tf.newaxis,:] - means[tf.newaxis,tf.newaxis, :,:]) , vars[tf.newaxis,tf.newaxis, ...])
            gmm_embedding_log = (-0.5 * tf.reduce_sum(gmm_embedding_log, axis=-1)) - 0.5 * tf.reduce_sum(tf.math.log(vars), axis=-1) - (features.shape[-1] * ln2pi_2)            
        return gmm_embedding_log

# constraints on mixtures
class WeightsSumToOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.keras.activations.softmax(w, axis=0)

# constraints on mixtures
class SumToOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        w = tf.clip_by_value(w, 1e-4, 1)
        return w / tf.reduce_sum(w, axis=0, keepdims=True)

# GMM for cellular clustering and classification 
class GMM(tf.keras.Model):
    '''
    input_spec: input specification
    unit: number of centroids
    GRF_encoding_method: Gaussian Radial Field encoding method
    initialization_info: GMM parameters initialization strategy
    instance_final_act: instance logits activation function
    phase: whether training GMM through clustering or use the GMM to encode the FCM samples
    maximum_normalization: normalized the log_logits with its maximal value to ensure the numerical stability, which is generally larger than 0 in our case, used by cautious
    logit_threshold: upper bound of instance logits, used in the feature extraction process
    return_features_var: whether return computed variance for optimization
    classification_head_spec: sample level classification specification
    cluster_model_trainable: whether the clustering model is trainable
    inst_logits_normalization: whether the instance logits is normalized again for classification
    '''
    def __init__(self,
                input_spec: tuple = None,
                unit: int = 16,
                GRF_encoding_method: str = 'GRF',
                initialization_info: dict = None,
                instance_final_act: str = None,
                phase: str = 'clustering',
                maximum_normalization: bool = False,
                logit_threshold: int = None,
                return_features_var: bool = False, 
                classification_head_spec: bool = None,
                cluster_model_trainable: bool = True,
                inst_logits_normalization: bool = True):
        
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        input_tensor = tf.keras.layers.Input(type_spec=input_spec)
        self.logit_threshold = logit_threshold
        if GRF_encoding_method == 'GRF':
            embedding_layer = GaussianRadialBasisLayer(units=unit,initialization_info=initialization_info,phase=phase,trainable=cluster_model_trainable)
            rbf_embedding_log = embedding_layer(input_tensor)
            return_features_var = False
        elif GRF_encoding_method == 'mlp':
            if instance_final_act ==None:
                instance_final_act = 'softmax'
                embedding_layer = GaussianMLPBasisLayer(input_spec=input_tensor.shape[1:],instance_mlp_spec=[16,16,unit],instance_final_act=instance_final_act,phase = phase)
            if phase == 'clustering':
                rbf_embedding_log,features_var = embedding_layer(input_tensor)
            elif phase == 'classification':
                inst_logits = embedding_layer.instance_att_encoder(input_tensor)
        if maximum_normalization:
            regularization_term = tf.reduce_max(rbf_embedding_log)
            rbf_embedding_log = rbf_embedding_log - regularization_term    
        if phase == 'clustering':    
            if initialization_info is not None:
                gmm_prob = tf.keras.layers.Dense(units=1, use_bias=False, activation=None,
                                            kernel_initializer=tf.keras.initializers.Constant(value=initialization_info['mixture_coef']),
                                            kernel_regularizer = component_regularizer,
                                            )(tf.math.exp(rbf_embedding_log))
            else:
                gmm_prob = tf.keras.layers.Dense(units=1, use_bias=False, activation=None,
                                                # kernel_constraint=WeightsSumToOne(),
                                                # kernel_constraint=SumToOne(),
                                                kernel_initializer=tf.keras.initializers.Constant(value=1/unit),
                                                # kernel_initializer=tf.keras.initializers.Ones(),
                                                # kernel_regularizer = tf.keras.regularizers.L2(-0),
                                                kernel_regularizer = component_regularizer,
                                                # kernel_regularizer = None,
                                                )(tf.math.exp(rbf_embedding_log))
            gmm_prob = tf.identity(gmm_prob)
            rbf_embedding_log = tf.identity(rbf_embedding_log)

            if return_features_var:
                features_var = tf.identity(features_var)
                super(GMM, self).__init__(inputs=[input_tensor], outputs=[gmm_prob, rbf_embedding_log,features_var])
            else:
                super(GMM, self).__init__(inputs=[input_tensor], outputs=[gmm_prob, rbf_embedding_log])
        elif phase == 'classification':
            
            if GRF_encoding_method == 'GRF':
                inst_logits = tf.clip_by_value(rbf_embedding_log,-128,3)
                inst_logits = tf.exp(inst_logits)
                if inst_logits_normalization:
                    inst_logits = tf.divide(inst_logits ,tf.reduce_sum(inst_logits,axis=-1,keepdims=True))
                else:
                    inst_logits = self.ragged_z_transform(inst_logits)
            elif GRF_encoding_method == 'mlp':
                inst_logits = inst_logits
            inst_logits = tf.reduce_mean(inst_logits,axis = -2)
            sample_prediction = MLP(input_spec=tf.TensorSpec(shape=(None,unit), dtype=tf.float32),
                        model_spec=classification_head_spec, output_activation='softmax',bn=True)(inst_logits)
            sample_prediction = tf.identity(sample_prediction)
            super(GMM, self).__init__(inputs=[input_tensor], outputs=[sample_prediction])
        elif phase == 'feature_extraction':
            assert initialization_info is not None, 'embedding layer should be initialized first'
            embedding_layer.trainable = False
            density_estimation = self.bag_density_estimation(tf.math.exp(rbf_embedding_log),initialization_info['mixture_coef'].squeeze())
            super(GMM, self).__init__(inputs=[input_tensor], outputs=[density_estimation])
    def get_config(self):
        cfg = super().get_config()
        return cfg
    def weights_init(self,seed=42):
        np.random.seed(seed)
    def bag_density_estimation(self,features,mixture_coeffcients):
        features = tf.clip_by_value(features,0,self.logit_threshold)
        features = features * mixture_coeffcients[tf.newaxis,tf.newaxis,:]
        bag_density = tf.reduce_sum(features,axis=1)
        bag_density = bag_density / tf.reduce_sum(bag_density,axis=-1,keepdims=True)
        return bag_density


    def ragged_z_transform(self,inst_logits):
        inst_mean = tf.reduce_mean(tf.reduce_mean(inst_logits,axis = 0),axis = 0) 
        inst_std = (tf.reduce_mean(tf.reduce_mean(tf.square(inst_logits - inst_mean),axis = 0),axis = 0) )**(0.5)
        normalized_inst_logits = tf.math.divide_no_nan(inst_logits - inst_mean ,inst_std)
        return normalized_inst_logits 

# instance feature extraction for AML dataset structures
def inst_feature_extraction(model,dataset,saving_dir,mode='inst'):
    if mode == 'inst':
        os.makedirs(saving_dir,exist_ok=True)
        i = 0
        for data in dataset:
            sample_saving_dir = os.path.join(saving_dir,'sample_'+str(i)+'.h5')
            sample_feature = model(data[0])
            with h5py.File(sample_saving_dir, 'w') as hf:
                hf.create_dataset('instance_label', data=sample_feature.flat_values.numpy())
                hf.create_dataset('sample_labels', data=tf.argmax(data[1],axis=1).numpy().squeeze())
                hf.create_dataset('mean_instance', data=tf.reduce_mean(sample_feature,axis=1).numpy())
            i += 1
            
    elif mode == 'bag':
        inst_mean = []
        inst_feature = []
        bag_labels = []
        inst_feature_extractor = MapTensorFunctionRagged(tf.keras.Model(inputs=model.fn.input, outputs=model.fn.get_layer('dense_2').output))
        for data in dataset:
            inst_mean.append(tf.reduce_mean(model(data[0]),axis=1).numpy())
            inst_feature.append(tf.reduce_mean(inst_feature_extractor(data[0]),axis=1).numpy())
            bag_labels.append(tf.argmax(data[1],axis=1).numpy().squeeze())
        inst_mean = np.array(inst_mean)
        bag_labels = np.array(bag_labels)
        inst_feature = np.array(inst_feature)
        with h5py.File(saving_dir, 'w') as hf:
            hf.create_dataset('instance_label', data=inst_mean.squeeze())
            hf.create_dataset('sample_labels', data=bag_labels)
            hf.create_dataset('inst_feature', data=inst_feature.squeeze())
            
# cellular attention regularization target, from Alex Baras, the score is computed with the idea of f-score
def inv_f_stat(inputs, ins_att, weight = 0.1):
    '''
    input:
    inputs: b x ins x f
    att: b x ins x nhead
    output:
    computed inv_f_stat value
    '''
    batch_attention_sums_unit = tf.reduce_sum(ins_att.flat_values, axis=0)
    batch_cluster_means_unit = tf.divide(tf.tensordot(ins_att.flat_values, inputs.flat_values, [0, 0]) , batch_attention_sums_unit[:, tf.newaxis])
    batch_cluster_vars_unit =tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_mean(tf.math.squared_difference(inputs.flat_values[:, tf.newaxis, :], batch_cluster_means_unit[tf.newaxis, :, :]), axis=-1) , ins_att.flat_values), axis=0) , batch_attention_sums_unit)
    batch_attention_sum_unit = tf.reduce_sum(batch_attention_sums_unit)
    batch_global_mean_unit = tf.divide(tf.tensordot(batch_attention_sums_unit[:, tf.newaxis], batch_cluster_means_unit, [0, 0]) , batch_attention_sum_unit)
    batch_across_var_unit = tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_mean(tf.math.squared_difference(batch_cluster_means_unit, batch_global_mean_unit), axis=-1) , batch_attention_sums_unit)) , batch_attention_sum_unit)
    batch_within_var_unit = tf.divide(tf.reduce_sum(tf.multiply(batch_cluster_vars_unit , batch_attention_sums_unit)) , batch_attention_sum_unit)
    inv_f_stat_unit = tf.divide(batch_within_var_unit , batch_across_var_unit)
    return inv_f_stat_unit * weight

# cellular attention regularization target, variances on cluster level
def batch_global_var(inputs, ins_att, weight = 0.1):
    '''
    input:
    inputs: b x ins x f
    att: b x ins x nhead
    output:
    computed inv_f_stat value
    '''
    batch_attention_sums_unit = tf.reduce_sum(ins_att.flat_values, axis=0)
    batch_cluster_means_unit = tf.divide(tf.tensordot(ins_att.flat_values, inputs.flat_values, [0, 0]) , batch_attention_sums_unit[:, tf.newaxis])
    batch_attention_sum_unit = tf.reduce_sum(batch_attention_sums_unit)
    batch_global_mean_unit = tf.divide(tf.tensordot(batch_attention_sums_unit[:, tf.newaxis], batch_cluster_means_unit, [0, 0]) , batch_attention_sum_unit)
    batch_across_var_unit = tf.sqrt(tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_mean(tf.math.squared_difference(batch_cluster_means_unit, batch_global_mean_unit), axis=-1) , batch_attention_sums_unit)) , batch_attention_sum_unit))

    return -batch_across_var_unit * weight

# cellular attention regularization target, cluster centroids Euclidean distance (clipped)
def centroid_dist_euclidean(inputs, ins_att, weight = 0.1):
    batch_attention_sums_unit = tf.reduce_sum(ins_att.flat_values, axis=0)
    batch_cluster_means_unit = tf.divide(tf.tensordot(ins_att.flat_values, inputs.flat_values, [0, 0]) , batch_attention_sums_unit[:, tf.newaxis])
    cluster_euclidean = tf.sqrt(tf.reduce_sum(tf.square(batch_cluster_means_unit[:,tf.newaxis,:] - batch_cluster_means_unit[tf.newaxis,:,:]), axis=-1))
    cluster_euclidean = tf.clip_by_value(cluster_euclidean,-1e12,1e12)
    norms = tf.norm(batch_cluster_means_unit, axis=1)
    norm_matrix = tf.multiply(norms[:,tf.newaxis],norms[tf.newaxis,:])
    normalized_distances = tf.divide(cluster_euclidean, norm_matrix)
    off_diag_mask = 1 - tf.eye(batch_cluster_means_unit.shape[0])
    off_diag_values = tf.multiply(normalized_distances, off_diag_mask)
    centroid_dist = tf.divide(tf.reduce_sum(off_diag_values),tf.reduce_sum(off_diag_mask))
    return - centroid_dist * weight

# cellular attention regularization target, cluster centroids Euclidean distance (unclipped)
def centroid_dist_unnormalized_euclidean(inputs, ins_att,weight = 0.1):
    batch_attention_sums_unit = tf.reduce_sum(ins_att.flat_values, axis=0)
    batch_cluster_means_unit = tf.math.divide_no_nan(tf.tensordot(ins_att.flat_values, inputs.flat_values, [0, 0]) , batch_attention_sums_unit[:, tf.newaxis])
    cluster_euclidean = tf.sqrt(tf.reduce_sum(tf.square(batch_cluster_means_unit[:,tf.newaxis,:] - batch_cluster_means_unit[tf.newaxis,:,:]), axis=-1))
    cluster_euclidean = tf.clip_by_value(cluster_euclidean,-1e12,1e12)
    norm_matrix = tf.ones_like(cluster_euclidean) * tf.math.sqrt(tf.constant(cluster_euclidean.shape[-1],tf.float32))
    normalized_distances = tf.math.divide(cluster_euclidean, norm_matrix)
    off_diag_mask = 1 - tf.eye(batch_cluster_means_unit.shape[0])
    off_diag_values = tf.multiply(normalized_distances, off_diag_mask)
    centroid_dist = tf.divide(tf.reduce_sum(off_diag_values),tf.reduce_sum(off_diag_mask))    
    return -centroid_dist * weight

# cellular attention regularization target, cluster centroids correlation distance
def centroid_dist_correlation(inputs, ins_att):
    batch_attention_sums_unit = tf.reduce_sum(ins_att.flat_values, axis=0)
    batch_cluster_means_unit = tf.divide(tf.tensordot(ins_att.flat_values, inputs.flat_values, [0, 0]) , batch_attention_sums_unit[:, tf.newaxis])
    batch_cluster_means_unit = tf.nn.l2_normalize(batch_cluster_means_unit,axis = 1)
    cluster_correlation = tf.math.abs(tf.matmul(batch_cluster_means_unit, tf.transpose(batch_cluster_means_unit)))
    cluster_correlation = tf.clip_by_value(cluster_correlation,-1e12,1e12)
    off_diag_mask = 1 - tf.eye(batch_cluster_means_unit.shape[0])
    off_diag_values = tf.multiply(cluster_correlation, off_diag_mask)
    centroid_dist = tf.divide(tf.reduce_sum(off_diag_values),tf.reduce_sum(off_diag_mask))
    return centroid_dist*0.1

# collection of COVID multi-tube model specifications
def COVID_model_attr_collection(args):
    args.instance_mlps_spec = []
    args.instance_mlp_spec_1 = []
    args.instance_mlp_spec_2 = []
    args.instance_mlp_spec_3 = []
    args.instance_mlp_spec_4 = []
    if 'BDC-CR1' in args.selected_panels:
        for i in range(len(args.inst_mlp_setting_1)):
            args.instance_mlp_spec_1.append(args.inst_mlp_setting_1[i])
            args.instance_mlp_spec_1.append(args.dropout)
        args.instance_mlp_spec_1.append(args.n_classes+args.ashbin)
        args.instance_mlps_spec.append(args.instance_mlp_spec_1)
    if 'BDC-CR2' in args.selected_panels:
        for i in range(len(args.inst_mlp_setting_2)):
            args.instance_mlp_spec_2.append(args.inst_mlp_setting_2[i])
            args.instance_mlp_spec_2.append(args.dropout)
        args.instance_mlp_spec_2.append(args.n_classes+args.ashbin)
        args.instance_mlps_spec.append(args.instance_mlp_spec_2)
    if 'TNK-CR1' in args.selected_panels:
        for i in range(len(args.inst_mlp_setting_3)):
            args.instance_mlp_spec_3.append(args.inst_mlp_setting_3[i])
            args.instance_mlp_spec_3.append(args.dropout)
        args.instance_mlp_spec_3.append(args.n_classes+args.ashbin)
        args.instance_mlps_spec.append(args.instance_mlp_spec_3)
    if 'TNK-CR2' in args.selected_panels:
        for i in range(len(args.inst_mlp_setting_4)):
            args.instance_mlp_spec_4.append(args.inst_mlp_setting_4[i])
            args.instance_mlp_spec_4.append(args.dropout)
        args.instance_mlp_spec_4.append(args.n_classes+args.ashbin)
        args.instance_mlps_spec.append(args.instance_mlp_spec_4)