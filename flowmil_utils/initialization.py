import sklearn
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
import json
import os
from flowmil_utils.models import *

def kmeans_based_initialization(data, unit):
    if isinstance(data,tf.RaggedTensor):
        data = data.flat_values.numpy()
    elif isinstance(data,tf.data.Dataset):
        if isinstance(list(data.take(1))[0],tuple):
            if isinstance(list(data.take(1))[0][0],tf.RaggedTensor):
                data = np.concatenate([x[0].flat_values.numpy() for x in data],axis=0)
            else:
                data = np.concatenate([x[0].numpy() for x in data if isinstance(x,tuple)],axis=0)
        else:
            data = np.concatenate([x.numpy() for x in data],axis=0)
    else:
        raise NotImplementedError("data type not supported yet")
    
    #randomly select 1e5 instances
    if data.shape[0] > 1e5:
        data = data[np.random.choice(data.shape[0], int(1e5), replace=False)]
    
    kmeans_model = KMeans(n_clusters=unit,  max_iter=1000, random_state=42, tol=1e-4,
                                          algorithm = 'elkan').fit(data)
    #get means and clustering assignments
    means = kmeans_model.cluster_centers_
    assignments = kmeans_model.labels_
    #compute the std of each cluster
    stds = np.zeros((unit,data.shape[1]))
    for i in range(unit):
        stds[i] = np.std(data[assignments == i], axis=0)
    #compute the ratio of each cluster
    ratios = np.bincount(assignments) / data.shape[0]
    #rearrange the order of clusters based on the mixture coef
    order = np.argsort(ratios)[::-1]
    means = means[order]
    stds = stds[order]
    ratios = ratios[order]
    initialization_info = {'mean': means, 'std': stds,'mixture_coef':ratios}
    return initialization_info

def pretraining_parameter_loading(args,model):
    pretraining_loading_dir = args.pretraining_loading_dir
    pretraining_settings= json.load(open(os.path.join(pretraining_loading_dir,'settings.json'),'r'))

    if (args.dataset == 'aml_2015')*(args.model == 'mil_v0'):
        pretraining_model = MLP(input_spec=tf.TensorSpec(shape=(None,args.instance_f), dtype=tf.float32), model_spec=[32,0.5,16,0.5,3],output_activation='sigmoid',bn=True)

    elif (args.dataset == 'aml_2015')*(args.model == 'simple_mlp'):

        pretraining_model = FlowMIL(input_spec=(None, pretraining_settings['instance_f']), instance_weight_spec=None,bag_spec=pretraining_settings['bag_spec'],output_spec=args.n_classes,
                    instance_mlp_spec=pretraining_settings['instance_mlp_spec'],block_settings=pretraining_settings['block_settings'],final_act='softmax',instance_final_act=pretraining_settings['instance_final_act'],
                    instance_supervision=False,seed=42,att_normalization = None,pre_normalization=False)
    else:
        pretraining_model = FlowMIL(input_spec=(None, args.instance_f), instance_weight_spec=None,bag_spec=None,output_spec=args.n_classes,
                    instance_mlp_spec=args.instance_weight_spec,block_settings=pretraining_settings['block_settings'],final_act='softmax',instance_final_act=pretraining_settings['instance_final_act'],
                    instance_supervision=False,seed=42,att_normalization = None,pre_normalization=False)
    pretraining_model.load_weights(os.path.join(pretraining_loading_dir,'model'))
    
    if (args.dataset == 'aml_2015')*(args.model in ['mil_v0']):

        for n_layer in range(len(pretraining_model.layers)):
            model.get_layer('map_tensor_function_ragged').fn.layers[n_layer].set_weights(pretraining_model.layers[n_layer].get_weights())
            if not args.aml_inst_encoder_trainable:
                model.get_layer('map_tensor_function_ragged').fn.layers[n_layer].trainable = False
    elif (args.dataset == 'aml_2015')*(args.model in ['simple_mlp']):

        for n_layer in range(len(model.layers)):
            model.layers[n_layer].set_weights(pretraining_model.get_layer('map_tensor_function_ragged').fn.layers[n_layer].get_weights())
            if not args.aml_inst_encoder_trainable:
                model.layers[n_layer].trainable = False
    elif (args.dataset in ['CRCLC','hivnh'])*(args.model in ['abmil_sh']):
            for n_layer in range(len(model.get_layer('map_tensor_function_ragged_1').fn.layers)):
                model.get_layer('map_tensor_function_ragged_1').fn.layers[n_layer].set_weights(pretraining_model.get_layer('map_tensor_function_ragged_2').fn.layers[n_layer].get_weights())
                if not args.inst_att_trainable:
                    model.get_layer('map_tensor_function_ragged_1').fn.layers[n_layer].trainable = False

def gmm_classification_initialization(args,model):
    pretrained_model =  GMM(input_spec=tf.TensorSpec(shape=(None,None, args.instance_f), dtype=tf.float32),unit=args.unit,GRF_encoding_method=args.GRF_encoding_method,
        maximum_normalization= False,phase= 'clustering',return_features_var=args.punishment_on_chol)
    pretrained_model.load_weights(os.path.join(args.pretraining_loading_dir,'model'))
    
    if args.GRF_encoding_method == 'GRF':
        model.get_layer('gaussian_radial_basis_layer').l.assign(pretrained_model.get_layer('gaussian_radial_basis_layer_1').l)
        model.get_layer('gaussian_radial_basis_layer').s.assign(pretrained_model.get_layer('gaussian_radial_basis_layer_1').s)
        if args.gmm_embedding_layer_trainable:
            assert model.get_layer('gaussian_radial_basis_layer').l.trainable == True
            assert model.get_layer('gaussian_radial_basis_layer').s.trainable == True
        else:
            assert model.get_layer('gaussian_radial_basis_layer').l.trainable == False
            assert model.get_layer('gaussian_radial_basis_layer').s.trainable == False

    elif args.GRF_encoding_method == 'mlp':
        model.get_layer('gaussian_mlp_basis_layer').instance_att_encoder.fn.set_weights(pretrained_model.get_layer('gaussian_mlp_basis_layer_1').instance_att_encoder.fn.get_weights())
        if args.gmm_embedding_layer_trainable:
            model.get_layer('gaussian_mlp_basis_layer').trainable = True
        else:
            model.get_layer('gaussian_mlp_basis_layer').trainable = False
    pass

def multi_tube_model_loading(model,selected_panels,split,args):
    
    with open(args.COVID_single_panel_pretrained_dir, "r") as f:
        data = json.load(f)

    tube_1_model_list = data["tube_1_model_list"]
    tube_2_model_list = data["tube_2_model_list"]
    tube_3_model_list = data["tube_3_model_list"]
    tube_4_model_list = data["tube_4_model_list"]
        
    #obtain the corresponding pos of tube in the model
    if len(selected_panels) ==4:
        tube_exist = [1,1,1,1]
    else:
        tube_exist = [0,0,0,0]
        if 'BDC-CR1' in selected_panels:
            tube_exist[0]=1
        if 'BDC-CR2' in selected_panels:
            tube_exist[1]=1
        if 'TNK-CR1' in selected_panels:
            tube_exist[2]=1
        if 'TNK-CR2' in selected_panels:
            tube_exist[3]=1

    if 'BDC-CR1' in selected_panels:
        pretraining_model = FlowMIL(input_spec=(None, args.instance_f), instance_weight_spec=None,bag_spec=None,output_spec=args.n_classes,
                instance_mlp_spec=args.instance_mlps_spec[0],block_settings=args.block_settings,final_act='softmax',instance_final_act=args.instance_final_act,
                instance_supervision=False,seed=42,att_normalization = None,pre_normalization=False)
        pretraining_model.load_weights(os.path.join(tube_1_model_list[split],'model'))
        pretrained_weight_name  = [layer.name for layer in pretraining_model.layers if 'map_tensor_function_ragged' in layer.name ][0]
        model.get_layer('map_tensor_function_ragged').fn.set_weights(pretraining_model.get_layer(pretrained_weight_name).fn.get_weights())
    if 'BDC-CR2' in selected_panels:
        panel_pos = sum(tube_exist[:2])
        pretraining_model = FlowMIL(input_spec=(None, args.instance_f), instance_weight_spec=None,bag_spec=None,output_spec=args.n_classes,
                instance_mlp_spec=args.instance_mlps_spec[panel_pos-1],block_settings=args.block_settings,final_act='softmax',instance_final_act=args.instance_final_act,
                instance_supervision=False,seed=42,att_normalization = None,pre_normalization=False)
        pretraining_model.load_weights(os.path.join(tube_2_model_list[split],'model'))
        pretrained_weight_name  = [layer.name for layer in pretraining_model.layers if 'map_tensor_function_ragged' in layer.name ][0]
        if panel_pos == 2:
            current_panel_name = 'map_tensor_function_ragged_'+str(panel_pos-1)
        else:
            current_panel_name = 'map_tensor_function_ragged'
        model.get_layer(current_panel_name).fn.set_weights(pretraining_model.get_layer(pretrained_weight_name).fn.get_weights())  
    if 'TNK-CR1' in selected_panels:
        panel_pos = sum(tube_exist[:3])
        pretraining_model = FlowMIL(input_spec=(None, args.instance_f), instance_weight_spec=None,bag_spec=None,output_spec=args.n_classes,
                instance_mlp_spec=args.instance_mlps_spec[panel_pos-1],block_settings=args.block_settings,final_act='softmax',instance_final_act=args.instance_final_act,
                instance_supervision=False,seed=42,att_normalization = None,pre_normalization=False)
        pretraining_model.load_weights(os.path.join(tube_3_model_list[split],'model'))
        pretrained_weight_name  = [layer.name for layer in pretraining_model.layers if 'map_tensor_function_ragged' in layer.name ][0]
        
        if panel_pos >1:
            current_panel_name = 'map_tensor_function_ragged_'+str(panel_pos-1)
        else:
            current_panel_name = 'map_tensor_function_ragged'
        model.get_layer(current_panel_name).fn.set_weights(pretraining_model.get_layer(pretrained_weight_name).fn.get_weights()) 
    if 'TNK-CR2' in selected_panels:
        panel_pos = sum(tube_exist[:4])
        pretraining_model = FlowMIL(input_spec=(None, args.instance_f), instance_weight_spec=None,bag_spec=None,output_spec=args.n_classes,
                instance_mlp_spec=args.instance_mlps_spec[panel_pos-1],block_settings=args.block_settings,final_act='softmax',instance_final_act=args.instance_final_act,
                instance_supervision=False,seed=42,att_normalization = None,pre_normalization=False)
        pretraining_model.load_weights(os.path.join(tube_4_model_list[split],'model'))
        pretrained_weight_name  = [layer.name for layer in pretraining_model.layers if 'map_tensor_function_ragged' in layer.name ][0]
        
        if panel_pos >1:
            current_panel_name = 'map_tensor_function_ragged_'+str(panel_pos-1)
        else:
            current_panel_name = 'map_tensor_function_ragged'
        model.get_layer(current_panel_name).fn.set_weights(pretraining_model.get_layer(pretrained_weight_name).fn.get_weights())         

    return model