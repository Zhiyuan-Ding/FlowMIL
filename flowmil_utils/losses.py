import tensorflow as tf

@tf.function
def contrastive_loss(xi, xj,  tau=1, normalize=False):
        ''' this loss is the modified torch implementation by M Diephuis here: https://github.com/mdiephuis/SimCLR/
        the inputs:
        xi, xj: image features extracted from a batch of images 2N, composed of N matching paints
        tau: temperature parameter
        normalize: normalize or not. seem to not be very useful, so better to try without.
        '''

        x = tf.keras.backend.concatenate((xi, xj), axis=0)

        sim_mat = tf.keras.backend.dot(x, tf.keras.backend.transpose(x))
        
        if normalize:
            sim_mat_denom = tf.keras.backend.dot(tf.norm(x, axis=1,keepdims=True),tf.transpose(tf.norm(x, axis=1,keepdims=True)))
            sim_mat = sim_mat / tf.clip_by_value(sim_mat_denom,1e-8,1e8)

        sim_mat_exp = tf.keras.backend.exp(sim_mat /tau)

        if normalize:
            sim_match_denom = tf.norm(xi, axis=1,keepdims=False) * tf.norm(xj, axis=1,keepdims=False)
            sim_match = tf.keras.backend.sum(xi * xj, axis=-1) / sim_match_denom 
            sim_match_exp = tf.keras.backend.exp(sim_match/ tau)
        else:
            sim_match = tf.keras.backend.sum(xi * xj, axis=-1)
            sim_match_exp = tf.keras.backend.exp(sim_match / tau)

        sim_match_exp = tf.keras.backend.concatenate((sim_match_exp, sim_match_exp), axis=0)
        
        norm_sum = tf.keras.backend.exp(tf.ones_like(x)[:,0] / tau)
        loss = -tf.keras.backend.log(sim_match_exp / tf.clip_by_value(tf.keras.backend.sum(sim_mat_exp, axis=-1) - norm_sum,-1,1e8))
        inside_loss = tf.reduce_sum(tf.cast(tf.math.is_nan(loss),tf.float32))
        mean_pos_sim = tf.keras.backend.mean(sim_match)
        mean_neg_sim = (tf.reduce_sum(sim_mat) - tf.reduce_sum(sim_match)*2 - tf.reduce_sum(tf.ones_like(x)[:,0])*2) / ((tf.keras.backend.sum(tf.ones_like(sim_mat))) - tf.reduce_sum(tf.ones_like(x)[:,0])*4 + 1e-5)
        # mean_neg_sim = (tf.reduce_sum(sim_mat) - tf.reduce_sum(sim_match)*2 - tf.reduce_sum(tf.ones_like(x)[:,0])*2)
        loss = tf.reduce_mean(loss)
        return loss, mean_pos_sim, mean_neg_sim,inside_loss
    
# def add_contrastive_loss(hidden,
#                         hidden_norm=True,
#                         temperature=1.0,
#                         strategy=None):
#   """Compute loss for model.

#   Args:
#     hidden: hidden vector (`Tensor`) of shape (bsz, dim).
#     hidden_norm: whether or not to use normalization on the hidden vector.
#     temperature: a `floating` number for temperature scaling.
#     strategy: context information for tpu.

#   Returns:
#     A loss scalar.
#     The logits for contrastive prediction task.
#     The labels for contrastive prediction task.
#   """
#   # Get (normalized) hidden1 and hidden2.
#   if hidden_norm:
#     hidden = tf.math.l2_normalize(hidden, -1)
#   hidden1, hidden2 = tf.split(hidden, 2, 0)
#   batch_size = tf.shape(hidden1)[0]

#   # Gather hidden1/hidden2 across replicas and create local labels.
#   if strategy is not None:
#     hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
#     hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
#     enlarged_batch_size = tf.shape(hidden1_large)[0]
#     # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
#     replica_context = tf.distribute.get_replica_context()
#     replica_id = tf.cast(
#         tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
#     labels_idx = tf.range(batch_size) + replica_id * batch_size
#     labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
#     masks = tf.one_hot(labels_idx, enlarged_batch_size)
#   else:
#     hidden1_large = hidden1
#     hidden2_large = hidden2
#     labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
#     masks = tf.one_hot(tf.range(batch_size), batch_size)

#   logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
#   logits_aa = logits_aa - masks * LARGE_NUM
#   logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
#   logits_bb = logits_bb - masks * LARGE_NUM
#   logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
#   logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

#   loss_a = tf.nn.softmax_cross_entropy_with_logits(
#       labels, tf.concat([logits_ab, logits_aa], 1))
#   loss_b = tf.nn.softmax_cross_entropy_with_logits(
#       labels, tf.concat([logits_ba, logits_bb], 1))
#   loss = tf.reduce_mean(loss_a + loss_b)

#   return loss, logits_ab, labels
    
class NegLogLikelihood(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred,1e-32,1-1e-32)
        return tf.reduce_mean(-tf.math.log(y_pred))

class UnsupervisedMinimizationSquare(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred**2)
    
class UnsupervisedMinimization(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred)
    
class UnsupervisedMinimizationL2(tf.keras.losses.Loss):
    def __init__(self,model,l2_weight = 0.01,name = 'UnsupervisedMinimizationL2'):
        super().__init__(name=name)
        self.model = model
        self.l2_weight = l2_weight
    def compute_model_l2(self):
        l2_sum = tf.add_n([tf.nn.l2_loss(var) for var in self.model.trainable_variables])
        l2_num = tf.add_n([tf.size(var,out_type = tf.float32) for var in self.model.trainable_variables])
        return l2_sum / l2_num
    def call(self, y_true, y_pred):
        l2_loss = self.compute_model_l2()
        return tf.reduce_mean(y_pred) + self.l2_weight * l2_loss
    
class UnsupervisedMinimizationSig(tf.keras.losses.Loss):
    def call(self, y_true, y_pred,k=1,b = 0):
        y_pred = 1 / (tf.exp(-k*(y_pred - b))+1)
        return tf.reduce_mean(y_pred)

class L2Sparse(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return -tf.reduce_mean(y_pred**2)

class channel_wise_mse(tf.keras.losses.Loss):
    def __init__(self,channel_weights):
        self.channel_weights = tf.convert_to_tensor(channel_weights,dtype = tf.float32)
        if tf.math.reduce_sum(self.channel_weights) != 1:
            self.channel_weights = self.channel_weights / tf.math.reduce_sum(self.channel_weights)
        super(channel_wise_mse,self).__init__()
    @tf.function
    def call(self,y_true,y_pred,sample_weight=None):
        # assert tf.shape(self.channel_weights)[-1] == tf.shape(y_pred)[-1], 'channel weights must have the same length as the last dimension of y_pred'
        if isinstance(y_true,tf.RaggedTensor):
            y_true = y_true.flat_values
            y_pred = y_pred.flat_values
        mse = tf.reduce_mean((y_true-y_pred)**2,axis = 0)
        weighted_loss = tf.reduce_mean(mse*self.channel_weights)
        return weighted_loss

# class combination_mrd_loss(tf.keras.losses.Loss):
    
#     @tf.function
#     def call(self,y_true,y_pred,sample_weight=None):
#         tf.
#         mse = tf.reduce_mean((y_true-y_pred)**2,axis = 0)
#         weighted_loss = tf.reduce_mean(mse*self.channel_weights)
#         return weighted_loss
class UnsupervisedNagMinimization(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return -tf.reduce_mean(y_pred)

# class MaxNegLogLikelihood(tf.keras.losses.Loss):
#     def call(self, y_true, y_pred):
#         #only pick pred larger than 0
#         # y_pred = y_pred[y_pred>0]
#         min_eps = tf.reduce_min(y_pred[y_pred>0])
#         min_eps = tf.minimum(min_eps,1e-5)
#         max_prob = tf.reduce_max(y_pred,axis=-1)
#         return tf.reduce_mean(-tf.math.log(y_pred+min_eps))
# class NegLogLikehoodMetric(tf.keras.metrics.Metric):
#     def update_state(self, y_true, y_pred):
#         y_pred = y_pred[y_pred>0]
#         return tf.reduce_mean(-tf.math.log(y_pred))
#     def result(self):
#         return self.true_positives
    
class NegLogLikehoodMetric(tf.keras.metrics.Metric):

  def __init__(self, name='nll', **kwargs):
    super(NegLogLikehoodMetric, self).__init__(name=name, **kwargs)
    self.nll = self.add_weight(name='nll', initializer='zeros')

  def update_state(self, y_true, y_pred,sample_weight=None):
    y_pred = y_pred[y_pred>0]
    min_eps = tf.reduce_min(y_pred)
    min_eps = tf.minimum(min_eps,1e-5)
    self.nll.assign_add(tf.reduce_mean(-tf.math.log(y_pred+min_eps)))

  def result(self):
    return self.nll

class NegLogLikehoodMetric2(tf.keras.metrics.Metric):

  def __init__(self, name='nll', **kwargs):
    super(NegLogLikehoodMetric2, self).__init__(name=name, **kwargs)
    self.nll2 = self.add_weight(name='nll2', initializer='zeros')

  def update_state(self, y_true, y_pred,sample_weight=None):
    y_pred = y_pred[y_pred>0]
    self.nll2.assign_add(tf.reduce_mean(-tf.math.log(y_pred)))

  def result(self):
    return self.nll2
  

class ChannelWiseCE(tf.keras.losses.Loss):
    def __init__(self,channel_weights):
        self.channel_weights = tf.convert_to_tensor(channel_weights,dtype = tf.float32)
        if tf.math.reduce_sum(self.channel_weights) != 1:
            self.channel_weights = self.channel_weights / tf.math.reduce_sum(self.channel_weights)
        # self.CCE = tf.keras.losses.CategoricalCrossentropy(reduction = 'none',axis = 0,from_logits = False)
        super(ChannelWiseCE,self).__init__()
    @tf.function
    def call(self,y_true,y_pred,sample_weight=None):
        # assert tf.shape(self.channel_weights)[-1] == tf.shape(y_pred)[-1], 'channel weights must have the same length as the last dimension of y_pred'
        if isinstance(y_true,tf.RaggedTensor):
            y_true = y_true.flat_values
            y_pred = y_pred.flat_values
        # entropy = self.CCE(y_true,y_pred)
        entropy = -tf.reduce_mean(y_true*tf.math.log(y_pred+1e-10),axis = 0)
        # weighted_loss = tf.math.reduce_sum(entropy*self.channel_weights) / tf.cast(tf.shape(y_pred)[0],tf.float32) 
        weighted_loss = tf.reduce_mean(entropy*self.channel_weights)
        return weighted_loss

def channel_weighted_binary_crossentropy(weights):
   
    def loss_fn(y_true, y_pred):
        # Compute the binary crossentropy for each channel
        # tf.print(y_true.shape)
        # tf.print(y_pred.shape)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred,from_logits=False,axis = 0)
        
        # Apply channel-wise weights
        weights_tensor = tf.constant(weights, dtype=tf.float32)
        weights_tensor = weights_tensor / tf.reduce_sum(weights_tensor)
        # tf.print(weights_tensor.shape)
        # tf.print(bce.shape)
        weighted_bce = tf.multiply(bce, weights_tensor)

        # Reduce along the last dimension (channels)
        return tf.reduce_mean(weighted_bce, axis=-1)
        # return tf.reduce_mean(bce, axis=-1)
    return loss_fn


def inst_sample_weights_computing(inst_labels):
    if isinstance(inst_labels,tf.RaggedTensor):
        inst_labels = inst_labels.flat_values
    inst_counts = tf.reduce_sum(inst_labels,axis = 0)
    class_weights = 1/inst_counts
    class_weights = class_weights / tf.math.reduce_sum(class_weights) *100
    inst_weights = tf.gather(class_weights,tf.argmax(inst_labels,axis = 1))
    return inst_weights

class UnsupervisedNegLogLikelihood(tf.keras.losses.Loss):
    def __init__(self,model,std_weight = 1,centroid_dist_weight = 0.1,lb = 2.0, ub=5.0,lb_mu=10.0, loss_style=1,name = 'UnsupervisedNegLogLikelihood'):
        super().__init__(name=name)
        self.model = model
        self.std_weight = std_weight
        self.centroid_dist_weight = centroid_dist_weight
        self.lb,self.ub = tf.constant([lb]), tf.constant([ub])
        self.loss_style = loss_style
        self.lb_mu = lb_mu
    @tf.function
    def compute_chol_value(self):
        l2_sum = tf.reduce_sum(tf.square(1 / self.model.get_layer('gaussian_radial_basis_layer').s))
        l2_num = tf.size(self.model.get_layer('gaussian_radial_basis_layer').s,out_type = tf.float32)
        return l2_sum / l2_num
    def piecewise_loss_func(self,val):
        return tf.square(tf.maximum(0.0,val-self.lb))+tf.square(tf.maximum(0.0,self.ub-val))
    def piecewise_loss_func_v2(self,val):
        return tf.square(tf.minimum(0.0,val-self.lb))+tf.square(tf.minimum(0.0,self.ub-val))   
    def compute_double_side_chol_loss(self):
        l2_std = tf.reduce_sum(tf.square(1 / self.model.get_layer('gaussian_radial_basis_layer').s),axis = 0)**(0.5)
        n_feat = tf.cast(tf.shape(self.model.get_layer('gaussian_radial_basis_layer').s,out_type=tf.float32)[0],tf.float32)
        piecewise_loss = tf.reduce_mean(self.piecewise_loss_func(l2_std)) / n_feat 
        return piecewise_loss
    def centroid_dists(self):
        centroids = self.model.get_layer('gaussian_radial_basis_layer').l
        centroids_dists = tf.reduce_sum(tf.square(centroids[:,tf.newaxis,:] - centroids[:,:,tf.newaxis]))
        n_cluster = tf.cast(tf.shape(centroids,out_type=tf.float32)[1],tf.float32)
        n_feat = tf.cast(tf.shape(centroids,out_type=tf.float32)[0],tf.float32)
        return centroids_dists / ((n_cluster**2)*n_feat)
    def piecewise_centroid_dists(self):
        centroids = self.model.get_layer('gaussian_radial_basis_layer').l
        # centroid matrix (f,nh,nh)
        centroids_dists = tf.reduce_sum(tf.square(centroids[:,tf.newaxis,:] - centroids[:,:,tf.newaxis]),axis = 0)
        n_cluster = tf.cast(tf.shape(centroids,out_type=tf.float32)[1],tf.float32)
        n_feat = tf.cast(tf.shape(centroids,out_type=tf.float32)[0],tf.float32)
        off_diag_mask = tf.ones((n_cluster,n_cluster)) - tf.eye((n_cluster,n_cluster))
        off_diag_mask = tf.cast(off_diag_mask,dtype = tf.bool)
        off_diag_centroid_dists = centroids_dists[off_diag_mask]
        piecewise_off_diag_centroid_dists = tf.square(tf.maximum(0.0,off_diag_centroid_dists/n_feat-self.lb))

    def call(self, y_true, y_pred):

        y_pred = tf.clip_by_value(y_pred,1e-32,1-1e-32)
        if self.loss_style == 1:
            return tf.reduce_mean(-tf.math.log(y_pred)) + self.compute_chol_value() * self.std_weight - self.centroid_dist_weight * self.centroid_dists()
        elif self.loss_style == 2:
            return tf.reduce_mean(-tf.math.log(y_pred)) + self.compute_double_side_chol_loss()* self.std_weight - self.centroid_dist_weight * self.centroid_dists()