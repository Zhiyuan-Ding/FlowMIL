import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split,StratifiedKFold
import h5py
import warnings
import flowio

'''
Dataset related script basis, which includes
1. instance augmented functions to sample instance from the bag for MIL training, work with ragged tensor format
2. single sample loading
3. mapping functions for single sample loading
4. mapping functions from single sample dataset to batch
5. wrappered dataset specific mapping functions for step 4
6. train/val/test split
'''

#---------------- part one, data augmentation functions --------------
@tf.function
def compulsory_sampling_func_single_half(x):
    '''randomly sample half of bags'''
    direction = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
    if direction == 0:
        x = tf.ragged.boolean_mask(x,x[...,0]>0)
    elif direction == 1:
        x = tf.ragged.boolean_mask(x,x[...,0]<0)
    elif direction == 2:
        x = tf.ragged.boolean_mask(x,x[...,1]>0)
    else:
        x = tf.ragged.boolean_mask(x,x[...,1]<0)
    
    return x
    
@tf.function
def compulsory_sampling_func_single_quater(x):
    '''randomly sample a quater from bags'''
    direction = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
    if direction == 0:
        x = tf.ragged.boolean_mask(x,tf.logical_and(x[...,0]>0,x[...,1]>0))
    elif direction == 1:
        x = tf.ragged.boolean_mask(x,tf.logical_and(x[...,0]<0,x[...,1]>0))
    elif direction == 2:
        x = tf.ragged.boolean_mask(x,tf.logical_and(x[...,0]>0,x[...,1]<0))
    else:
        x = tf.ragged.boolean_mask(x,tf.logical_and(x[...,0]<0,x[...,1]<0))
    
    return x    

@tf.function
def random_sampling_func(x,prob):
    '''for each sample, randomly sample bags with given probs'''
    #generate masks for instances within each sample
    mask = tf.map_fn(lambda y: tf.random.uniform(shape=(tf.shape(y)[0],),minval=0,maxval=1)<prob,x,fn_output_signature=tf.RaggedTensorSpec(shape=[None],dtype=tf.bool))
    x = tf.ragged.boolean_mask(x,mask)
    return x

@tf.function
def random_sampling_func_varying_prob(x,prob_low=0.05,prob_high=0.95):
    '''randomly sampled from the bag with the prob sampled with range [prob_low, prob_high]'''
    mask = tf.map_fn(lambda y: tf.less_equal(tf.random.uniform(shape=(tf.shape(y)[0],),minval=0,maxval=1),tf.random.uniform(shape=(),minval=prob_low,maxval=prob_high)),
                     x,fn_output_signature=tf.RaggedTensorSpec(shape=[None],dtype=tf.bool))
    x = tf.ragged.boolean_mask(x,mask)
    return x

#---------------- part 2 single sample loading ------------------

# single sample loading for AML2015 simulated experiment, the data is in .h5 format with train/val/test sets generated from preprocessing.py
def load_AML(args):
    train_dir = os.path.join(args.dataset_dir,'trainset')
    test_dir = os.path.join(args.dataset_dir,'testset')
    val_dir = os.path.join(args.dataset_dir,'valset')
    train_list = os.listdir(train_dir)
    train_list = [os.path.join(train_dir,x) for x in train_list]
    test_list = os.listdir(test_dir)
    test_list = [os.path.join(test_dir,x) for x in test_list]
    valid_list = os.listdir(val_dir)
    valid_list = [os.path.join(val_dir,x) for x in valid_list]
    args.instance_f = h5py.File(train_list[0])[args.feature_type][:].shape[-1]
    if args.load_cell_label:
        if args.task == 'classification':
            output_spec = (tf.RaggedTensorSpec(shape=(1, None, args.instance_f), dtype=tf.float64, ragged_rank=1),
                        tf.TensorSpec(shape = (1,),dtype=tf.float64),
                        tf.RaggedTensorSpec(shape=(1, None), dtype=tf.int32, ragged_rank=1))
        elif args.task == 'regression':
            output_spec = (tf.RaggedTensorSpec(shape=(1, None, args.instance_f), dtype=tf.float64, ragged_rank=1),
                        tf.TensorSpec(shape = (3,),dtype=tf.float64),
                        tf.RaggedTensorSpec(shape=(1, None), dtype=tf.int32, ragged_rank=1))
    else:
        if args.task == 'classification':
            output_spec = (tf.RaggedTensorSpec(shape=(1, None, args.instance_f), dtype=tf.float64, ragged_rank=1),
                            tf.TensorSpec(shape = (1,),dtype=tf.float64))
        elif args.task == 'regression':
            output_spec = (tf.RaggedTensorSpec(shape=(1, None, args.instance_f), dtype=tf.float64, ragged_rank=1),
                            tf.TensorSpec(shape = (3,),dtype=tf.float64))
    if args.validation:
        train_dataset = tf.data.Dataset.range(len(train_list))
        test_dataset = tf.data.Dataset.range(len(test_list))
        valid_dataset = tf.data.Dataset.range(len(valid_list))
        train_func = MRDSyntheticDatasetMapping(train_list,output_tensor_spec=output_spec,task = args.task,feature_type=args.feature_type,use_cell_labels=args.load_cell_label,instance_f=args.instance_f)
        valid_func = MRDSyntheticDatasetMapping(valid_list,output_tensor_spec=output_spec,task = args.task,feature_type=args.feature_type,use_cell_labels=args.load_cell_label,instance_f=args.instance_f)
        test_func = MRDSyntheticDatasetMapping(test_list,output_tensor_spec=output_spec,task = args.task,feature_type=args.feature_type,use_cell_labels=args.load_cell_label,instance_f=args.instance_f)
        train_dataset = train_dataset.map(train_func)
        valid_dataset = valid_dataset.map(valid_func)
        test_dataset = test_dataset.map(test_func)
        data_dict = [train_dataset,valid_dataset,test_dataset]
    else:
        train_dataset = tf.data.Dataset.range(len(train_list))
        test_dataset = tf.data.Dataset.range(len(test_list))
        train_func = MRDSyntheticDatasetMapping(train_list,output_tensor_spec=output_spec,task = args.task, use_cell_labels=args.load_cell_label,feature_type=args.feature_type,instance_f=args.instance_f)
        test_func = MRDSyntheticDatasetMapping(test_list,output_tensor_spec=output_spec,task = args.task, use_cell_labels=args.load_cell_label,feature_type=args.feature_type,instance_f=args.instance_f)
        train_dataset = train_dataset.map(train_func)
        test_dataset = test_dataset.map(test_func)
        data_dict = [train_dataset,test_dataset]
    return data_dict

# loading cells from the preprocessed normalized cells from AML2015, each file is cells from patients with one phenotype
def load_AML_real_cells(args):
    assert os.path.isfile(os.path.join(args.dataset_dir,'normalized_CN_blasts.npy')), 'check the dir for cell loading'
    CN_blasts = np.load(os.path.join(args.dataset_dir,'normalized_CN_blasts.npy'))
    CBF_blasts = np.load(os.path.join(args.dataset_dir,'normalized_CBF_blasts.npy'))
    healthy_cells = np.load(os.path.join(args.dataset_dir,'normalized_healthy_cells.npy'))
    np.random.seed(42)
    np.random.shuffle(healthy_cells)
    np.random.shuffle(CN_blasts)
    np.random.shuffle(CBF_blasts)
    cells = np.concatenate([healthy_cells,CN_blasts,CBF_blasts])
    selected_idx = np.random.choice(cells.shape[0],120000,replace=False)
    healthy_used_idx = selected_idx[selected_idx<len(healthy_cells)]
    cn_used_idx = selected_idx[(selected_idx>=len(healthy_cells)) & (selected_idx<len(healthy_cells)+len(CN_blasts))] - len(healthy_cells)
    cbf_used_idx = selected_idx[selected_idx>=len(healthy_cells)+len(CN_blasts)] - len(healthy_cells) - len(CN_blasts)
    healthy_cells = np.delete(healthy_cells,healthy_used_idx,axis=0)
    CN_blasts = np.delete(CN_blasts,cn_used_idx,axis=0)
    CBF_blasts = np.delete(CBF_blasts,cbf_used_idx,axis=0)
    train_healthy_cells = healthy_cells[:int(0.25 * len(healthy_cells))][:30000]
    train_CN_blasts = CN_blasts[:int(0.25 * len(CN_blasts))][:30000]
    train_CBF_blasts = CBF_blasts[:int(0.25 * len(CBF_blasts))][:30000]
    val_healthy_cells = healthy_cells[int(0.25 * len(healthy_cells)):int(0.5 * len(healthy_cells))][:30000]
    val_CN_blasts = CN_blasts[int(0.25 * len(CN_blasts)):int(0.5 * len(CN_blasts))][:30000]
    val_CBF_blasts = CBF_blasts[int(0.25 * len(CBF_blasts)):int(0.5 * len(CBF_blasts))][:30000]
    test_healthy_cells = healthy_cells[int(0.5 * len(healthy_cells)):][:30000]
    test_CN_blasts = CN_blasts[int(0.5 * len(CN_blasts)):][:30000]
    test_CBF_blasts = CBF_blasts[int(0.5 * len(CBF_blasts)):][:30000]
    train_cells = np.concatenate([train_healthy_cells,train_CN_blasts,train_CBF_blasts],axis = 0)
    val_cells = np.concatenate([val_healthy_cells,val_CN_blasts,val_CBF_blasts],axis = 0)
    test_cells = np.concatenate([test_healthy_cells,test_CN_blasts,test_CBF_blasts],axis = 0)

    return [train_cells,val_cells,test_cells]

# single sample loading for AML2015 simulated experiment with bag level aggregated feature
def load_AML_sample(args):
    train_dir = os.path.join(args.dataset_dir,'trainset_'+args.feature_type+'.npy')
    test_dir = os.path.join(args.dataset_dir,'testset_'+args.feature_type+'.npy')
    val_dir = os.path.join(args.dataset_dir,'valset_'+args.feature_type+'.npy')
    train_list = np.load(train_dir,allow_pickle=True)
    test_list = np.load(test_dir,allow_pickle=True)
    valid_list = np.load(val_dir,allow_pickle=True)
    #replace nan with 0
    train_list[np.isnan(train_list)] = 0
    test_list[np.isnan(test_list)] = 0
    valid_list[np.isnan(valid_list)] = 0
    if args.task == 'classification':
        train_labels = np.load(os.path.join(args.dataset_dir,'trainset_labels.npy'))
        test_labels = np.load(os.path.join(args.dataset_dir,'testset_labels.npy'))
        valid_labels = np.load(os.path.join(args.dataset_dir,'valset_labels.npy'))
    elif args.task == 'regression':
        train_labels = np.load(os.path.join(args.dataset_dir,'trainset_mrd_ratio.npy'))
        test_labels = np.load(os.path.join(args.dataset_dir,'testset_mrd_ratio.npy'))
        valid_labels = np.load(os.path.join(args.dataset_dir,'valset_mrd_ratio.npy'))
    if args.validation:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_list.reshape(train_labels.shape[0],-1),train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_list.reshape(test_labels.shape[0],-1),test_labels))
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_list.reshape(valid_labels.shape[0],-1),valid_labels))
        data_dict = [train_dataset,valid_dataset,test_dataset]
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_list.reshape(train_labels.shape[0],-1),train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_list.reshape(test_labels.shape[0],-1),test_labels))
        data_dict = [train_dataset,test_dataset]
    return data_dict

# optional for future api, folders contains aggregated feature and labels, CRCLC
def load_CRCLC_sample(args):
    labels = np.load(os.path.join(args.dataset_dir,'labels.npy')).squeeze()
    if args.feature_type == 'mstep_mean':
        features = np.load(os.path.join(args.dataset_dir,'mstep_mean.npy'))
    elif args.feature_type == 'mstep_std':
        features = np.load(os.path.join(args.dataset_dir,'mstep_std.npy'))
    elif args.feature_type == 'mstep_pi':
        features = np.load(os.path.join(args.dataset_dir,'mstep_pi.npy'))
    elif args.feature_type == 'umap_avgs':
        features = np.load(os.path.join(args.dataset_dir,'umap_avgs.npy'))
    elif args.feature_type == 'unnormalized_mstep_pi':
        features = np.load(os.path.join(args.dataset_dir,'unnormalized_mstep_pi.npy'))
    elif args.feature_type == 'log_unnormalized_mstep_pi':
        features = np.load(os.path.join(args.dataset_dir,'log_unnormalized_mstep_pi.npy'))
    elif args.feature_type == 'mstep_combined':
        features = np.load(os.path.join(args.dataset_dir,'mstep_combined.npy'))
    elif args.feature_type == 'pdf_ratio_clip':
        features = np.load(os.path.join(args.dataset_dir,'pdf_ratio_clip.npy'))
    elif args.feature_type == 'pdf_ratio_unclip':
        features = np.load(os.path.join(args.dataset_dir,'pdf_ratio_unclip.npy'))
    features = features.reshape(features.shape[0],-1)
    if args.validation:
        train_idx, val_idx,test_idx = train_val_test_split(args,labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((features[train_idx],labels[train_idx]))
        test_dataset = tf.data.Dataset.from_tensor_slices((features[test_idx],labels[test_idx]))
        valid_dataset = tf.data.Dataset.from_tensor_slices((features[val_idx],labels[val_idx]))
        data_dict = [train_dataset,valid_dataset,test_dataset]
    else:
        train_idx,test_idx = train_test_split_by_label(args,labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((features[train_idx],labels[train_idx]))
        test_dataset = tf.data.Dataset.from_tensor_slices((features[test_idx],labels[test_idx]))
        data_dict = [train_dataset,test_dataset]
    return data_dict
    
# optional for future api, folders contains aggregated feature and labels, AML, HIVNH, CRCLC
def load_sample_v2(args):
    if args.dataset in ['aml_2015']:
        train_dir = os.path.join(args.dataset_dir,'trainset_'+args.feature_type+'.npy')
        test_dir = os.path.join(args.dataset_dir,'testset_'+args.feature_type+'.npy')
        val_dir = os.path.join(args.dataset_dir,'valset_'+args.feature_type+'.npy')
        train_list = np.load(train_dir,allow_pickle=True).item()
        test_list = np.load(test_dir,allow_pickle=True).item()
        valid_list = np.load(val_dir,allow_pickle=True).item()
        train_data = train_list['features']
        test_data = test_list['features']
        valid_data = valid_list['features']
        train_labels = train_list['labels']
        test_labels = test_list['labels']
        valid_labels = valid_list['labels']
    elif args.dataset in ['CRCLC','hivnh']:
        train_dir = os.path.join(args.dataset_dir,'trainset_'+args.feature_type+'.npy')
        test_dir = os.path.join(args.dataset_dir,'testset_'+args.feature_type+'.npy')
        val_dir = os.path.join(args.dataset_dir,'valset_'+args.feature_type+'.npy')
        train_list = np.load(train_dir,allow_pickle=True).item()
        test_list = np.load(test_dir,allow_pickle=True).item()
        valid_list = np.load(val_dir,allow_pickle=True).item()
        train_data = train_list['features']
        test_data = test_list['features']
        valid_data = valid_list['features']
        train_labels = train_list['labels']
        test_labels = test_list['labels']
        valid_labels = valid_list['labels']
        data = np.concatenate([train_data,test_data,valid_data],axis=0)
        labels = np.concatenate([train_labels,test_labels,valid_labels],axis=0)
        if len(labels.shape) == 2:
            labels_to_split = labels.argmax(-1)
        else:
            labels_to_split = labels
        train_idx, val_idx,test_idx = train_val_test_split(args,labels_to_split)
        train_data = data[train_idx]
        test_data = data[test_idx]
        valid_data = data[val_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        valid_labels = labels[val_idx]
        
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data,train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data,test_labels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data,valid_labels))
    data_dict = [train_dataset,valid_dataset,test_dataset]
    return data_dict

# single sample loading for TIL(CRCLC) experiment, the data is in .h5 format from preprocessing.py
def load_CRCLC(args):

    sample_names = np.array([x.split('.')[0] for x in os.listdir(args.dataset_dir)])
    labels = np.array([h5py.File(os.path.join(args.dataset_dir,x),'r')['sample_labels'][()] for x in os.listdir(args.dataset_dir)]).squeeze()
    if args.feature_type == 'cells':
        fcs_data = np.array([h5py.File(os.path.join(args.dataset_dir,x),'r')['cells'][:] for x in os.listdir(args.dataset_dir)],dtype=object)
    elif args.feature_type =='umap_cells':
        fcs_data = np.array([h5py.File(os.path.join(args.dataset_dir,x),'r')['umap_cells'][:] for x in os.listdir(args.dataset_dir)],dtype=object)
    elif args.feature_type == 'log_responsibilities':
        fcs_data = np.array([h5py.File(os.path.join(args.dataset_dir,x),'r')['log_responsibilities'][:] for x in os.listdir(args.dataset_dir)],dtype=object)
    elif args.feature_type == 'sample_log_prob':
        fcs_data = np.array([h5py.File(os.path.join(args.dataset_dir,x),'r')['sample_log_prob'][:] for x in os.listdir(args.dataset_dir)],dtype=object)
    else:
        raise ValueError('feature type not recognized')
    cell_size = np.array([len(sample) for sample in fcs_data])
    fcs_data = np.concatenate(fcs_data.tolist())
    import flowutils
    fcs_data = flowutils.transforms.logicle(fcs_data, t = 16409, m = 4.5, w = 0.25,a=0,channel_indices=None)
    mean_fcs_all = fcs_data.mean()
    std_fcs_all = fcs_data.std()
    fcs_data = np.array([fcs_data[cell_size[:i].sum():cell_size[:i+1].sum()] for i in range(cell_size.shape[0])],dtype=object)
    
    return_sample_info = args.return_sample_info
    if args.validation:
        train_idx, val_idx,test_idx = train_val_test_split(args,labels)
        train_fcs_data = tf.RaggedTensor.from_row_lengths(tf.concat(fcs_data[train_idx].tolist(),axis=0),cell_size[train_idx])
        test_fcs_data = tf.RaggedTensor.from_row_lengths(tf.concat(fcs_data[test_idx].tolist(),axis=0),cell_size[test_idx])
        val_fcs_data = tf.RaggedTensor.from_row_lengths(tf.concat(fcs_data[val_idx].tolist(),axis=0),cell_size[val_idx])
        train_fcs_data = (train_fcs_data-mean_fcs_all)/std_fcs_all
        val_fcs_data = (val_fcs_data-mean_fcs_all)/std_fcs_all
        test_fcs_data = (test_fcs_data-mean_fcs_all)/std_fcs_all
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]
        train_cell_size = cell_size[train_idx]
        val_cell_size = cell_size[val_idx]
        test_cell_size = cell_size[test_idx]
        train_sample_names = sample_names[train_idx]
        val_sample_names = sample_names[val_idx]
        test_sample_names = sample_names[test_idx]
        
        if return_sample_info:
            train_tfds = tf.data.Dataset.from_tensor_slices((train_fcs_data,train_labels,train_cell_size,train_sample_names))
            val_tfds = tf.data.Dataset.from_tensor_slices((val_fcs_data,val_labels,val_cell_size,val_sample_names))
            test_tfds = tf.data.Dataset.from_tensor_slices((test_fcs_data,test_labels,test_cell_size,test_sample_names))
        else:
            train_tfds = tf.data.Dataset.from_tensor_slices((train_fcs_data,train_labels))
            val_tfds = tf.data.Dataset.from_tensor_slices((val_fcs_data,val_labels))
            test_tfds = tf.data.Dataset.from_tensor_slices((test_fcs_data,test_labels))
        data_dict = [train_tfds,val_tfds,test_tfds]
    else:
        train_idx,test_idx = train_test_split_by_label(args,labels)
        train_fcs_data = tf.RaggedTensor.from_row_lengths(tf.concat(fcs_data[train_idx].tolist(),axis=0),cell_size[train_idx])
        test_fcs_data = tf.RaggedTensor.from_row_lengths(tf.concat(fcs_data[test_idx].tolist(),axis=0),cell_size[test_idx])
        train_fcs_data = (train_fcs_data-mean_fcs_all)/std_fcs_all
        val_fcs_data = (val_fcs_data-mean_fcs_all)/std_fcs_all
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        train_cell_size = cell_size[train_idx]
        test_cell_size = cell_size[test_idx]
        train_sample_names = sample_names[train_idx]
        test_sample_names = sample_names[test_idx]
        
        if return_sample_info:
            train_tfds = tf.data.Dataset.from_tensor_slices((train_fcs_data,train_labels,train_cell_size,train_sample_names))
            test_tfds = tf.data.Dataset.from_tensor_slices((test_fcs_data,test_labels,test_cell_size,test_sample_names))
        else:
            train_tfds = tf.data.Dataset.from_tensor_slices((train_fcs_data,train_labels))
            test_tfds = tf.data.Dataset.from_tensor_slices((test_fcs_data,test_labels))
        data_dict = [train_tfds,test_tfds]
    
    return data_dict

# single sample loading for HIVNH experiment, the data is in .h5 format from preprocessing.py    
def load_HIVNH(args):

    if args.feature_type == 'cells':
        h5_dir = os.path.join(args.dataset_dir,'h5')
        arr = np.array([os.path.join(h5_dir,x) for x in os.listdir(h5_dir)])
        labels = np.array([h5py.File(x,'r')['sample_labels'][()] for x in arr])

    elif args.feature_type in ['umap_cells','log_responsibilities','sample_log_prob']:
        arr = np.array([os.path.join(args.dataset_dir,x) for x in os.listdir(args.dataset_dir)])
        labels = np.array([h5py.File(x,'r')['sample_labels'][()] for x in arr])

    # TODO: this part it redundant and have been changed in preprocessing step, should be deleted later
    if args.feature_type == 'cells':
        feature_type = 'sample_features'
    else:
        feature_type = args.feature_type

    args.instance_f = h5py.File(arr[0])[feature_type][:].shape[-1]

    output_spec = (tf.RaggedTensorSpec(shape=(1, None, args.instance_f), dtype=tf.float64, ragged_rank=1),
                    tf.TensorSpec(shape = (1,),dtype=tf.float64))
    return_sample_info = args.return_sample_info
    if return_sample_info:
        output_spec = (tf.RaggedTensorSpec(shape=(1, None, args.instance_f), dtype=tf.float64, ragged_rank=1),
                        tf.TensorSpec(shape = (1,),dtype=tf.float64),
                        tf.TensorSpec(shape = (1,),dtype=tf.string))
    if args.validation:
        train_idx, val_idx,test_idx = train_val_test_split(args,labels)
        train_dataset = tf.data.Dataset.range(len(train_idx))
        test_dataset = tf.data.Dataset.range(len(test_idx))
        val_dataset = tf.data.Dataset.range(len(val_idx))
        train_func = HIVNHDatasetMapping(arr[train_idx].tolist(),output_tensor_spec=output_spec,task = args.task,feature_type=args.feature_type,return_sample_names=return_sample_info)
        val_func = HIVNHDatasetMapping(arr[val_idx].tolist(),output_tensor_spec=output_spec,task = args.task,feature_type=args.feature_type,return_sample_names=return_sample_info)
        test_func = HIVNHDatasetMapping(arr[test_idx].tolist(),output_tensor_spec=output_spec,task = args.task,feature_type=args.feature_type,return_sample_names=return_sample_info)
        train_dataset = train_dataset.map(train_func)
        val_dataset = val_dataset.map(val_func)
        test_dataset = test_dataset.map(test_func)
        data_dict = [train_dataset,val_dataset,test_dataset]
    else:
        train_dataset = tf.data.Dataset.range(len(train_idx))
        test_dataset = tf.data.Dataset.range(len(test_idx))
        train_func = HIVNHDatasetMapping(arr[train_idx].tolist(),output_tensor_spec=output_spec,task = args.task,feature_type=args.feature_type)
        test_func = HIVNHDatasetMapping(arr[test_idx].tolist(),output_tensor_spec=output_spec,task = args.task,feature_type=args.feature_type)
        train_dataset = train_dataset.map(train_func)
        test_dataset = test_dataset.map(test_func)
        data_dict = [train_dataset,test_dataset]
    return data_dict

# single sample loading for HIVNH experiment with bag level aggregated feature
def load_HIVNH_sample(args):
    labels = np.load(os.path.join(args.dataset_dir,'labels.npy')).squeeze()
    if args.feature_type == 'mstep_mean':
        features = np.load(os.path.join(args.dataset_dir,'mstep_mean.npy'))
    elif args.feature_type == 'mstep_std':
        features = np.load(os.path.join(args.dataset_dir,'mstep_std.npy'))
    elif args.feature_type == 'mstep_pi':
        features = np.load(os.path.join(args.dataset_dir,'mstep_pi.npy'))
    elif args.feature_type == 'umap_avgs':
        features = np.load(os.path.join(args.dataset_dir,'umap_avgs.npy'))
    elif args.feature_type == 'unnormalized_mstep_pi':
        features = np.load(os.path.join(args.dataset_dir,'unnormalized_mstep_pi.npy'))
    elif args.feature_type == 'log_unnormalized_mstep_pi':
        features = np.load(os.path.join(args.dataset_dir,'log_unnormalized_mstep_pi.npy'))
    elif args.feature_type == 'mstep_combined':
        features = np.load(os.path.join(args.dataset_dir,'mstep_combined.npy'))
    elif args.feature_type == 'pdf_ratio_clip':
        features = np.load(os.path.join(args.dataset_dir,'pdf_ratio_clip.npy'))
    elif args.feature_type == 'pdf_ratio_unclip':
        features = np.load(os.path.join(args.dataset_dir,'pdf_ratio_unclip.npy'))
        
    features = features.reshape(features.shape[0],-1)
    if args.validation:
        train_idx, val_idx,test_idx = train_val_test_split(args,labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((features[train_idx],labels[train_idx]))
        test_dataset = tf.data.Dataset.from_tensor_slices((features[test_idx],labels[test_idx]))
        valid_dataset = tf.data.Dataset.from_tensor_slices((features[val_idx],labels[val_idx]))
        data_dict = [train_dataset,valid_dataset,test_dataset]
    else:
        train_idx,test_idx = train_test_split_by_label(args,labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((features[train_idx],labels[train_idx]))
        test_dataset = tf.data.Dataset.from_tensor_slices((features[test_idx],labels[test_idx]))
        data_dict = [train_dataset,test_dataset]
    return data_dict

# single sample loading for COVID experiment, the data is in .h5 format from preprocessing.py, work for both single and multi-panel mode
def load_COVID(args):
    df = pd.read_csv(args.COVID_demographic)
    binary_classification_labels =   df['Severity'] == 'Healthy'
    binary_classification_labels = binary_classification_labels.values.astype(np.int32)
    severity_prediction_labels = np.zeros(len(df),dtype=np.int32)
    severity_prediction_labels[((df['Severity'] == 'Mild')+(df['Severity'] == 'Moderate')).values] = 1
    severity_prediction_labels[((df['Severity'] == 'Critical (ventilation)')+(df['Severity'] == 'Severe (no venitlation)')).values] = 2
    all_prediction_labels = np.zeros(len(df),dtype=np.int32)
    all_prediction_labels[(df['Severity'] == 'Mild')] = 1
    all_prediction_labels[(df['Severity'] == 'Moderate')] = 2
    all_prediction_labels[(df['Severity'] == 'Severe (no venitlation)')] = 3
    all_prediction_labels[(df['Severity'] == 'Critical (ventilation)')] = 4

    sample_inds = df['Sample ID (starting number of fcs files)'].values
    filtered_sample_inds = [i for i in sample_inds if os.path.isfile(os.path.join(args.dataset_dir,str(i)+'_BDC-CR1.fcs'))]
    filtered_sample_inds = [i for i in filtered_sample_inds if os.path.isfile(os.path.join(args.dataset_dir,str(i)+'_BDC-CR2.fcs'))]
    filtered_sample_inds = [i for i in filtered_sample_inds if os.path.isfile(os.path.join(args.dataset_dir,str(i)+'_TNK-CR1.fcs'))]
    filtered_sample_inds = [i for i in filtered_sample_inds if os.path.isfile(os.path.join(args.dataset_dir,str(i)+'_TNK-CR2.fcs'))]
    filtered_binary_classification_labels = [i for idx,i in enumerate(binary_classification_labels) if sample_inds[idx] in filtered_sample_inds]
    filtered_severity_prediction_labels = [i for idx,i in enumerate(severity_prediction_labels) if sample_inds[idx] in filtered_sample_inds]
    filtered_all_prediction_labels = [i for idx,i in enumerate(all_prediction_labels) if sample_inds[idx] in filtered_sample_inds]
    h5_file_list = np.array([os.path.join(args.dataset_dir,str(i)+'.h5') for i in filtered_sample_inds])
    if args.COVID_task == 'binary_classification':
        args.n_classes = 2
        filtered_labels = filtered_binary_classification_labels
    elif args.COVID_task == 'severity_prediction':
        filtered_labels = filtered_severity_prediction_labels
        args.n_classes = 3
    elif args.COVID_task == 'all_prediction':
        filtered_labels = filtered_all_prediction_labels
        args.n_classes = 5
    args.n_tubes = len(args.selected_panels)

    output_spec = ()
    for _ in range(len(args.selected_panels)):
        output_spec += (tf.RaggedTensorSpec(shape=(1, None, args.instance_f), dtype=tf.float64, ragged_rank=1),)
    output_spec += (tf.TensorSpec(shape = (1,),dtype=tf.float64),)
    return_sample_info = args.return_sample_info
    if return_sample_info:
        output_spec = ()
        for _ in range(len(args.selected_panels)):
            output_spec += (tf.RaggedTensorSpec(shape=(1, None, args.instance_f), dtype=tf.float64, ragged_rank=1),)
        output_spec += ( tf.TensorSpec(shape = (1,),dtype=tf.float64),
                        tf.TensorSpec(shape = (1,),dtype=tf.string),)
    
    if args.validation:
        train_idx, val_idx,test_idx = train_val_test_split(args,np.array(filtered_labels))
        train_dataset = tf.data.Dataset.range(len(train_idx))
        test_dataset = tf.data.Dataset.range(len(test_idx))
        val_dataset = tf.data.Dataset.range(len(val_idx))
        train_func = COVIDDatasetMapping(h5_file_list[train_idx].tolist(),output_tensor_spec=output_spec,return_sample_names=return_sample_info,selected_panels = args.selected_panels,label_names=args.COVID_task)
        val_func = COVIDDatasetMapping(h5_file_list[val_idx].tolist(),output_tensor_spec=output_spec,return_sample_names=return_sample_info,selected_panels = args.selected_panels,label_names=args.COVID_task)
        test_func = COVIDDatasetMapping(h5_file_list[test_idx].tolist(),output_tensor_spec=output_spec,return_sample_names=return_sample_info,selected_panels = args.selected_panels,label_names=args.COVID_task)
        train_dataset = train_dataset.map(train_func)
        val_dataset = val_dataset.map(val_func)
        test_dataset = test_dataset.map(test_func)
        data_dict = [train_dataset,val_dataset,test_dataset]

    return data_dict

# loading cells from the preprocessed normalized cells from AML2015, each file is cells from patients with one phenotype
def load_cell_instances(args):
    np.random.seed(42)
    healthy_cells = np.load(os.path.join(args.dataset_dir,'healthy_cells_normalized.npy'))
    CN_blasts = np.load(os.path.join(args.dataset_dir,'CN_blasts_normalized.npy'))
    CBF_blasts = np.load(os.path.join(args.dataset_dir,'CBF_blasts_normalized.npy'))
    np.random.shuffle(healthy_cells)
    np.random.shuffle(CN_blasts)
    np.random.shuffle(CBF_blasts)
    cells = np.concatenate([healthy_cells,CN_blasts,CBF_blasts])
    cell_labels = np.concatenate([np.zeros(healthy_cells.shape[0]),np.ones(CN_blasts.shape[0]),2*np.ones(CBF_blasts.shape[0])])
    # train_idx, val_idx,test_idx = train_val_test_split(args,cell_labels)
    selected_idx = np.random.choice(cells.shape[0],120000,replace=False)
    train_idx, val_idx, test_idx = selected_idx[:60000],selected_idx[30000:60000],selected_idx[60000:]
    cell_labels = tf.keras.utils.to_categorical(cell_labels)
    train_cells = tf.gather(cells,train_idx)
    train_labels = cell_labels[train_idx]
    valid_cells = tf.gather(cells,val_idx)
    valid_labels = cell_labels[val_idx]
    test_cells = tf.gather(cells,test_idx)
    test_labels = cell_labels[test_idx]
    train_tfds = tf.data.Dataset.from_tensor_slices((train_cells,train_labels))
    valid_tfds = tf.data.Dataset.from_tensor_slices((valid_cells,valid_labels))
    test_tfds = tf.data.Dataset.from_tensor_slices((test_cells,test_labels)) 
    train_tfds = train_tfds.shuffle(buffer_size=2*args.batch_size,reshuffle_each_iteration=True).batch(args.batch_size)
    valid_tfds = valid_tfds.shuffle(buffer_size=2*args.batch_size,reshuffle_each_iteration=True).batch(args.batch_size)
    test_tfds = test_tfds.shuffle(buffer_size=2*args.batch_size,reshuffle_each_iteration=True).batch(args.batch_size)
    data_dict = [train_tfds,valid_tfds,test_tfds]
    
    return data_dict

#---------------- part three, mapping func for single sample loading  --------------

# single sample loading func for AML dataset 
class MRDSyntheticDatasetMapping():
    # give list of hdf5 files and the expected output specifications of the loader output(s)    
    def __init__(self, filenames, output_tensor_spec,task = 'regression',use_cell_labels = False,feature_type = 'cells',instance_f = 41):
        self.filenames, self.output_tensor_spec = filenames, output_tensor_spec
        self.task, self.use_cell_labels,self.feature_type = task,use_cell_labels,feature_type
        self.instance_f = instance_f
    def _load(self, idx):      
        with h5py.File(self.filenames[idx.numpy()], 'r') as h5:
            if self.task == 'regression':
                if self.use_cell_labels:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.feature_type][:],dtype = np.float64), [h5[self.feature_type].shape[0]]), tf.convert_to_tensor(h5['mrd_ratio'][()], dtype=tf.float64), tf.RaggedTensor.from_row_lengths(np.array(h5['sample_cell_labels'][:],dtype = np.int32), [h5['sample_cell_labels'].shape[0]])
                # using () to get the value of the scalar is necessary for h5!!!
                else:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.feature_type][:],dtype = np.float64), [h5[self.feature_type].shape[0]]), tf.convert_to_tensor(h5['mrd_ratio'][()], dtype=tf.float64)
            elif self.task == 'classification':
                if self.use_cell_labels:
                    cell_labels = h5['sample_cell_labels'][:]
                    # return tf.RaggedTensor.from_row_lengths(np.array(h5[self.feature_type][:],dtype = np.float64), [h5[self.feature_type].shape[0]]), tf.convert_to_tensor(h5['sample_labels'][()], dtype=tf.float64), tf.RaggedTensor.from_row_lengths(np.array(cell_labels,dtype = np.int32), [cell_labels.shape[0]])
                    return tf.convert_to_tensor(h5[self.feature_type][:], dtype=tf.float32), tf.convert_to_tensor(h5['sample_labels'][()], dtype=tf.int32), tf.convert_to_tensor(cell_labels, dtype=tf.int32)
                else:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.feature_type][:],dtype = np.float64), [h5[self.feature_type].shape[0]]), tf.convert_to_tensor(h5['sample_labels'][()], dtype=tf.float64)
    def __call__(self, idx):
        if self.use_cell_labels:        
            ins_feature, label, ins_label =  tf.py_function(self._load, inp=[idx], Tout=self.output_tensor_spec)
            return ins_feature, label, ins_label
        else:
            ins_feature, label =  tf.py_function(self._load, inp=[idx], Tout=self.output_tensor_spec)
            return ins_feature, label

# single sample loading func for HIVNH dataset 
class HIVNHDatasetMapping():
    def __init__(self, filenames, output_tensor_spec,task = 'regression',feature_type = 'cells',return_sample_names = False):
        self.filenames, self.output_tensor_spec = filenames, output_tensor_spec
        if feature_type == 'cells':
            feature_type = 'sample_features'
        self.task, self.feature_type,self.return_sample_names = task,feature_type,return_sample_names
        
    def _load(self,idx):
        with h5py.File(self.filenames[idx.numpy()], 'r') as h5:
            if self.task == 'regression':
                return tf.RaggedTensor.from_row_lengths(np.array(h5[self.feature_type][:],dtype = np.float64), [h5[self.feature_type].shape[0]]),  tf.convert_to_tensor(h5['sample_survival_time'][()], dtype=tf.float64)
            elif self.task == 'classification':
                if self.return_sample_names:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.feature_type][:],dtype = np.float64), [h5[self.feature_type].shape[0]]), tf.convert_to_tensor(h5['sample_labels'][()], dtype=tf.float64), tf.convert_to_tensor(self.filenames[idx.numpy()], dtype=tf.string)
                else:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.feature_type][:],dtype = np.float64), [h5[self.feature_type].shape[0]]), tf.convert_to_tensor(h5['sample_labels'][()], dtype=tf.float64)
    def __call__(self, idx):        
        return tf.py_function(self._load, inp=[idx], Tout=self.output_tensor_spec)

# single sample loading func for COVID dataset, work for both single and multi-panel mode
class COVIDDatasetMapping():
    def __init__(self, filenames, output_tensor_spec,return_sample_names = False,selected_panels = ['BDC-CR1','BDC-CR2','TNK-CR1','TNK-CR2'],label_names = 'severity_prediction'):
        self.filenames, self.output_tensor_spec = filenames, output_tensor_spec
        self.label_names = label_names+'_labels'
        self.return_sample_names,self.selected_panels = return_sample_names,selected_panels
        
    def _load(self,idx):
        with h5py.File(self.filenames[idx.numpy()], 'r') as h5:
            
            if self.return_sample_names:
                # return tf.RaggedTensor.from_row_lengths(np.array(h5[self.feature_type][:],dtype = np.float64), [h5[self.feature_type].shape[0]]), tf.convert_to_tensor(h5['sample_labels'][()], dtype=tf.float64), tf.convert_to_tensor(self.filenames[idx.numpy()], dtype=tf.string)
                if len(self.selected_panels) == 1:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[0]][:],dtype = np.float64), [h5[self.selected_panels[0]].shape[0]]), tf.convert_to_tensor(h5[self.label_names][()], dtype=tf.float64),tf.convert_to_tensor(self.filenames[idx.numpy()], dtype=tf.string)
                elif len(self.selected_panels) == 2:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[0]][:],dtype = np.float64), [h5[self.selected_panels[0]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[1]][:],dtype = np.float64), [h5[self.selected_panels[1]].shape[0]]), tf.convert_to_tensor(h5[self.label_names][()], dtype=tf.float64),tf.convert_to_tensor(self.filenames[idx.numpy()], dtype=tf.string)    
                elif len(self.selected_panels) == 3:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[0]][:],dtype = np.float64), [h5[self.selected_panels[0]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[1]][:],dtype = np.float64), [h5[self.selected_panels[1]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[2]][:],dtype = np.float64), [h5[self.selected_panels[2]].shape[0]]), tf.convert_to_tensor(h5[self.label_names][()], dtype=tf.float64),tf.convert_to_tensor(self.filenames[idx.numpy()], dtype=tf.string)
                elif len(self.selected_panels) == 4:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[0]][:],dtype = np.float64), [h5[self.selected_panels[0]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[1]][:],dtype = np.float64), [h5[self.selected_panels[1]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[2]][:],dtype = np.float64), [h5[self.selected_panels[2]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[3]][:],dtype = np.float64), [h5[self.selected_panels[3]].shape[0]]), tf.convert_to_tensor(h5[self.label_names][()], dtype=tf.float64),tf.convert_to_tensor(self.filenames[idx.numpy()], dtype=tf.string)
            
            else:
                if len(self.selected_panels) == 1:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[0]][:],dtype = np.float64), [h5[self.selected_panels[0]].shape[0]]), tf.convert_to_tensor(h5[self.label_names][()], dtype=tf.float64)
                elif len(self.selected_panels) == 2:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[0]][:],dtype = np.float64), [h5[self.selected_panels[0]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[1]][:],dtype = np.float64), [h5[self.selected_panels[1]].shape[0]]), tf.convert_to_tensor(h5[self.label_names][()], dtype=tf.float64)    
                elif len(self.selected_panels) == 3:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[0]][:],dtype = np.float64), [h5[self.selected_panels[0]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[1]][:],dtype = np.float64), [h5[self.selected_panels[1]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[2]][:],dtype = np.float64), [h5[self.selected_panels[2]].shape[0]]), tf.convert_to_tensor(h5[self.label_names][()], dtype=tf.float64)
                elif len(self.selected_panels) == 4:
                    return tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[0]][:],dtype = np.float64), [h5[self.selected_panels[0]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[1]][:],dtype = np.float64), [h5[self.selected_panels[1]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[2]][:],dtype = np.float64), [h5[self.selected_panels[2]].shape[0]]),tf.RaggedTensor.from_row_lengths(np.array(h5[self.selected_panels[3]][:],dtype = np.float64), [h5[self.selected_panels[3]].shape[0]]), tf.convert_to_tensor(h5[self.label_names][()], dtype=tf.float64)    
    def __call__(self, idx):        
        return tf.py_function(self._load, inp=[idx], Tout=self.output_tensor_spec)

#---------------- part four, mapping func from single sample dataset to batch   --------------

def Dataset_Wrapper(args,data_dict):

    if args.dataset in[ 'aml_2015']:
        if args.feature_type in ['unbiased_cells_for_clustering','cell_instance']:
            if args.validation:
                train_dataset = data_dict[0]
                valid_dataset = data_dict[1]
                test_dataset = data_dict[2]
                return train_dataset,valid_dataset,test_dataset
            else:
                train_dataset = data_dict[0]
                test_dataset = data_dict[1]
                return train_dataset,test_dataset
        else:
            if args.validation:
                train_dataset = data_dict[0]
                valid_dataset = data_dict[1]
                test_dataset = data_dict[2]
                train_dataset = dataset_wrapper_aml(train_dataset,batch_size = args.batch_size,use_cell_labels=args.load_cell_label,task = args.task, n_classes=args.n_classes,feature_type=args.feature_type,instance_aug = args.instance_aug)
                test_dataset = dataset_wrapper_aml(test_dataset,batch_size = args.batch_size,use_cell_labels=args.load_cell_label,task = args.task, n_classes=args.n_classes,feature_type=args.feature_type)
                valid_dataset = dataset_wrapper_aml(valid_dataset,batch_size = args.batch_size,use_cell_labels=args.load_cell_label,task = args.task, n_classes=args.n_classes,feature_type=args.feature_type)
                return train_dataset,valid_dataset,test_dataset
            else:
                train_dataset = data_dict[0]
                test_dataset = data_dict[1]
                train_dataset = dataset_wrapper_aml(train_dataset,batch_size = args.batch_size,use_cell_labels=args.load_cell_label,task = args.task, n_classes=args.n_classes,feature_type=args.feature_type,instance_aug = args.instance_aug)
                test_dataset = dataset_wrapper_aml(test_dataset,batch_size = args.batch_size,use_cell_labels=args.load_cell_label,task = args.task, n_classes=args.n_classes,feature_type=args.feature_type)
                return train_dataset,test_dataset
    elif args.dataset in ['CRCLC']:
        if args.validation:
            train_dataset = data_dict[0]
            valid_dataset = data_dict[1]
            test_dataset = data_dict[2]
            train_dataset = dataset_wrapper_crclc(train_dataset,batch_size = args.batch_size,n_classes = 2,feature_type = args.feature_type,instance_aug = args.instance_aug)
            valid_dataset = dataset_wrapper_crclc(valid_dataset,batch_size = args.batch_size,n_classes = 2,feature_type = args.feature_type)
            test_dataset = dataset_wrapper_crclc(test_dataset,batch_size = args.batch_size,n_classes = 2,feature_type = args.feature_type)
            return train_dataset,valid_dataset,test_dataset
        else:
            train_dataset = data_dict[0]
            test_dataset = data_dict[1]
            train_dataset = dataset_wrapper_crclc(train_dataset,batch_size = args.batch_size,n_classes = 2,feature_type = args.feature_type,instance_aug = args.instance_aug)
            test_dataset = dataset_wrapper_crclc(test_dataset,batch_size = args.batch_size,n_classes = 2,feature_type = args.feature_type)
            return train_dataset,test_dataset
    elif args.dataset in ['hivnh']:
        if args.validation:
            train_dataset = data_dict[0]
            valid_dataset = data_dict[1]
            test_dataset = data_dict[2]
            if args.feature_type == 'cells':
                assert os.path.isfile('./preprocessing_used_files/hivnh_mean_fcs.npy'), 'path error for the stat feature, please check the preprocessing_used_files folder again'
                mean = np.load('./preprocessing_used_files/hivnh_mean_fcs.npy')
                std = np.load('./preprocessing_used_files/hivnh_std_fcs.npy')
            else:
                mean,std = None,None
            train_dataset = dataset_wrapper_hivnh(train_dataset,batch_size=args.batch_size ,n_classes = 2,task = args.task,feature_type = args.feature_type,mean=mean,std=std,instance_aug = args.instance_aug,inst_aug_range = args.inst_aug_range)
            valid_dataset = dataset_wrapper_hivnh(valid_dataset,batch_size=args.batch_size ,n_classes = 2,task = args.task,feature_type = args.feature_type,mean=mean,std=std,instance_aug = args.validation_dropout,inst_aug_range = args.inst_aug_range)
            test_dataset = dataset_wrapper_hivnh(test_dataset,batch_size=args.batch_size ,n_classes = 2,task = args.task,feature_type = args.feature_type,mean=mean,std=std,instance_aug = args.validation_dropout,inst_aug_range = args.inst_aug_range)
            return train_dataset,valid_dataset,test_dataset
        else:
            train_dataset = data_dict[0]
            test_dataset = data_dict[1]
            train_dataset = dataset_wrapper_hivnh(train_dataset,batch_size=args.batch_size ,n_classes = 2,task = args.task,feature_type = args.feature_type,mean=mean,std=std,instance_aug = args.instance_aug,inst_aug_range = args.inst_aug_range)
            test_dataset = dataset_wrapper_hivnh(test_dataset,batch_size=args.batch_size ,n_classes = 2,task = args.task,feature_type = args.feature_type,mean=mean,std=std)
            return train_dataset,test_dataset
    elif args.dataset in ['COVID']:
        if args.validation:
            train_dataset = data_dict[0]
            valid_dataset = data_dict[1]
            test_dataset = data_dict[2]
            train_dataset = dataset_wrapper_covid(train_dataset,batch_size=args.batch_size,n_classes = args.n_classes,instance_aug=args.instance_aug, inst_aug_range= args.inst_aug_range,n_tubes = args.n_tubes)
            valid_dataset = dataset_wrapper_covid(valid_dataset,batch_size=args.batch_size//4,n_classes = args.n_classes,instance_aug=args.validation_dropout, inst_aug_range= args.inst_aug_range,n_tubes = args.n_tubes)
            test_dataset = dataset_wrapper_covid(test_dataset,batch_size=args.batch_size//4,n_classes = args.n_classes,instance_aug=args.validation_dropout, inst_aug_range= args.inst_aug_range,n_tubes = args.n_tubes)
        return train_dataset,valid_dataset,test_dataset
    else:
        raise ValueError('dataset not implemented')

#---------------- part five, mapping func used for the wrapping --------------

# AML dataset batching
def dataset_wrapper_aml(dataset,batch_size=2,use_cell_labels=False,task = 'regression',n_classes = 3,feature_type='cells',instance_aug = False):
    if feature_type in ['cells','umap_cells','log_responsibilities','sample_log_prob','instance_label']:
        if task == 'regression':
            if use_cell_labels:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE ).map(lambda x,y,z: (tf.squeeze(x,1),y,tf.one_hot(tf.squeeze(tf.cast(z,tf.int32),1),depth=n_classes)))
            else:
                if instance_aug:
                    dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (random_sampling_func_varying_prob(tf.squeeze(x,1)),y))
                else:
                    dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (tf.squeeze(x,1),y))
        else:
            if use_cell_labels:
                # dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y,z: (tf.squeeze(x,1),{'tf.identity':tf.one_hot(tf.cast(y,tf.int32),depth=n_classes),'map_tensor_function_ragged':tf.one_hot(tf.squeeze(tf.cast(z,tf.int32),1),depth=n_classes,axis=-1)}))
                # dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y,z: (tf.squeeze(x,1),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes),tf.one_hot(tf.squeeze(tf.cast(z,tf.int32),1),depth=n_classes,axis=-1)))
                dataset = dataset.ragged_batch(batch_size).map(lambda x,y,z: (x,(tf.one_hot(y,depth=n_classes),tf.one_hot(z,depth=n_classes))))
            else:
                if instance_aug:
                    dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (random_sampling_func_varying_prob(tf.squeeze(x,1)),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
                else:
                    dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (tf.squeeze(x,1),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
    
    elif feature_type in ['umap_avgs','sample_log_prob','mstep_mean','mstep_std','mstep_pi','unnormalized_mstep_pi','log_unnormalized_mstep_pi','mstep_combined','pdf_ratio_clip','pdf_ratio_unclip',
                        'mean_inst','mean_inst_feature']:
        if task == 'regression':
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (x,y))
        else:
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (x,tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
    elif feature_type in ['tf_avgs','tf_avgs_sample']:
        dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE)
    # aa = list(dataset.take(1))[0]
    return dataset

# CRCLC dataset batching
def dataset_wrapper_crclc(dataset,batch_size=2,n_classes = 2,feature_type = 'cells',instance_aug = False):
    if feature_type in ['cells','umap_cells','log_responsibilities','sample_log_prob','incremental_cells']:
        if instance_aug:
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (random_sampling_func_varying_prob(x),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
        else:
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (x,tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
    elif feature_type in ['umap_avgs','sample_log_prob','mstep_mean','mstep_std','mstep_pi','unnormalized_mstep_pi','log_unnormalized_mstep_pi','mstep_combined','pdf_ratio_clip','pdf_ratio_unclip']:
        dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (x,tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
    elif feature_type in ['tf_avgs','tf_avgs_sample']:
        dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE)
    return dataset

# HIVNH dataset batching
def dataset_wrapper_hivnh(dataset,batch_size=2,n_classes = 2,task = 'regression',feature_type = 'cells',mean=None,std=None,instance_aug = False,inst_aug_range = None):

    if task == 'regression':
        if feature_type in ['cells']:
            if instance_aug:
                if inst_aug_range is None:
                    prob_low,prob_high = 0.05,0.25
                else:
                    prob_low,prob_high = inst_aug_range[0],inst_aug_range[1]
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: ((random_sampling_func_varying_prob(tf.squeeze(x,1),prob_low=prob_low,prob_high=prob_high)-mean)/std,y))
            else:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: ((tf.squeeze(x,1)-mean)/std,y))
        elif feature_type in ['log_responsibilities','sample_log_prob','umap_cells']:
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (tf.squeeze(x,1),y))
        elif feature_type in ['umap_avgs','mstep_mean','mstep_std','mstep_pi','unnormalized_mstep_pi','log_unnormalized_mstep_pi','mstep_combined','pdf_ratio_clip','pdf_ratio_unclip']:
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (x,y))
    else :
        if feature_type in ['cells']:
            if instance_aug:
                if inst_aug_range is None:
                    prob_low,prob_high = 0.05,0.25
                else:
                    prob_low,prob_high = inst_aug_range[0],inst_aug_range[1]
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: ((random_sampling_func_varying_prob(tf.squeeze(x,1),prob_low=prob_low,prob_high=prob_high)-mean)/std,tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
            else:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: ((tf.squeeze(x,1)-mean)/std,tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
        elif feature_type in ['log_responsibilities','sample_log_prob','umap_cells']:
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (tf.squeeze(x,1),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
        elif feature_type in ['umap_avgs','mstep_mean','mstep_std','mstep_pi','unnormalized_mstep_pi','log_unnormalized_mstep_pi','mstep_combined','pdf_ratio_clip','pdf_ratio_unclip']:
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (x,tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
        elif feature_type in ['tf_avgs','tf_avgs_sample']:
            dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE)
    return dataset

# COVID dataset batching
def dataset_wrapper_covid(dataset,batch_size=2,n_classes = 2,feature_type = 'cells',instance_aug = False,inst_aug_range = None,n_tubes = 4):
 
    if feature_type in ['cells']:
        if instance_aug:
            if inst_aug_range is None:
                prob_low,prob_high = 0.05,0.25
            else:
                prob_low,prob_high = inst_aug_range[0],inst_aug_range[1]
            if n_tubes == 1:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (random_sampling_func_varying_prob(tf.squeeze(x,1),prob_low=prob_low,prob_high=prob_high),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
            elif n_tubes == 2: 
                dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).map(lambda x1, x2, y: ((random_sampling_func_varying_prob(tf.squeeze(x1, 1), prob_low=prob_low, prob_high=prob_high), random_sampling_func_varying_prob(tf.squeeze(x2, 1), prob_low=prob_low, prob_high=prob_high)), tf.one_hot(tf.cast(y, tf.int32), depth=n_classes)))
            elif n_tubes == 3:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x1,x2,x3,y: ((random_sampling_func_varying_prob(tf.squeeze(x1,1),prob_low=prob_low,prob_high=prob_high),random_sampling_func_varying_prob(tf.squeeze(x2,1),prob_low=prob_low,prob_high=prob_high),random_sampling_func_varying_prob(tf.squeeze(x3,1),prob_low=prob_low,prob_high=prob_high)),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
            elif n_tubes == 4:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x1,x2,x3,x4,y: ((random_sampling_func_varying_prob(tf.squeeze(x1,1),prob_low=prob_low,prob_high=prob_high),random_sampling_func_varying_prob(tf.squeeze(x2,1),prob_low=prob_low,prob_high=prob_high),random_sampling_func_varying_prob(tf.squeeze(x3,1),prob_low=prob_low,prob_high=prob_high),random_sampling_func_varying_prob(tf.squeeze(x4,1),prob_low=prob_low,prob_high=prob_high)),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))

        else:
            if n_tubes ==1:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x,y: (tf.squeeze(x,1),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
            elif n_tubes == 2:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x1,x2,y: ((tf.squeeze(x1,1),tf.squeeze(x2,1)),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
            elif n_tubes == 3:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x1,x2,x3,y: ((tf.squeeze(x1,1),tf.squeeze(x2,1),tf.squeeze(x3,1)),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
            elif n_tubes == 4:
                dataset = dataset.batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE).map(lambda x1,x2,x3,x4,y: ((tf.squeeze(x1,1),tf.squeeze(x2,1),tf.squeeze(x3,1),tf.squeeze(x4,1)),tf.one_hot(tf.cast(y,tf.int32),depth=n_classes)))
    return dataset

#---------------- part 5.5, optional, wrapper func provided by Alex Baras for ragged tensor and fcs loading --------------

class LoadBatchByIndices:
    def loader(self):
        raise NotImplementedError

    def __call__(self, indices, ragged_output):
        # flat_values and additional_args together should be the input into the ragged_constructor of the loader
        flat_values, *additional_args = tf.py_function(self.loader, [indices], self.tf_output_types)
        flat_values.set_shape((None, ) + self.inner_shape)

        if ragged_output:
            return self.ragged_constructor(flat_values, *additional_args)
        else:
            return flat_values

class TubeLoader(LoadBatchByIndices):
    @staticmethod
    def load_fcs(fcs_file):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = flowio.FlowData(fcs_file, ignore_offset_error=True)
        return np.reshape(data.events, (-1, data.channel_count)), data.channels
        
    def __init__(self, fcs_files, tube_channels, sub_sample=None):
        self.fcs_files = fcs_files
        self.sub_sample = sub_sample
        
        self.tube_channels = tube_channels
        self.inner_shape = (len(tube_channels), )
        
        self.ragged_constructor = tf.RaggedTensor.from_row_lengths
        self.tf_output_types = [tf.uint16, tf.int32]

    def loader(self, indices):
        batch = list()
        for idx in indices.numpy():
            data, channels = TubeLoader.load_fcs(self.fcs_files[idx])
            batch.append(data)
        
        return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])

#---------------- part six, train/val/test split --------------

# randomly splitting
def train_val_test_split(args,labels):
    kfolds = list(StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42).split(labels,labels))
    kfolds = [x[1] for x in kfolds]
    test_idx = kfolds[args.n_split]
    val_idx = kfolds[(args.n_split+1)%args.cv]
    train_idx = np.setdiff1d(np.arange(labels.shape[0]),np.concatenate([test_idx,val_idx]))
    return train_idx, val_idx, test_idx

# splitting with label
def train_test_split_by_label(args,labels):
    kfolds = list(StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42).split(labels,labels))
    kfolds = [x[1] for x in kfolds]
    test_idx = kfolds[args.n_split]
    train_idx = np.setdiff1d(np.arange(labels.shape[0]),test_idx)
    return train_idx, test_idx

# wrapper of sklearn split
def train_val_split(n_split=0.2,seed=42,labels=None):
    #generate train and valid idx
    train_idx, valid_idx = train_test_split(np.arange(labels.shape[0]),test_size=n_split,random_state=seed)
    return train_idx, valid_idx