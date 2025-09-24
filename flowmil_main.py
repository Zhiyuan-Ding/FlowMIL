import time
import json
import argparse
import pickle
import tensorflow_addons as tfa
from flowmil_utils.dataset import *
from flowmil_utils.models import *
from flowmil_utils.visualization import *
from flowmil_utils.callbacks import *
from flowmil_utils.initialization import *
from flowmil_utils.feature_extractions import *
from flowmil_utils.losses import *
from flowmil_utils.training import *

parser = argparse.ArgumentParser(description='Training profile for FlowMIL')

parser.add_argument('--dataset_dir', type=str, default='', help='dataset directory')
parser.add_argument('--dataset', type=str, default='CRCLC',choices=['aml_2015','CRCLC','hivnh','COVID'], help='dataset name')
parser.add_argument('--model', type=str, default='mil_v0',choices=['simple_mlp','mil_v0','abmil_sh','abmil_mh','gmm'], help='different types of models')
parser.add_argument('--output_dir', type=str, default='./training_result', help='output directory')
parser.add_argument('--n_classes', type=int, default=2, help='num of sample classes for classification')
parser.add_argument('--n_split', type=int, default=0, help='specific split')
parser.add_argument('--inst_mlp_setting', type=int, nargs='+', default=[32,16,8,4], help='instance mlp setting')
parser.add_argument('--instance_final_act', default='sigmoid',type=str, help='instance final activation')
parser.add_argument('--exp_code', default=None,type=str, help='exp code to cover other specified settings')
parser.add_argument('--final_act', type=str, default='softmax', help='final sample level activation')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--feature_type', type=str, default='cells',choices=['cells','mstep_pi','cell_instance','instance_label','mean_inst','mean_inst_feature'], help='feature type')
parser.add_argument('--model_loading_dir', type=str, default='', help='loading dir for model')
parser.add_argument('--validation',  action='store_true',default=False, help='train/val/test split')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--task', type=str, default='classification', choices=['regression','classification'], help='task type')
parser.add_argument('--cv', type=int, default=5, help='n split of cross validation')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--phase', type=str, default='classification',choices=['classification','feature_extraction','evaluation','clustering'], help='phases')
parser.add_argument('--instance_aug', action='store_true', default=False, help='if introducing instance augmentation')
parser.add_argument('--inst_aug_range',  default=None,type=float,nargs='+', help='instance augmentation range (for hivnh,covid training set)')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--n_exp', type=int, default=0, help='experiment number')
parser.add_argument('--return_sample_info', action="store_true", default=False, help='return sample info, only used in sample-specific feature extraction')
parser.add_argument('--pretraining', action='store_true', default=False, help='pretraining activation')
parser.add_argument('--pretraining_loading_dir',type = str, default = None, help='pretraining model loading dir')
parser.add_argument('--inst_att_trainable',  default=False,action='store_true', help='if instance attention network is trainable')
parser.add_argument('--patience', type=int, default=300, help='patience of early stopping')
parser.add_argument('--max_epoch', type=int, default=1000, help='max epochs for training')

# MIL_V0 specific
parser.add_argument('--mil_v0_bag_logits_aggr', default='mean',type=str, help='method for mil_v0 aggr, which works identical to att_normalization for abmil,' \
'mean is eq to normalize_based_on_instance, and sum is eq to normalize_along_instance')

# ABMIL specific
parser.add_argument('--abmil_att_encoding', default='attention_dependent',type=str, help='instance attention encoding method for abmil')
parser.add_argument('--inst_att_setting', type=int, nargs='+', default=[16,8], help='instance att mlp setting')
parser.add_argument('--instance_att_act', default='sigmoid',type=str, help='inst att final act')
parser.add_argument('--att_normalization', default='normalize_along_instance',type=str, help='att normalization strategy')
parser.add_argument('--att_regularization_method', default=None,type=str, help='att regularization method')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1,1], help='loss weights for multiple target')

# AML specific
parser.add_argument('--load_cell_label',  default=False, action="store_true",help='load cell labels')
parser.add_argument('--aml_instance_encoder_loading', action='store_true', default=False, help='aml instance encoder loading')
parser.add_argument('--aml_inst_encoder_trainable', action='store_true', default=False, help='make inst encoder trainable')
# clustering specific
parser.add_argument('--unit', type=int, default=16, help='number of kernels in gmm')

#post-operation after training or evaluation
parser.add_argument('--result_collection', action='store_true', default=False, help='result collection activations')
parser.add_argument('--inst_att_extraction', action='store_true', default=False, help='instance attention extraction')
parser.add_argument('--predict_value_collection', action='store_true', default=False, help='predict value collection')

#COVID dataset specific variables
parser.add_argument('--COVID_task', type=str, default='binary_classification',choices=['binary_classification','severity_prediction','all_prediction'], help='prediction target for COVID dataset')
parser.add_argument('--selected_panels', type=str, nargs='+', default=['BDC-CR1','BDC-CR2','TNK-CR1','TNK-CR2'], help='selected panels for COVID dataset')
#ashbin attention
parser.add_argument('--ashbin', default=0,type=int, help='number of extra head in model')

# only used in abmil_mh experiment
parser.add_argument('--l2_reg_weight', type=float, default=0.0, help='l2 regularization on model variable weight')

#args added for public codes
parser.add_argument('--COVID_single_panel_pretrained_dir', type=str, default=None, help='pretrained dir for COVID single panel model')
parser.add_argument('--COVID_demographic', type=str, default='./preprocessing_used_files/220115_Demographics_for_raw_data_upload.csv', help='demographic information for COVID dataset')
#COVID dataset multi-tube settings
parser.add_argument('--inst_mlp_setting_1', type=int, nargs='+', default=[32,16,8,4], help='instance mlp setting tube 1')
parser.add_argument('--inst_mlp_setting_2', type=int, nargs='+', default=[32,16,8,4], help='instance mlp setting tube 2')
parser.add_argument('--inst_mlp_setting_3', type=int, nargs='+', default=[32,16,8,4], help='instance mlp setting tube 3')
parser.add_argument('--inst_mlp_setting_4', type=int, nargs='+', default=[32,16,8,4], help='instance mlp setting tube 4')
parser.add_argument('--inst_att_setting_1', type=int, nargs='+', default=[16,8], help='instance att mlp setting tube 1')
parser.add_argument('--inst_att_setting_2', type=int, nargs='+', default=[16,8], help='instance att mlp setting tube 2')
parser.add_argument('--inst_att_setting_3', type=int, nargs='+', default=[16,8], help='instance att mlp setting tube 3')
parser.add_argument('--inst_att_setting_4', type=int, nargs='+', default=[16,8], help='instance att mlp setting tube 4')
parser.add_argument('--inst_att_extraction_multi_tube', action='store_true', default=False, help='instance attention extraction for multitube')
parser.add_argument('--inst_mlp_pretraining_loading', action='store_true', default=False, help='loading pretrain single tube model for pretraining')

def train_process():
    args = parser.parse_args()
    def str_none_to_none(ns):
        for k, v in vars(ns).items():
            if v == "None":
                setattr(ns, k, None)
        return ns
    args = str_none_to_none(args)
    print(args)
    if args.phase in ['classification']:
        if args.exp_code is None:
            args.output_dir = os.path.join(args.output_dir,args.dataset,args.model,args.feature_type,time.strftime("%Y%m%d-%H%M%S"))
        else:
            args.output_dir = os.path.join(args.output_dir,args.dataset,args.model,args.exp_code,args.feature_type,time.strftime("%Y%m%d-%H%M%S"))
    else:
        if args.exp_code is not None:
            args.output_dir = os.path.join(args.output_dir,args.dataset,args.model,args.feature_type,args.exp_code,'evaluations')
        else:
            args.output_dir = os.path.join(args.output_dir,args.dataset,args.model,args.feature_type,'evaluations')

    os.makedirs(args.output_dir,exist_ok=True)
    if args.instance_final_act == 'none':
        args.instance_final_act = None
    #dataset loading part
    if args.dataset == 'aml_2015':
        if (args.phase in ['classification']) or (args.phase in ['evaluation'])*(args.predict_value_collection):
            if args.feature_type == 'cells':
                data_dict = load_AML(args)
            elif args.feature_type in ['mstep_mean','mstep_std','mstep_pi','unnormalized_mstep_pi','log_unnormalized_mstep_pi','mstep_combined', 'pdf_ratio_clip','pdf_ratio_unclip']:
                data_dict = load_AML_sample(args)
                args.instance_f = next(iter(data_dict[0]))[0].shape[-1]
        elif (args.phase in ['evaluation'])*(not args.predict_value_collection):
            if args.feature_type == 'cells':
                data_dict = load_AML_real_cells(args) 
                args.instance_f = next(iter(data_dict[0]))[0].shape[-1]
        

    elif args.dataset == 'CRCLC':
        if args.feature_type == 'cells':
            data_dict = load_CRCLC(args)
            args.instance_f = next(iter(data_dict[0]))[0].shape[-1]

    elif args.dataset == 'hivnh':
        if args.feature_type == 'cells':
            data_dict = load_HIVNH(args)


    elif args.dataset == 'COVID':
        if args.feature_type == 'cells':
            args.dataset_dir = '/home/kasm-user/Desktop/volumes/nfs-hdd-storage/chronos/exp2_v2/mid_variable/COVID'
            args.instance_f = next(iter(data_dict[0]))[0].shape[-1]
            data_dict = load_COVID(args)
    else:
        raise ValueError('dataset not implemented')
    
    if args.phase in ['classification','clustering']:
        if args.validation:
            train_dataset,valid_dataset,test_dataset = Dataset_Wrapper(args,data_dict)
        else:
            train_dataset,test_dataset = Dataset_Wrapper(args,data_dict)
    elif (args.phase == 'evaluation'):
        # train_dataset,valid_dataset,test_dataset = Dataset_Wrapper(args,data_dict)
        train_dataset,valid_dataset,test_dataset = data_dict[0],data_dict[1],data_dict[2]
        
    if args.model == 'mil_v0':

        if (args.dataset not in ['COVID']) + (len(args.selected_panels) == 1)*(args.dataset in ['COVID']):
            args.block_settings = ['MLP',args.mil_v0_bag_logits_aggr,'ratio']
            args.instance_mlp_spec = []

            for i in range(len(args.inst_mlp_setting)):
                args.instance_mlp_spec.append(args.inst_mlp_setting[i])
                args.instance_mlp_spec.append(args.dropout)
            args.instance_mlp_spec.append(args.n_classes+args.ashbin)
            model = FlowMIL(input_spec=(None, args.instance_f), instance_weight_spec=None,bag_spec=None,output_spec=args.n_classes,
                instance_mlp_spec=args.instance_mlp_spec,block_settings=args.block_settings,final_act=args.final_act,instance_final_act=args.instance_final_act,
                instance_supervision=False,seed=42,att_normalization = None,pre_normalization=False,ashbin=args.ashbin)
            loss_weights = [1]

            if args.dataset in ['aml_2015']:
                metrics = [tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()]
                losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(reduction='sum')}
                assert args.final_act == None, 'for AML dataset, no final_act should be used' 
                if args.aml_instance_encoder_loading:
                    pretraining_parameter_loading(args,model)
            elif args.dataset in ['CRCLC','hivnh']:
                losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(from_logits = False)}
                metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]
            elif args.dataset in ['COVID']:
                losses = {'tf.identity':channel_weighted_binary_crossentropy([0.1,1,1])}
                metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]

            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,weight_decay=args.weight_decay)
            model.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights,metrics = metrics)     
        elif (len(args.selected_panels) > 1)*(args.dataset in ['COVID']):
            COVID_model_attr_collection(args)
            args.block_settings = ['MLP','mean','ratio']
            model = FlowMIL_multi_tubes(input_spec=(None, args.instance_f), instance_weights_spec=None,bag_spec=None,output_spec=args.n_classes,
                    instance_mlps_spec=args.instance_mlps_spec,block_settings=args.block_settings,final_act=args.final_act,instance_final_act=args.instance_final_act,
                    instance_supervision=False,seed=42,att_normalization = None,pre_normalization=False,ashbin=args.ashbin)
            if args.inst_mlp_pretraining_loading:
                # when you start runing this mode, ensure you have collected trained single tube models
                model = multi_tube_model_loading(model,args.selected_panels,args.n_split,args)
            losses = {'tf.identity':channel_weighted_binary_crossentropy([0.1,1,1])}
            metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]
            loss_weights = [1]
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,weight_decay=args.weight_decay,clipnorm=1.0)
            model.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights,metrics = metrics)
        else:
            raise NotImplementedError("Provided dataset not implemented for MIL v0")

    elif args.model == 'abmil_sh':
        if (args.dataset not in ['COVID']) + (len(args.selected_panels) == 1)*(args.dataset in ['COVID']):
            args.instance_mlp_spec = []
            for i in range(len(args.inst_mlp_setting)-1):
                args.instance_mlp_spec.append(args.inst_mlp_setting[i])
                args.instance_mlp_spec.append(args.dropout)
            args.instance_mlp_spec.append(args.inst_mlp_setting[-1])
            args.instance_weight_spec=[]
            for i in range(len(args.inst_att_setting)):
                args.instance_weight_spec.append(args.inst_att_setting[i])
                args.instance_weight_spec.append(args.dropout)
            args.instance_weight_spec.append(args.n_classes)
            args.bag_spec = [args.instance_mlp_spec[-1],1]
            args.block_settings = ['MLP',args.abmil_att_encoding,'class_specific_mlp']
            model = FlowMIL(input_spec=(None, args.instance_f), instance_weight_spec=args.instance_weight_spec,bag_spec=args.bag_spec,output_spec=args.n_classes,
                    instance_mlp_spec=args.instance_mlp_spec,block_settings=args.block_settings,final_act=args.final_act,instance_final_act=args.instance_final_act,
                    instance_supervision=False,seed=42,att_normalization = args.att_normalization,pre_normalization=False,instance_att_act=args.instance_att_act)
            loss_weights = [1]
            if args.dataset in ['aml_2015']:
                metrics = [tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()]
                losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(reduction='sum')}
                assert args.final_act == None, 'for AML dataset, no final_act should be used' 
                if args.aml_instance_encoder_loading:
                    pretraining_parameter_loading(args,model)
            elif args.dataset in ['CRCLC','hivnh']:
                losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(from_logits = False)}
                metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]
            elif args.dataset in ['COVID']:
                losses = {'tf.identity':channel_weighted_binary_crossentropy([0.1,1,1])}
                metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,weight_decay=args.weight_decay)
            model.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights,metrics = metrics)
        elif (len(args.selected_panels) > 1)*(args.dataset in ['COVID']):
            raise NotImplementedError("Multi-tube ABMIL-sh has not be implemented yet")
        else:
            raise NotImplementedError("Provided dataset not implemented for ABMIL-sh")

    elif args.model == 'simple_mlp':
        #currently only work for AML 
        if args.dataset == 'aml_2015':
            args.output_activation = 'sigmoid'
            args.mlp_spec = [32,0.5,16,0.5,3]
            model = MLP(input_spec=tf.TensorSpec(shape=(None,args.instance_f), dtype=tf.float32), model_spec=args.mlp_spec,output_activation=args.output_activation,bn=True)
            losses = tf.keras.losses.BinaryCrossentropy(from_logits = False)
            metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]
            model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,weight_decay=args.weight_decay),metrics = metrics)
    
    elif args.model == 'abmil_mh':
        if (args.dataset not in ['COVID']) + (len(args.selected_panels) == 1)*(args.dataset in ['COVID']):
            args.instance_mlp_spec = []
            for i in range(len(args.inst_mlp_setting)-1):
                args.instance_mlp_spec.append(args.inst_mlp_setting[i])
                args.instance_mlp_spec.append(args.dropout)
            args.instance_mlp_spec.append(args.inst_mlp_setting[-1])
            args.instance_weight_spec=[]
            for i in range(len(args.inst_att_setting)-1):
                args.instance_weight_spec.append(args.inst_att_setting[i])
                args.instance_weight_spec.append(args.dropout)
            args.instance_weight_spec.append(args.inst_att_setting[-1])
            x = 0.5
            args.bag_spec = [args.instance_mlp_spec[-1], x, 4, x, args.n_classes]
            args.block_settings = ['MLP',args.abmil_att_encoding,'MLP_multiple_head']
            model = FlowMIL(input_spec=(None, args.instance_f), instance_weight_spec=args.instance_weight_spec,bag_spec=args.bag_spec,output_spec=args.n_classes,
                    instance_mlp_spec=args.instance_mlp_spec,block_settings=args.block_settings,final_act=args.final_act,instance_final_act=args.instance_final_act,
                    instance_supervision=False,seed=42,att_normalization = args.att_normalization,pre_normalization=False,instance_att_act=args.instance_att_act)
            
            if args.dataset in ['aml_2015']:
                if args.att_regularization_method is None:
                    metrics = [tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()]
                    losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(reduction='sum')}
                    loss_weights = [1]
                elif (len(model.output)==2)*(args.att_regularization_method is not None):
                    if args.l2_reg_weight > 0:
                        losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(from_logits = False),
                                'tf.identity_1':UnsupervisedMinimizationL2(model=model,l2_weight=args.l2_reg_weight)}
                    else:
                        losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(from_logits = False),
                                'tf.identity_1':UnsupervisedMinimization()}
                    loss_weights = args.loss_weights
                    metrics = {'tf.identity':[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()]}
                assert args.final_act == None, 'for AML dataset, no final_act should be used' 
                if args.aml_instance_encoder_loading:
                    pretraining_parameter_loading(args,model)
                
            elif args.dataset in ['CRCLC','hivnh']:
                if args.att_regularization_method is None:
                    losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(from_logits = False)}
                    metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]
                    loss_weights = [1]
                elif (len(model.output)==2)*(args.att_regularization_method is not None):
                    if args.l2_reg_weight > 0:
                        losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(from_logits = False),
                                'tf.identity_1':UnsupervisedMinimizationL2(model=model,l2_weight=args.l2_reg_weight)}
                    else:
                        losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(from_logits = False),
                                'tf.identity_1':UnsupervisedMinimization()}
                    metrics = {'tf.identity':[tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]}
                    loss_weights = args.loss_weights
            elif args.dataset in ['COVID']:
                if args.att_regularization_method is None:
                    losses = {'tf.identity':channel_weighted_binary_crossentropy([0.1,1,1])}
                    metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]
                    loss_weights = [1]
                elif (len(model.output)==2)*(args.att_regularization_method is not None):
                    if args.l2_reg_weight > 0:
                        losses = {'tf.identity':channel_weighted_binary_crossentropy([0.1,1,1]),
                                'tf.identity_1':UnsupervisedMinimizationL2(model=model,l2_weight=args.l2_reg_weight)}
                    else:
                        losses = {'tf.identity':channel_weighted_binary_crossentropy([0.1,1,1]),
                                'tf.identity_1':UnsupervisedMinimization()}
                    metrics = {'tf.identity':[tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]}
                    loss_weights = args.loss_weights
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,weight_decay=args.weight_decay)
            model.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights,metrics = metrics)
        elif (len(args.selected_panels) > 1)*(args.dataset in ['COVID']):
            raise NotImplementedError("Multi-tube ABMIL-mh has not be implemented yet")
        else:
            raise NotImplementedError("Provided dataset not implemented for ABMIL-mh")     

    elif args.model == 'gmm':
        #TODO clean up the gmm part
        if args.phase == 'clustering':    
            model = GMM(input_spec=tf.TensorSpec(shape=(None,None, args.instance_f), dtype=tf.float32),unit=args.unit,GRF_encoding_method=args.GRF_encoding_method,
                        maximum_normalization= False,phase= 'clustering',return_features_var=args.punishment_on_chol)
            kmeans_initialization = True
            if (kmeans_initialization)*(args.GRF_encoding_method == 'GRF'):
                # clusters = np.load('/home/local/zding20/exp2_v2/mid_variable/AML_2015/models_old/kmeans_centers64.npy')
                import joblib
                # conventional_gmm_model = joblib.load('/home/kasm-user/Desktop/volumes/nfs-hdd-storage/chronos/exp2_v2/mid_variable/result_fcm_script_v3_reg_checks/CRCLC/direct_clustering/4_gmm_model.pkl')
                # model.get_layer('gaussian_radial_basis_layer').l.assign(conventional_gmm_model.means_.T)
                # model.get_layer('gaussian_radial_basis_layer').s.assign(1 / conventional_gmm_model.precisions_cholesky_.T)
                conventional_kmeans_model = joblib.load(args.gmm_initialization_loading_dir)                
                model.get_layer('gaussian_radial_basis_layer').l.assign(conventional_kmeans_model.cluster_centers_.T)
            if args.punishment_on_chol:
                if args.GRF_encoding_method == 'GRF':
                    losses = {'tf.identity':UnsupervisedNegLogLikelihood(model=model,std_weight=0.5,lb=2.0,ub=4.0,centroid_dist_weight=0.0,loss_style=2)}
                    loss_weights = [1]
                elif args.GRF_encoding_method == 'mlp':
                    losses = {'tf.identity':NegLogLikelihood(),'tf.identity_2':UnsupervisedMinimization()}
                    loss_weights = [1,0.1]
            else:
                losses = {'tf.identity':NegLogLikelihood()}
                loss_weights = [1]
            metrics = []

        elif args.phase in ['classification','evaluation']:
            assert args.pretraining_loading_dir is not None
            model = GMM(input_spec=tf.TensorSpec(shape=(None,None, args.instance_f), dtype=tf.float32),unit=args.unit,GRF_encoding_method=args.GRF_encoding_method,
                        maximum_normalization= False,phase= 'classification',return_features_var=args.punishment_on_chol,classification_head_spec=[6,0.5,4,0.5,args.n_classes],
                        cluster_model_trainable=False,inst_logits_normalization=True)
            kmeans_initialization = False
            if (kmeans_initialization)*(args.GRF_encoding_method == 'GRF'):
                # clusters = np.load('/home/local/zding20/exp2_v2/mid_variable/AML_2015/models_old/kmeans_centers64.npy')
                import joblib
                conventional_gmm_model = joblib.load('/home/kasm-user/Desktop/volumes/nfs-hdd-storage/chronos/exp2_v2/mid_variable/result_fcm_script_v3_reg_checks/CRCLC/direct_clustering/64_gmm_model.pkl')
                model.get_layer('gaussian_radial_basis_layer').l.assign(conventional_gmm_model.means_.T)
                model.get_layer('gaussian_radial_basis_layer').s.assign(1 / conventional_gmm_model.precisions_cholesky_.T)
                # conventional_kmeans_model = joblib.load(args.gmm_initialization_loading_dir)
                # model.get_layer('gaussian_radial_basis_layer').l.assign(conventional_kmeans_model.cluster_centers_.T)
            
            gmm_classification_initialization(args,model)
            losses = {'tf.identity':tf.keras.losses.BinaryCrossentropy(from_logits = False)}
            metrics = [tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=args.n_classes,average='macro'),tf.keras.metrics.CategoricalAccuracy()]
            loss_weights = [1]
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,weight_decay=args.weight_decay)
        model.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights,metrics = metrics)
    
    if args.phase in ['classification','clustering']:
        with open(os.path.join(args.output_dir,'settings.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    #add tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.output_dir, histogram_freq=1)
    #add checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output_dir,'model'), save_weights_only=True, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
    
    if args.phase == 'classification':
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,restore_best_weights=True,mode='min',start_from_epoch=10)
    elif args.phase == 'clustering':
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,restore_best_weights=True,mode='min',start_from_epoch=10,min_delta=1e-4)

    if args.phase == 'classification':
        if args.validation:
            if args.dataset == 'aml_2015':
                model.fit(train_dataset, epochs=args.max_epoch, validation_data=valid_dataset, callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback,TestCallback(test_dataset,task=args.task, dataset_type = args.dataset,feature_type = args.feature_type),logcallback(args.task,feature_type = args.feature_type)],verbose=2)

            elif args.dataset == 'COVID':
                if not args.inst_mlp_pretraining_loading:
                    model.fit(train_dataset, epochs=args.max_epoch, validation_data=valid_dataset, callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback,TestCallback(test_dataset,task=args.task,feature_type = args.feature_type, using_unsupervised_loss_target=(args.att_regularization_method is not None)),logcallback(args.task,args.feature_type,(args.att_regularization_method is not None))],verbose=2)
                else:
                    model.fit(train_dataset, epochs=args.max_epoch, validation_data=valid_dataset, callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback,TestCallback(test_dataset,task=args.task,feature_type = args.feature_type, using_unsupervised_loss_target=(args.att_regularization_method is not None)),logcallback(args.task,args.feature_type,(args.att_regularization_method is not None))],verbose=2)
            else:
                
                model.fit(train_dataset, epochs=args.max_epoch, validation_data=valid_dataset, callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback,TestCallback(test_dataset,task=args.task,feature_type = args.feature_type, using_unsupervised_loss_target=(args.att_regularization_method is not None)),logcallback(args.task,args.feature_type,(args.att_regularization_method is not None))],verbose=2)
        else:
            model.fit(train_dataset, epochs=args.max_epoch,validation_data=test_dataset, callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback],verbose = 2)
    elif args.phase == 'feature_extraction':
        model.load_weights(os.path.join(args.model_loading_dir,'model'))
        wrapped_model = MapTensorFunctionRagged(model)
        mode = 'bag'
        if mode == 'inst':
            inst_feature_extraction(wrapped_model,train_dataset,os.path.join(args.output_dir,'trainset'))
            inst_feature_extraction(wrapped_model,test_dataset,os.path.join(args.output_dir,'testset'))
            inst_feature_extraction(wrapped_model,valid_dataset,os.path.join(args.output_dir,'valset'))
        elif mode == 'bag':
            inst_feature_extraction(wrapped_model,train_dataset,os.path.join(args.output_dir,'inst_mean_trainset.h5'),mode='bag')
            inst_feature_extraction(wrapped_model,test_dataset,os.path.join(args.output_dir,'inst_mean_testset.h5'),mode='bag')
            inst_feature_extraction(wrapped_model,valid_dataset,os.path.join(args.output_dir,'inst_mean_valset.h5'),mode='bag')
    elif args.phase == 'clustering':
        if args.validation:
            model.fit(train_dataset, epochs=args.max_epoch, validation_data=train_dataset,callbacks = [early_stopping_callback,checkpoint_callback], verbose=2)

    #collect the results
    result_collection = args.result_collection
    inst_att_extraction = args.inst_att_extraction
    predict_value_collection = args.predict_value_collection
    inst_att_extraction_multi_tube = args.inst_att_extraction_multi_tube
    
    if result_collection:
        assert args.phase not in ['evaluation'], 'not supported for evaluation phase due to the dataset format'
        sample_level_result_collection(args, model, train_dataset, valid_dataset, test_dataset)
    if predict_value_collection:
        if args.phase in ['classification','clustering']:
            raise Warning("current sample names is not included in dataset")
        sample_level_logits_collection(args, model, train_dataset, valid_dataset, test_dataset)
    if inst_att_extraction:
        inst_level_logits_collection(args, model, train_dataset, valid_dataset, test_dataset)
    if inst_att_extraction_multi_tube:
        inst_level_logits_collection_multi_tube(args, model, train_dataset, valid_dataset, test_dataset)

if __name__ == '__main__':
    train_process()