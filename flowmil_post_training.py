import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import pickle
import argparse
from flowmil_utils.post_training import *


parser = argparse.ArgumentParser(description='Post training analysis profile for FlowMIL')
# general settings
parser.add_argument('--stat_training_results', action='store_true', default=False, help='searching and collecting hyperparameter searching results')
parser.add_argument('--instance_logits_extraction', action='store_true', default=False, help='whether or not the instance feature is reextracted')
parser.add_argument('--output_dir',type = str, default = '', help='output dir')

# result collection & stat settings
parser.add_argument('--results_collection_dir',type = str, default = None, help='pretraining model loading dir')
parser.add_argument("--filtering_dict", type=json.loads, default={}, help="json format of dicts, which contains variables to filtering on for experiments")

# instance logits extraction settings
parser.add_argument('--from_filtered_result', action='store_true', default=False, help='whether or not to use the ')
parser.add_argument('--inst_extraction_exp_code',type = str, default = None, help='feature extraction exp code')
parser.add_argument('--inst_extraction_dataset',type = str, default = 'CRCLC', help='feature extraction dataset')
parser.add_argument('--inst_extraction_model',type = str, default = 'mil_v0', help='feature extraction model')
parser.add_argument('--inst_extraction_n_exp',type = int, default = 0, help='number id of exp')
parser.add_argument('--inst_extraction_task',type = str, default = 'mil_v0', help='feature extraction task')
parser.add_argument('--inst_extraction_dropout', type=float, default=0.0, help='feature extraction dropout rate')
parser.add_argument('--inst_extraction_inst_mlp_setting', type=int, nargs='+', default=[32,16,8,4], help='feature extraction instance mlp setting')
parser.add_argument('--inst_extraction_best_result_dirs', type=str, nargs='+', default=None, help='feature extraction best result dirs')
parser.add_argument('--inst_extraction_mil_v0_bag_logits_aggr', default='mean',type=str, help='feature extraction method for mil_v0 aggr, which only works when normalize is not used')
parser.add_argument('--inst_extraction_instance_final_act', default='sigmoid',type=str, help='feature extraction instance final activation')
parser.add_argument('--inst_extraction_inst_att_setting', type=int, nargs='+', default=[16,8], help='feature extraction instance att mlp setting')
parser.add_argument('--inst_extraction_abmil_att_encoding', default='attention_dependent',type=str, help='feature extraction instance attention encoding method for abmil')
parser.add_argument('--inst_extraction_att_normalization', default='normalize_along_instance',type=str, help='feature extraction att normalization strategy')
parser.add_argument('--inst_extraction_instance_att_act', default='sigmoid',type=str, help='feature extraction inst att final act')


def post_training_process():

    args = parser.parse_args()
    print(args)
    if args.stat_training_results:
        folders = os.listdir(args.results_collection_dir)
        folders = [x for x in folders if 'best_results.json' in os.listdir(os.path.join(args.results_collection_dir,x))]
        folders = [os.path.join(args.results_collection_dir,f) for f in folders]
        filtering_dicts = args.filtering_dict
        filtered_folders = result_subset_filtering(folders,filtering_dicts)
        best_result,results,result_dict,setting_dict, best_settings = result_collection(filtered_folders)    
        n_splits = [x['n_split'] for x in best_settings]
        best_result_dirs = [x['output_dir'] for x in best_settings]
        order = np.argsort(n_splits)
        best_result_dirs = [best_result_dirs[o] for o in order]
    
    if args.instance_logits_extraction:
        
        exp_code = args.inst_extraction_exp_code
        if args.from_filtered_result:
            assert args.stat_training_results
            loaded_setting = best_settings[0]
            assert len(best_result_dirs) == 5, 'please check if all 5 cross-validation finished'
            assert loaded_setting['dataset'] not in ['COVID','aml_2015'], 'not implement for these dataset yet, please refer to manual mode'
            for n_split in range(5):
                    prefix = 'nohup python flowmil_main.py --dataset %s --dataset_dir %s --model %s --n_exp %d --phase evaluation --task %s --inst_att_extraction --return_sample_info \
                    --n_split %d --dropout %.1f --inst_mlp_setting %s --validation  --output_dir %s --model_loading_dir %s --mil_v0_bag_logits_aggr %s \
                    --exp_code %s --instance_final_act %s  --inst_att_setting %s --abmil_att_encoding %s --att_normalization %s --instance_att_act %s  \
                    ' % (loaded_setting['dataset'],loaded_setting['dataset_dir'], loaded_setting['model'], loaded_setting['n_exp'], loaded_setting['task'],n_split, loaded_setting['dropout'], 
                         ' '.join([str(x) for x in loaded_setting['inst_mlp_setting']]), args.output_dir, best_result_dirs[n_split], loaded_setting['mil_v0_bag_logits_aggr'], 
                        exp_code, loaded_setting['instance_final_act'],' '.join([str(x) for x in loaded_setting['inst_att_setting']]), loaded_setting['abmil_att_encoding'], 
                        loaded_setting['att_normalization'], loaded_setting['instance_att_act'])
                    os.system(prefix)
        else:
            assert args.inst_extraction_dataset not in ['COVID','aml_2015'], 'not implement for these dataset yet, please refer to manual mode'
            assert len(args.inst_extraction_best_result_dirs) == 5
            for n_split in range(5):
                    prefix = 'nohup python flowmil_main.py --dataset %s --model %s --n_exp %d --phase evaluation --task %s --return_sample_info --inst_att_extraction \
                    --n_split %d --dropout %.1f --inst_mlp_setting %s --validation  --output_dir %s --model_loading_dir %s --mil_v0_bag_logits_aggr %s \
                    --exp_code %s --instance_final_act %s  --inst_att_setting %s --abmil_att_encoding %s --att_normalization %s --instance_att_act %s  \
                    ' % (args.inst_extraction_dataset, args.inst_extraction_model, args.inst_extraction_n_exp, args.inst_extraction_task, n_split, args.inst_extraction_dropout, 
                         ' '.join([str(x) for x in args.inst_extraction_inst_mlp_setting]), args.output_dir, args.inst_extraction_best_result_dirs[n_split], args.inst_extraction_mil_v0_bag_logits_aggr, 
                        exp_code, args.inst_extraction_instance_final_act,' '.join([str(x) for x in args.inst_extraction_inst_att_setting]), args.inst_extraction_abmil_att_encoding, 
                        args.inst_extraction_att_normalization, args.inst_extraction_instance_att_act)
                    os.system(prefix)

if __name__ == '__main__':
    post_training_process()