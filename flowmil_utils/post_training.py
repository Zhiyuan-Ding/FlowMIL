import numpy as np
import json

# filtering experiments based on dict info
def result_subset_filtering(folders,filtering_dicts,unexist_list=None):
    result_folders = []
    flags = len(filtering_dicts)
    for f in folders:
        current_flag = 0
        with open(f+'/settings.json') as json_file:
            result = json.load(json_file)
        for filtering_dicts_key in filtering_dicts:
            if filtering_dicts_key in result.keys():
                if (filtering_dicts[filtering_dicts_key] != result[filtering_dicts_key]):
                    break
                else:
                    current_flag +=1
        if unexist_list is not None and any(x in result.keys() for x in unexist_list):
            current_flag = 0
        if current_flag == flags:
            result_folders.append(f)
    return result_folders

#collecting results based on filtered exps
def result_collection(folders):
    setting_list = []
    result_dict = {}
    setting_dict = {}
    best_result = {'mean':[],'std':[],'setting':{}}

    folders = sorted(folders)
    for f in folders:
        #gather the settings
        with open(f+'/settings.json') as json_file:
            current_setting= json.load(json_file)
        setting_list.append(current_setting)
    #gather results with the same n_exp
    unique_setting_list = []
    for ns in range(len(setting_list)):
        cs_n_exp = setting_list[ns]['n_exp']
        with open(folders[ns]+'/best_results.json') as json_file:
            current_result = json.load(json_file)
        if cs_n_exp not in result_dict:
            result_dict[cs_n_exp] = [current_result]
            unique_setting_list.append(setting_list[ns])
            setting_dict[cs_n_exp] = [setting_list[ns]]
        else:
            result_dict[cs_n_exp].append(current_result)
            setting_dict[cs_n_exp].append(setting_list[ns])
    #calculate the mean and std
    all_means = []
    all_stds = []
    output_dir_list = []
    for n_exp in result_dict.keys():
        current_results = []
        for nr in range(len(result_dict[n_exp])):
            current_results.append([result_dict[n_exp][nr]['test'],result_dict[n_exp][nr]['valid']])
        current_results = np.array(current_results)
        current_mean = np.mean(current_results,axis=0).reshape(-1)
        current_std = np.std(current_results,axis=0).reshape(-1)
        all_means.append(current_mean)
        all_stds.append(current_std)
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    #find the best result based on min valid loss
    picked_idx = all_means.shape[-1]//2
    best_idx = np.argmin(all_means[:,picked_idx],axis=0)
    best_result['mean'] = all_means[best_idx]
    best_result['std'] = all_stds[best_idx]
    best_result['setting'] = unique_setting_list[best_idx]
    best_result['CI'] = [best_result['mean']-2.776*best_result['std'],best_result['mean']+2.776*best_result['std']]
    best_result['5cv'] = np.array([k['test'] for k in result_dict[list(result_dict.keys())[best_idx]]])
    best_setting_dict = setting_dict[unique_setting_list[best_idx]['n_exp']]
    return best_result, {'mean':all_means,'std':all_stds,'setting':unique_setting_list}, result_dict, setting_dict, best_setting_dict