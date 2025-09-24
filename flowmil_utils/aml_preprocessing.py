import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import h5py
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import umap
import pickle

def AML_2015_raw_AML_filtering(args):
    file_path = args.raw_data_dir
    file_list = os.listdir(file_path)
    processed_path = args.data_saving_dir
    os.makedirs(processed_path,exist_ok = True)
    column_name_list = []
    for fn in file_list:
        tmp_sample = pd.read_csv(os.path.join(file_path,fn),header=None,)
        tmp_sample = tmp_sample.iloc[1:]
        column_names = tmp_sample.iloc[0][0].split('\t')
        column_name_list.append(column_names)
        tmp_sample.iloc[1][0].split('\t')
        fcs_data = [tmp_sample.iloc[i][0].split('\t') for i in range(tmp_sample.shape[0])]
        fcs_data = np.array(fcs_data[1:],dtype = np.float16)
        # selected_cols = [column_names.index(i) for i in ['CD34','CD45']]
        # gating_data = fcs_data[:,selected_cols]
        gating_data = fcs_data
        gating_data = np.arcsinh(gating_data / 5)
        threshold_A = 1.85
        # threshold_B = [0,6.25]
        threshold_B = 6.25
        channel_A_name = 'CD34'
        channel_B_name = 'CD45'
        plotting =False
        if plotting:
            gating_data = pd.DataFrame(gating_data, columns=['CD34', 'CD45'])

            gating_data['Gate'] = np.select(
                [
                    (gating_data[channel_A_name] >= threshold_A) & (gating_data[channel_B_name] >= threshold_B),
                    (gating_data[channel_A_name] >= threshold_A) & (gating_data[channel_B_name] < threshold_B),
                    (gating_data[channel_A_name] < threshold_A) & (gating_data[channel_B_name] >= threshold_B),
                    (gating_data[channel_A_name] < threshold_A) & (gating_data[channel_B_name] < threshold_B)
                ],
                ['Gate 1', 'Gate 2', 'Gate 3', 'Gate 4']
                )
            proportions = [gating_data[gating_data.Gate =='Gate '+str(i)].shape[0] for i in range(1,5)]
            proportions = {'Gate '+str(i+1):proportions[i]/np.array(proportions).sum() for i in range(4)}
            plt.figure(figsize=(8, 6))
            plt.scatter(gating_data[channel_A_name], gating_data[channel_B_name], c=gating_data['Gate'].astype('category').cat.codes, alpha=0.5)
            # sns.kdeplot(x=gating_data[channel_A_name], y=gating_data[channel_B_name], cmap="Reds", fill=True, thresh=0, levels=100, alpha=0.5)
            plt.colorbar(label='Gate')
            plt.xlabel(channel_A_name)
            plt.ylabel(channel_B_name)
            plt.title( fn+'FCM Gating Scatter Plot')
            positions = [(0.85, 0.85),(0.85, 0.05),(0.05, 0.85),(0.05, 0.05)]
            for (gate, proportion), (x_pos, y_pos) in zip(proportions.items(), positions):
                plt.annotate(f"{gate}: {proportion:.4f}", xy=(x_pos, y_pos), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
            plt.show()
        filtering_blasts = True
        if filtering_blasts:
            # filtering out blasts by gathering cells with CD34+ and CD45+ greater than threshould
            blasts = gating_data[(gating_data[:,column_names.index(channel_A_name)] >= threshold_A) * (gating_data[:,column_names.index(channel_B_name)] <= threshold_B),:]
            #check if composition of blasts larger than 5% of total cells, if not then skip this sample
            if blasts.shape[0]/gating_data.shape[0] < 0.1:
                continue
            #save blasts
            blast_csv = pd.DataFrame(blasts,columns=column_names)
            blast_csv.to_csv(os.path.join(processed_path,fn.split('.')[0]+'_blasts.csv'),index=False)
    return column_name_list

def filtering_common_features(column_name_list):
    common_features = set(column_name_list[0])
    for i in column_name_list[1:]:
        common_features = common_features.intersection(set(i))
    common_features = list(common_features)
    return column_name_list

def blasts_collection(sample_list,processed_path,saving_path):
    # blasts collection
    file_list = os.listdir(processed_path)
    blasts_collection = []
    selected_features = ['pZap70-Syk', 'HLA-DR', 'CD7', 'DNA1', 'pAKT', 'CD34', 'CD123',
                         'BC2', 'BC6', 'CD19', 'CD11b', 'BC4', 'BC5', 'CD44',
                         'CD45', 'CD47', 'pSTAT1', 'pc-Cbl', 'CD38', 'BC3', 'pAMPK', 'CD33',
                         'cCaspase3', 'pP38', 'Viability', 'pErk1-2', 'DNA2', 'Cell_length',
                         'pCREB', 'pS6', 'BC1', 'pSTAT5', 'pRb', 'CD117', 'CD41', 'CD64', 
                         'p4EBP1', 'pPLCg2', 'CD15', 'pSTAT3', 'CD3']
    for fn in file_list:
        if fn.split('_')[0] not in sample_list:
            continue
        tmp_sample = pd.read_csv(os.path.join(processed_path,fn))
        tmp_sample = tmp_sample[selected_features]
        # transform to array
        tmp_sample = np.array(tmp_sample,dtype=np.float16)
        blasts_collection.append(tmp_sample)
    if len(blasts_collection)==0:
        warnings.warn("no sample is collected ", UserWarning)
    blasts_collection = np.concatenate(blasts_collection,axis=0)
    print('total blasts number: ',blasts_collection.shape[0])
    np.save(saving_path,blasts_collection,allow_pickle=True)

def healthy_cells_collection(sample_list,processed_path,saving_path):
    file_path = processed_path
    file_list = os.listdir(file_path)
    healthy_cells_list = []
    selected_features = ['pZap70-Syk', 'HLA-DR', 'CD7', 'DNA1', 'pAKT', 'CD34', 'CD123',
                         'BC2', 'BC6', 'CD19', 'CD11b', 'BC4', 'BC5', 'CD44',
                         'CD45', 'CD47', 'pSTAT1', 'pc-Cbl', 'CD38', 'BC3', 'pAMPK', 'CD33',
                         'cCaspase3', 'pP38', 'Viability', 'pErk1-2', 'DNA2', 'Cell_length',
                         'pCREB', 'pS6', 'BC1', 'pSTAT5', 'pRb', 'CD117', 'CD41', 'CD64', 
                         'p4EBP1', 'pPLCg2', 'CD15', 'pSTAT3', 'CD3']
    for fn in file_list:
        if fn.split('_')[0] not in sample_list:
            continue
        tmp_sample = pd.read_csv(os.path.join(file_path,fn),header=None,)
        tmp_sample = tmp_sample.iloc[1:]
        column_names = tmp_sample.iloc[0][0].split('\t')
        fcs_data = [tmp_sample.iloc[i][0].split('\t') for i in range(tmp_sample.shape[0])]
        fcs_data = np.array(fcs_data[1:],dtype = np.float16)
        fcs_data = pd.DataFrame(fcs_data,columns=column_names)
        fcs_data = fcs_data[selected_features]
        fcs_data = np.array(fcs_data,dtype=np.float16)
        fcs_data = np.arcsinh(fcs_data / 5)
        healthy_cells_list.append(fcs_data)
    if len(healthy_cells_list)== 0:
        warnings.warn("no sample is collected ", UserWarning)
    healthy_cells_list = np.concatenate(healthy_cells_list,axis=0)
    print('total healthy cells number: ',healthy_cells_list.shape[0])
    np.save(saving_path,healthy_cells_list,allow_pickle=True)

def trainset_generation(data_list, n_sample,save_dir, n_components=25,cell_range=[10000,50000],mrd_ratio_range = [0,0.1],sample_ratio = 0.5,mrd_distribution='uniform',mrd_params_dict = None):
    os.makedirs(save_dir,exist_ok=True)
    healthy_cells_mixture = GaussianMixture(n_components=n_components, random_state=42).fit(data_list[0])
    sample_labels = np.random.choice([0,1,2],size = n_sample,p = [1-sample_ratio,sample_ratio/2,sample_ratio/2])
    for ns in range(n_sample):
        n_cells = np.random.randint(cell_range[0],cell_range[1])
        sample_label = sample_labels[ns]
        if mrd_distribution == 'uniform':
            mrd_ratio = np.random.uniform(mrd_ratio_range[0],mrd_ratio_range[1])
        elif mrd_distribution == 'lognormal':
            mrd_ratio = np.random.lognormal(mrd_params_dict['mean'],mrd_params_dict['std'])
            while mrd_ratio < mrd_ratio_range[0] or mrd_ratio > mrd_ratio_range[1]:
                mrd_ratio = np.random.lognormal(mrd_params_dict['mean'],mrd_params_dict['std'])        
        if sample_label == 0:
            sample = healthy_cells_mixture.sample(n_cells)[0]
            mrd_ratio_target = [1,0,0]
            cell_labels = np.zeros(n_cells)
        elif sample_label == 1:
            n_health_cells = int(n_cells * (1-mrd_ratio))
            n_mrd_cells = n_cells - n_health_cells
            healthy_cells = healthy_cells_mixture.sample(n_health_cells)[0]
            mrd_cells = data_list[1][np.random.randint(0,len(data_list[1]),n_mrd_cells)]
            sample = np.concatenate([healthy_cells,mrd_cells])
            cell_labels = np.concatenate([np.zeros(n_health_cells),np.ones(n_mrd_cells)])
            mrd_ratio_target = [1-mrd_ratio,mrd_ratio,0]
        else:
            n_health_cells = int(n_cells * (1-mrd_ratio))
            n_mrd_cells = n_cells - n_health_cells
            healthy_cells = healthy_cells_mixture.sample(n_health_cells)[0]
            mrd_cells = data_list[2][np.random.randint(0,len(data_list[2]),n_mrd_cells)]
            sample = np.concatenate([healthy_cells,mrd_cells])
            cell_labels = np.concatenate([np.zeros(n_health_cells),2*np.ones(n_mrd_cells)])
            mrd_ratio_target = [1-mrd_ratio,0,mrd_ratio]
        sample_save_dir = os.path.join(save_dir,'sample_'+str(ns)+'.h5')
        with h5py.File(sample_save_dir,'w') as f:
            f.create_dataset('cells',data=sample)
            f.create_dataset('mrd_ratio',data=mrd_ratio_target)
            f.create_dataset('sample_cell_labels',data=cell_labels)
            f.create_dataset('sample_labels',data=sample_label)

def real_dataset_generation(data_list, n_sample,save_dir, cell_range=[10000,50000],mrd_ratio_range = [0,0.1],sample_ratio = 0.5,mrd_distribution='uniform',mrd_params_dict = None):
    os.makedirs(save_dir,exist_ok=True)
    sample_labels = np.random.choice([0,1,2],size = n_sample,p = [1-sample_ratio,sample_ratio/2,sample_ratio/2])
    for ns in range(n_sample):
        n_cells = np.random.randint(cell_range[0],cell_range[1])
        sample_label = sample_labels[ns]
        if mrd_distribution == 'uniform':
            mrd_ratio = np.random.uniform(mrd_ratio_range[0],mrd_ratio_range[1])
        elif mrd_distribution == 'lognormal':
            mrd_ratio = np.random.lognormal(mrd_params_dict['mean'],mrd_params_dict['std'])
            while mrd_ratio < mrd_ratio_range[0] or mrd_ratio > mrd_ratio_range[1]:
                mrd_ratio = np.random.lognormal(mrd_params_dict['mean'],mrd_params_dict['std'])
        if sample_label == 0:
            sample = data_list[0][np.random.randint(0,len(data_list[0]),n_cells)]
            mrd_ratio_target = [1,0,0]
            cell_labels = np.zeros(n_cells)
        elif sample_label == 1:
            n_health_cells = int(n_cells * (1-mrd_ratio))
            n_mrd_cells = n_cells - n_health_cells
            healthy_cells = data_list[0][np.random.randint(0,len(data_list[0]),n_health_cells)]
            mrd_cells = data_list[1][np.random.randint(0,len(data_list[1]),n_mrd_cells)]
            sample = np.concatenate([healthy_cells,mrd_cells])
            cell_labels = np.concatenate([np.zeros(n_health_cells),np.ones(n_mrd_cells)])
            mrd_ratio_target = [1-mrd_ratio,mrd_ratio,0]
        else:
            n_health_cells = int(n_cells * (1-mrd_ratio))
            n_mrd_cells = n_cells - n_health_cells
            healthy_cells = data_list[0][np.random.randint(0,len(data_list[0]),n_health_cells)]
            mrd_cells = data_list[2][np.random.randint(0,len(data_list[2]),n_mrd_cells)]
            sample = np.concatenate([healthy_cells,mrd_cells])
            cell_labels = np.concatenate([np.zeros(n_health_cells),2*np.ones(n_mrd_cells)])
            mrd_ratio_target = [1-mrd_ratio,0,mrd_ratio]
        sample_save_dir = os.path.join(save_dir,'sample_'+str(ns)+'.h5')
        with h5py.File(sample_save_dir,'w') as f:
            f.create_dataset('cells',data=sample)
            f.create_dataset('mrd_ratio',data=mrd_ratio_target)
            f.create_dataset('sample_cell_labels',data=cell_labels)
            f.create_dataset('sample_labels',data=sample_label)