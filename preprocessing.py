import argparse
import numpy as np
import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split,StratifiedKFold
import h5py
import warnings
import flowio
import pickle
from tqdm import tqdm
from flowmil_utils.dataset import *
from flowmil_utils.aml_preprocessing import *
parser = argparse.ArgumentParser(description='Training profile for FlowMIL')
parser.add_argument('--raw_data_dir', type=str, default='', help='raw data loading dir')
parser.add_argument('--data_saving_dir', type=str, default='', help='saving dir for preprocessed dataset')
parser.add_argument('--dataset', type=str, default='CRCLC',choices=['aml_2015','CRCLC','hivnh','COVID'], help='dataset name')
parser.add_argument('--used_feature', type=str, default=None,nargs='+', help='used feature list')
parser.add_argument('--sup_info_dir', type=str, default='./preprocessing_used_files/', help='dir to save the supplementary info')
parser.add_argument('--seed', type=int, default=42, help='rand seed')

#CRCLC specific
parser.add_argument('--CRCLC_channel_info', type=str, default='./preprocessing_used_files/Tcell_channel_info.xlsx', help='channel mapping info for CRCLC raw data')
#HIVNH specific
parser.add_argument('--HIVNH_demographic', type=str, default='./preprocessing_used_files/clinical_data_flow_repository.csv', help='demorgraphic information for hivnh dataset')
parser.add_argument('--HIVNH_stratification', type=bool, default=True, help='if true, the dataset will be processed as a classification task,' \
                                                                                'otherwise, the dataset will be processed as a regression task')
#AML2015 specific
parser.add_argument('--AML2015_gating', default=False,action='store_true', help='if gating blasts from original files')
parser.add_argument('--AML2015_recollection', default=False,action='store_true', help='if recollecting blasts from preprocessed files')
parser.add_argument('--AML2015_simulation_regeneration', default=False,action='store_true', help='regenerate the AML simulated samples')
parser.add_argument('--AML2015_healthy_raw_dir', type=str, default=None, help='dir for raw healthy cells')
parser.add_argument('--AML2015_subsampled_cells', type=int, default=120000, help='subsampled of cells for simulation')
parser.add_argument('--AML2015_simulated_n_components', type=int, default=64, help='simulated GMM kernels')
parser.add_argument('--AML2015_mrd_range', type=float, nargs='+',default=[5e-4,1e-1], help='MRD ratio ranges for generation')
parser.add_argument('--AML2015_cell_range', type=int, nargs='+',default=[3000,30000], help='MRD number of cell range')

#COVID specific
parser.add_argument('--COVID_demographic', type=str, default='./preprocessing_used_files/220115_Demographics_for_raw_data_upload.csv', help='demographic information for COVID dataset')
parser.add_argument('--COVID_dataset_initialization', default=False,action='store_true', help='check the channel info of COVID dataset')
parser.add_argument('--COVID_compute_stat', default=False,action='store_true', help='compute mean and stat of COVID dataset')
parser.add_argument('--COVID_h5_generation', default=False,action='store_true', help='generate the h5 file for COVID dataset')

def CRCLC_preprocessing(args):

    os.makedirs(args.data_saving_dir,exist_ok=True)
    arr = [x for x in os.listdir(args.raw_data_dir) if x.endswith(".fcs")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fio = flowio.FlowData(os.path.join(args.raw_data_dir , arr[0]), ignore_offset_error=True)
    tubes = dict(channels=fio.channels)
    tubes['loader'] = TubeLoader(args.raw_data_dir + arr[0], tube_channels=tubes['channels'])
    fcs_data = list()
    for i in arr:
        fcs_data.append(list(tuple(TubeLoader.load_fcs(os.path.join(args.raw_data_dir , i)))))
    fcs_data, channel_info = list(zip(*fcs_data))
    
    print('Dataset: T-Cell Tumor Classification')
    print(len(channel_info),' samples in total')
    print(len(fcs_data[0][0]), ' channels')
    
    #collecting all features
    feature_list = []
    feature_list_temp = []
    for i in range(len(channel_info)):
        current_channels = channel_info[i]
        current_channels_feature = [current_channels[j]['pnn'] for j in current_channels.keys()]
        current_channels_feature_temp = [current_channels[j]['pns'] for j in current_channels.keys()]
        feature_list.append(current_channels_feature)
        feature_list_temp.append(current_channels_feature_temp)
    
    unique_feature_channel_list = []
    unique_feature_channel_counting = []
    unique_feature_list = []
    unique_channel_list = []
    for i in range(len(feature_list)):
        current_sample_feature = feature_list_temp[i]
        current_sample_channel = feature_list[i]
        for j in range(len(current_sample_channel)):
            current_pair = [current_sample_feature[j],current_sample_channel[j]]
            if current_pair not in unique_feature_channel_list:
                unique_feature_channel_list.append(current_pair)
                unique_feature_channel_counting.append(1)
                unique_feature_list.append(current_sample_feature[j])
                unique_channel_list.append(current_sample_channel[j])
            else:
                pos = unique_feature_channel_list.index(current_pair)
                unique_feature_channel_counting[pos] = unique_feature_channel_counting[pos]+1
    

    #using filtered features

    sifted_feature = pd.read_excel(args.CRCLC_channel_info)
    sifted_feature_exnan = sifted_feature.dropna(how='any')
    mapping_keys = list(set(sifted_feature_exnan['mapping_keys'].values))
    common_feature_dict = {x:sifted_feature_exnan[sifted_feature_exnan.iloc[:,4].astype(str)==x].iloc[:,2].values.tolist() for x in mapping_keys}
    

    args.used_feature = None
    if args.used_feature is None:
        used_feature_list = ['CD45','CCR7','CD45RO','CD69','CD27','CD28','CD127','ICOS','TIGIT','CD57','PD-1','CD8',
                            'HLA-DR','Cisplatin','CD39','CD25','CD161','TCRgd','CD3','CD103','CD4','CD56','Va7-2',
                            'CD14','BC110','BC108','BC106','BC105','BC104','BC103']
        used_feature_list = np.intersect1d(used_feature_list,list(common_feature_dict))
    else:
        used_feature_list = args.used_feature

    used_feature = dict([(key,common_feature_dict[key]) for key in used_feature_list])
    
    corresponding_pos = (-1)*np.ones((len(feature_list_temp),len(used_feature)),dtype=np.int32)
    fs_name = list(used_feature.keys())
    for ns in range(len(feature_list_temp)):
        for nf in range(len(fs_name)):
            current_feature = common_feature_dict[fs_name[nf]]
            for f_var in current_feature:
                if (f_var in feature_list_temp[ns]):
                    corresponding_pos[ns,nf] = feature_list_temp[ns].index(f_var)
    
    # not determined if to deleting some of the features 
    # missing_feature_samples = np.sum(corresponding_pos==-1,axis=1)
    # corresponding_pos2 = corresponding_pos[missing_feature_samples<10,:]
    # missing_sample_feature = (corresponding_pos2==-1).sum(axis=0)
    # corresponding_pos3 = corresponding_pos2[:,missing_sample_feature<5]
    
    incomplete_cases = np.unique(np.where(corresponding_pos==-1)[0])
    print(len(incomplete_cases), ' samples are ignored for missing channels')
    complete_cases = []
    for i in range(len(fcs_data)):
        if not np.isin(i,incomplete_cases):
            complete_cases.append(i)
    
    complete_cases = np.array(complete_cases)
    print(len(complete_cases), ' samples are used for next step')

    #labels generation    
    labels = [x.split('_')[0]=='CRC' for x in arr]
    labels = np.array(labels,dtype=np.int16)
    sample_names = [x.split('.')[0] for x in arr]
    labels = labels[complete_cases]
    
    sample_names = np.array(sample_names)
    sample_names = sample_names[complete_cases]
    fcs_data = np.array(fcs_data,dtype=object)[complete_cases]
    corresponding_pos = corresponding_pos[complete_cases,:]
    recollected_fcs_data = []
    for tube_ind in range(fcs_data.shape[0]):
        recollected_fcs_data.append(fcs_data[tube_ind][:,corresponding_pos[tube_ind,:]])

    fcs_data = np.array(recollected_fcs_data,dtype = object)
    
    channel_stats = [x.shape[-1] for x in fcs_data]
    assert np.std(channel_stats)==0

    for sn in sample_names:
        sample_saving_path = os.path.join(args.data_saving_dir,sn.split('.')[0]+'.h5')
        hf = h5py.File(sample_saving_path,'w')
        hf.create_dataset('cells',data=fcs_data[sample_names==sn][0])
        hf.create_dataset('sample_labels',data=labels[sample_names==sn])
        hf.close()

def HIVNH_preprocessing(args):
        print('start processing hivnh dataset')
        d = pd.read_csv(args.HIVNH_demographic, sep='\t')
        d = pd.DataFrame(np.stack(d.iloc[:, 0].str.split('\t').values, axis=0), columns=['id', 'time', 'dead'])
        d['time'] = d['time'].astype(float)
        d['dead'] = d['dead'].astype(float)

        if args.HIVNH_stratification:
            idx_valid_time = d['time'] > 0
            d['label'] = 'Unknown'
            d.loc[(d['time'] < (4 * 365)) & (d['dead'] == 1), 'label'] = 'Poor Outcome'
            d.loc[d['time'] > (5 * 365), 'label'] = 'Good Outcome'        
            idx_label = d.loc[idx_valid_time, 'label'] != 'Unknown'
            labels = d.loc[idx_valid_time].loc[idx_label,'label']
            labels = (labels.values == 'Good Outcome').astype(np.int32)

            sample_inds = np.array(d.loc[idx_valid_time].loc[idx_label,'id'].values)
            survival_time = np.array(d.loc[idx_valid_time].loc[idx_label,'time'].values)
            
        else:
            idx_valid_time = d['time'] > 0
            labels = d.loc[idx_valid_time,'dead']
            labels = labels.values.astype(np.int32)
            sample_inds = np.array(d.loc[idx_valid_time,'id'].values)

        #filtering out the unexisting fcs files
        data_files = os.listdir(args.raw_data_dir)
        existing_sample_inds = np.array([x+'.fcs' in data_files for x in sample_inds])    
        arr = [os.path.join(args.raw_data_dir,x+'.fcs') for x in sample_inds[existing_sample_inds]]
        labels = labels[existing_sample_inds]
        print(len(labels),'samples are preserved')
        if args.HIVNH_stratification:
            print(labels.sum(),'samples are judged as good outcomes')
        tubes = dict()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fio = flowio.FlowData(arr[0], ignore_offset_error=True)
        tubes = dict(channels=fio.channels)
        #we use channels selected by cellcnn paper
        CellCNN_selected = (4,6,7,8,9,10,12,14,15,16)
        CellCNN_selected_channels = np.array([3,5,6,7,8,9,11,13,14,15])
        # CellCNN_unselected_list = ['CD3','VIVID/CD14','CD19']
        used_channels = dict()
        for channel in CellCNN_selected:
            used_channels[channel] = tubes['channels'][channel]
        tubes['loader'] = TubeLoader(arr[0], tube_channels=used_channels)
        CellCNN_list = np.array([tubes['channels'][x]['pns'] for x in tubes['channels'].keys()])
        CellCNN_selected_list = CellCNN_list[CellCNN_selected_channels].tolist()
        
        h5_dir = os.path.join(args.data_saving_dir,'h5')

        os.makedirs(h5_dir,exist_ok=True)
        # all channel info have been checked to be the same except some naming changes
        fcs_data_list = []
        print('start writing files')
        for i in tqdm(range(len(arr))):
            fcs_data,channel_info = tubes['loader'].load_fcs(arr[i])
            fcs_data = fcs_data[:,CellCNN_selected_channels]
            fcs_data = np.arcsinh(fcs_data/150)
            fcs_data_list.append(fcs_data)
            saving_path = os.path.join(h5_dir,arr[i].split('/')[-1].split('.')[0]+'.h5')
            with h5py.File(saving_path,'w') as hf:
                hf.create_dataset('cells',data=fcs_data)
                hf.create_dataset('sample_labels',data=labels[i])
                hf.create_dataset('sample_survival_time',data=survival_time[i])

        os.makedirs(args.sup_info_dir,exist_ok=True)
        fcs_data_all = np.concatenate(fcs_data_list)
        np.random.seed(42)
        idx = np.random.choice(fcs_data_all.shape[0],100000,replace=False)
        # np.save(os.path.join(args.sup_info_dir,'hivnh_sampled_cells.npy'),fcs_data_all[idx])
        mean_fcs_all = fcs_data_all.mean(axis = 0)
        std_fcs_all = fcs_data_all.std(axis = 0)
        np.save(os.path.join(args.sup_info_dir,'hivnh_mean_fcs.npy'),mean_fcs_all)
        np.save(os.path.join(args.sup_info_dir,'hivnh_std_fcs.npy'),std_fcs_all)

def AML2015_preprocessing(args):
    '''
    AML2015 preprocessing process includes
    step 1: gating AML samples with enough blasts
    step 2: collecting and normalize the cells
    step 3: generate simulated samples
    '''
    # gating based on cellcnn strategy to filtering out blasts
    if args.AML2015_gating:
        column_name_list = AML_2015_raw_AML_filtering(args)
        column_name_list = filtering_common_features(column_name_list)
        np.save(os.path.join(args.sup_info_dir,'aml_2015_common_feature.npy'),column_name_list,allow_pickle=True)
    
    if args.AML2015_recollection:
        #samples are selected to be consistent with cellcnn paper
        CN_sample_list = ['SJ10', 'SJ12', 'SJ13']
        CBF_sample_list = ['SJ01', 'SJ02', 'SJ03', 'SJ04', 'SJ05']
        Healthy_sample_list = ['H1', 'H2', 'H3', 'H4', 'H5']
        
        blasts_collection(CN_sample_list,processed_path = args.data_saving_dir, saving_path=os.path.join(args.data_saving_dir,'CN_blasts.npy'))
        blasts_collection(CBF_sample_list,processed_path = args.data_saving_dir,saving_path=os.path.join(args.data_saving_dir,'CBF_blasts.npy'))
        healthy_cells_collection(Healthy_sample_list,processed_path = args.AML2015_healthy_raw_dir,saving_path=os.path.join(args.data_saving_dir,'healthy_cells.npy'))

        CN_blasts = np.load(os.path.join(args.data_saving_dir,'CN_blasts.npy'),allow_pickle=True)
        CBF_blasts = np.load(os.path.join(args.data_saving_dir,'CBF_blasts.npy'),allow_pickle=True)
        healthy_cells = np.load(os.path.join(args.data_saving_dir,'healthy_cells.npy'),allow_pickle=True)
        concatenate_cells = np.concatenate([healthy_cells,CN_blasts,CBF_blasts],axis=0)
        concatenate_cells = concatenate_cells.astype(np.float32)
        cell_mean = concatenate_cells.mean(axis=0)
        cell_std = concatenate_cells.std(axis=0)
        healthy_cells_normalized = (healthy_cells - cell_mean) / cell_std
        CN_blasts_normalized = (CN_blasts - cell_mean) / cell_std
        CBF_blasts_normalized = (CBF_blasts - cell_mean) / cell_std
        np.save(os.path.join(args.data_saving_dir,'normalized_CN_blasts.npy'),healthy_cells_normalized,allow_pickle=True)
        np.save(os.path.join(args.data_saving_dir,'normalized_CBF_blasts.npy'),CN_blasts_normalized,allow_pickle=True)
        np.save(os.path.join(args.data_saving_dir,'normalized_healthy_cells.npy'),CBF_blasts_normalized,allow_pickle=True)

    if args.AML2015_simulation_regeneration:
        np.random.seed(args.seed)
        CN_blasts = np.load(os.path.join(args.data_saving_dir,'normalized_CN_blasts.npy'),allow_pickle=True)
        CBF_blasts = np.load(os.path.join(args.data_saving_dir,'normalized_CBF_blasts.npy'),allow_pickle=True)
        healthy_cells = np.load(os.path.join(args.data_saving_dir,'normalized_healthy_cells.npy'),allow_pickle=True)
        np.random.shuffle(healthy_cells)
        np.random.shuffle(CN_blasts)
        np.random.shuffle(CBF_blasts)

        cells = np.concatenate([healthy_cells,CN_blasts,CBF_blasts])
        selected_idx = np.random.choice(cells.shape[0],args.AML2015_subsampled_cells,replace=False)
        #pick the selected index from each subgroup again and exclude them
        healthy_used_idx = selected_idx[selected_idx<len(healthy_cells)]
        cn_used_idx = selected_idx[(selected_idx>=len(healthy_cells)) & (selected_idx<len(healthy_cells)+len(CN_blasts))] - len(healthy_cells)
        cbf_used_idx = selected_idx[selected_idx>=len(healthy_cells)+len(CN_blasts)] - len(healthy_cells) - len(CN_blasts)
        healthy_cells = np.delete(healthy_cells,healthy_used_idx,axis=0)
        CN_blasts = np.delete(CN_blasts,cn_used_idx,axis=0)
        CBF_blasts = np.delete(CBF_blasts,cbf_used_idx,axis=0)
        train_healthy_cells = healthy_cells[:int(0.25 * len(healthy_cells))]
        train_CN_blasts = CN_blasts[:int(0.25 * len(CN_blasts))]
        train_CBF_blasts = CBF_blasts[:int(0.25 * len(CBF_blasts))]
        val_healthy_cells = healthy_cells[int(0.25 * len(healthy_cells)):int(0.5 * len(healthy_cells))]
        val_CN_blasts = CN_blasts[int(0.25 * len(CN_blasts)):int(0.5 * len(CN_blasts))]
        val_CBF_blasts = CBF_blasts[int(0.25 * len(CBF_blasts)):int(0.5 * len(CBF_blasts))]
        test_healthy_cells = healthy_cells[int(0.5 * len(healthy_cells)):]
        test_CN_blasts = CN_blasts[int(0.5 * len(CN_blasts)):]
        test_CBF_blasts = CBF_blasts[int(0.5 * len(CBF_blasts)):]
        #mrd range
        mrd_range = args.AML2015_mrd_range
        #bag size
        cell_range = args.AML2015_cell_range
        #distribution params for mrd ratio generation, this part should be tuned based on the mrd ratio range
        params_dict = {'mean':-4.37,'std':1.53}
        saving_dir = os.path.join(args.data_saving_dir,'lognormal-mrd-'+str(mrd_range[0])+'-'+str(mrd_range[1])+'-maxbag-'+str(cell_range[0])+'-'+str(cell_range[1]))
        os.makedirs(saving_dir ,exist_ok=True)
        trainset_generation([train_healthy_cells,train_CN_blasts,train_CBF_blasts],128,os.path.join(saving_dir,'trainset'),
                                n_components=args.AML2015_simulated_n_components,cell_range=cell_range,mrd_ratio_range = mrd_range,mrd_distribution='lognormal',mrd_params_dict=params_dict,sample_ratio = 0.8)
        real_dataset_generation([val_healthy_cells,val_CN_blasts,val_CBF_blasts],512,os.path.join(saving_dir,'valset'),
                                cell_range=cell_range,mrd_ratio_range = mrd_range,mrd_distribution='lognormal',mrd_params_dict=params_dict,sample_ratio = 0.8)
        real_dataset_generation([test_healthy_cells,test_CN_blasts,test_CBF_blasts],512,os.path.join(saving_dir,'testset'),
                                cell_range=cell_range,mrd_ratio_range = mrd_range,mrd_distribution='lognormal',mrd_params_dict=params_dict,sample_ratio = 0.8)  

def COVID_preprocessing(args):
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
    filtered_sample_inds = [i for i in sample_inds if os.path.isfile(os.path.join(args.raw_data_dir,str(i)+'_BDC-CR1.fcs'))]
    filtered_sample_inds = [i for i in filtered_sample_inds if os.path.isfile(os.path.join(args.raw_data_dir,str(i)+'_BDC-CR2.fcs'))]
    filtered_sample_inds = [i for i in filtered_sample_inds if os.path.isfile(os.path.join(args.raw_data_dir,str(i)+'_TNK-CR1.fcs'))]
    filtered_sample_inds = [i for i in filtered_sample_inds if os.path.isfile(os.path.join(args.raw_data_dir,str(i)+'_TNK-CR2.fcs'))]
    filtered_binary_classification_labels = [i for idx,i in enumerate(binary_classification_labels) if sample_inds[idx] in filtered_sample_inds]
    filtered_severity_prediction_labels = [i for idx,i in enumerate(severity_prediction_labels) if sample_inds[idx] in filtered_sample_inds]
    filtered_all_prediction_labels = [i for idx,i in enumerate(all_prediction_labels) if sample_inds[idx] in filtered_sample_inds]

    if args.COVID_dataset_initialization:
        results = {}
        for panels in ['TNK-CR1','TNK-CR2','BDC-CR1','BDC-CR2']:
            pannel_channels = []
            for sn in filtered_sample_inds:
                current_sample = flowio.FlowData(os.path.join(args.raw_data_dir,str(sn)+'_'+panels+'.fcs'), ignore_offset_error=True)
                pannel_channels.append(current_sample.channels)
            pannel_common_channels = [pannel_channels[0][nc]['pns'] for nc in pannel_channels[0].keys() if 'pns' in pannel_channels[0][nc].keys()]
            pannel_channels_pns = []
            for channel_info in pannel_channels:
                current_sample_channel = [channel_info[nc]['pns'] for nc in channel_info.keys() if 'pns' in channel_info[nc].keys()]
                pannel_channels_pns.append(current_sample_channel)
                pannel_common_channels = [i for i in pannel_common_channels if i in current_sample_channel]
            results[panels] =  pannel_common_channels
        with open(os.path.join(args.sup_info_dir,'COVID_sample_info.json'), "w") as f:
            json.dump(results, f, indent=2)        
    else:
        TNK2_channels = ['TCR Vd1 FITC', 'CD127 BB630', 'PD1 BB660', 'CD16 BB700', 'CXCR5 BB790', 'TCR Vg9 PE',
                        'TCR Vd2 PE-CF594', 'CD161 PE-Cy5', 'HLA-DR PE-Cy55', 'CXCR3 PE-Cy7', 'CD1d:PBS57 tet APC',
                        'CD45RA Ax700', 'TIGIT APC-Cy7', 'CCR8 BUV395', 'Live Dead UV Blue', 'CCR7 BUV496', 'CD56 BUV563',
                        'CD39 BUV661', 'CD95 BUV737', 'CD4 BUV805', 'CCR9 BV421', 'CD3 BV510', 'CD8a BV570', 'CD38 BV605',
                        'CCR4 BV650', 'TCR Va7_2 BV711', 'CXCR6 BV750', 'CD27 BV786', 'QC']
        TNK1_channels = ['TCR Vd1 FITC', 'CD127 BB630', 'PD1 BB660', 'CD16 BB700', 'CXCR5 BB790', 'TCR Vg9 PE',
                          'TCR Vd2 PE-CF594', 'CD161 PE-Cy5', 'HLA-DR PE-Cy55', 'CCR1 PE-Cy7', 'CD1d:PBS57 tet APC',
                            'CD45RA Ax700', 'XCR1 APC-Fire750', 'CCR3 BUV395', 'Live Dead UV Blue', 'CCR7 BUV496',
                              'CD56 BUV563', 'CD39 BUV661', 'CD95 BUV737', 'CD4 BUV805', 'CCR2 BV421', 'CD3 BV510',
                                'CD8a BV570', 'CD38 BV605', 'CCR5 BV650', 'TCR Va7_2 BV711', 'CX3CR1 BV750',
                                  'CD27 BV786', 'QC']
        BDC1_channels = ['CADM1 FITC', 'CD141 BB630', 'CD123 BB660', 'FcER1a BB700', 'IgD BB790', 'IFNAR2 PE',
                          'CD88 PE-Dazzle594', 'CD3 PE-Cy5', 'CD5 PE-Cy55', 'CCR1 PE-Cy7', 'CD11c APC',
                            'CD27 APC-R700', 'XCR1 APC-Fire750', 'CCR3 BUV395', 'Live Dead UV Blue', 'CD40 BUV496',
                            'CD56 BUV563', 'CD21 BUV661', 'CD163 BUV737', 'CD20 BUV805', 'CCR2 BV421', 'CD14 BV510',
                            'CD16 BV570', 'CD38 BV605', 'CCR5 BV650', 'CD86 BV711', 'CX3CR1 BV750', 'HLA-DR BV786',
                            'QC']
        BDC2_channels = ['CADM1 FITC', 'CD141 BB630', 'CD123 BB660', 'FcER1a BB700', 'IgD BB790', 'IFNAR2 PE',
                          'CD88 PE-Dazzle594', 'CD3 PE-Cy5', 'CD5 PE-Cy55', 'CXCR3 PE-Cy7', 'CD11c APC',
                            'CD27 APC-R700', 'CD19 APC-H7', 'CCR8 BUV395', 'Live Dead UV Blue', 'CD40 BUV496',
                              'CD56 BUV563', 'CD21 BUV661', 'CD163 BUV737', 'CD20 BUV805', 'CCR9 BV421', 'CD14 BV510',
                                'CD16 BV570', 'CD38 BV605', 'CCR4 BV650', 'CD86 BV711', 'CXCR6 BV750', 'HLA-DR BV786',
                                  'QC']
        all_panels = {'BDC-CR1':BDC1_channels,'BDC-CR2':BDC2_channels,'TNK-CR1':TNK1_channels,'TNK-CR2':TNK2_channels}
        tubes_dict = dict()
        for tn in ['BDC-CR1','BDC-CR2','TNK-CR1','TNK-CR2']:
            tubes = dict()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fio = flowio.FlowData(os.path.join(args.raw_data_dir,str(filtered_sample_inds[0])+'_'+tn+'.fcs'), ignore_offset_error=True)
            tubes = dict(channels=fio.channels)
            used_channels = dict()
            for channel in all_panels[tn]:
                found_key = [k for k in tubes['channels'].keys() if ('pns' in tubes['channels'][k].keys())]
                found_key = [k for k in found_key if ( tubes['channels'][k]['pns'] == channel)][0]
                used_channels[found_key] = tubes['channels'][found_key]    
            tubes['loader'] = TubeLoader(os.path.join(args.raw_data_dir,str(filtered_sample_inds[0])+'_'+tn+'.fcs'), tube_channels=used_channels)
            tubes_dict[tn] = used_channels
    np.random.seed(42)

    if args.COVID_compute_stat:
        for tn in tubes_dict.keys():
            fcs_data_list = []
            selected_channel_idx = list(tubes_dict[tn].keys())
            selected_channel_idx = np.array([int(scn)-1 for scn in selected_channel_idx])
            for sn in np.random.choice(filtered_sample_inds,20,replace=True):
                fcs_data,channel_info = tubes['loader'].load_fcs(os.path.join(args.raw_data_dir,str(sn)+'_'+tn+'.fcs'))
                fcs_data_list.append(fcs_data[:,selected_channel_idx])
            fcs_data_list = np.concatenate(fcs_data_list,axis=0)
            current_panel_mean = fcs_data_list.mean(0)
            current_panel_std = fcs_data_list.std(0)
            np.save(os.path.join(args.sup_info_dir,tn+'_mean.npy'),current_panel_mean)
            np.save(os.path.join(args.sup_info_dir,tn+'_std.npy'),current_panel_std)
            
    if args.COVID_h5_generation:
        os.makedirs(args.data_saving_dir,exist_ok=True)
        for sind,sn in enumerate(filtered_sample_inds):
            print(f"processing {sn}")
            current_sample_all_panels = {}
            for tn in tubes_dict.keys():
                selected_channel_idx = list(tubes_dict[tn].keys())
                selected_channel_idx = np.array([int(scn)-1 for scn in selected_channel_idx])
                current_panel_mean = np.load(os.path.join(args.sup_info_dir,tn+'_mean.npy'))
                current_panel_std = np.load(os.path.join(args.sup_info_dir,tn+'_std.npy'))
                fcs_data,channel_info = tubes['loader'].load_fcs(os.path.join(args.raw_data_dir,str(sn)+'_'+tn+'.fcs'))
                current_sample = fcs_data[:,selected_channel_idx]
                current_sample = (current_sample - current_panel_mean) / current_panel_std
                current_sample_all_panels[tn] = current_sample
            saving_path = os.path.join(args.data_saving_dir,str(sn)+'.h5')
            with h5py.File(saving_path,'w') as hf:
                for tn in tubes_dict.keys():
                    hf.create_dataset(tn,data=current_sample_all_panels[tn])
                hf.create_dataset('binary_classification_labels',data=filtered_binary_classification_labels[sind])
                hf.create_dataset('severity_prediction_labels',data=filtered_severity_prediction_labels[sind])
                hf.create_dataset('all_prediction_labels',data=filtered_all_prediction_labels[sind])

if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset in ['CRCLC']:
        CRCLC_preprocessing(args)
    if args.dataset in ['hivnh']:
        HIVNH_preprocessing(args)
    if args.dataset in ['aml_2015']:
        AML2015_preprocessing(args)
    if args.dataset in ['COVID']:
        COVID_preprocessing(args)