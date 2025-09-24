import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import tensorflow as tf
import os
import pickle
import flowutils
import umap.umap_ as UMAP

# weighted histogram plotting function modified from UMAP paper 
def plot_2d_density(X, Y=None, bins=2000, n_pad=0, w=None, ax=None, gaussian_sigma=0.5,
                    cmap=plt.get_cmap('viridis'), vlim=np.array([0.001, 0.98]), circle_type='bg',
                    box_off=True, return_matrix=False,saving_path = None,dpi = 300,bin_range=None,
                    fix_vlim=None,colorbar_plotting=False):
    
    if Y is not None:
        if w is not None:
            if bin_range is None:
                b, _, _ = np.histogram2d(X, Y, bins=bins)
            else:
                b, _, _ = np.histogram2d(X, Y, bins=bins, range=bin_range)
            b = ndi.gaussian_filter(b.T, sigma=gaussian_sigma)

            if bin_range is None:
                s, _, _ = np.histogram2d(X, Y, bins=bins, weights=w)
            else:
                s, _, _ = np.histogram2d(X, Y, bins=bins, weights=w, range=bin_range)
            s = ndi.gaussian_filter(s.T, sigma=gaussian_sigma)

            d = np.zeros_like(b)
            d[b > 0] = s[b > 0] / b[b > 0]
            d = ndi.gaussian_filter(d, sigma=gaussian_sigma)
            # d = s
            # d = ndi.gaussian_filter(d, sigma=gaussian_sigma)
        else:
            if range is None:
                d, _, _ = np.histogram2d(X, Y, bins=bins)
            else:
                d, _, _ = np.histogram2d(X, Y, bins=bins, range=bin_range)
            d /= np.sum(d)
            d = ndi.gaussian_filter(d.T, sigma=gaussian_sigma)
    else:
        d = X

    if return_matrix:
        return d
    else:
        if np.isscalar(vlim):
            vlim = np.array([0, np.quantile(d[d > 0].flatten(), vlim)])
        else:
            if fix_vlim is not None:
                vlim = np.array([0, fix_vlim])
            else:
                if np.all((vlim < 1) & (vlim > 0)):
                    vlim = np.quantile(d[d > 0].flatten(), vlim)

        # if ax is None:
        #     _, ax = plt.subplots()

        if np.isscalar(bins):
            n_bins = bins
        else:
            n_bins = len(bins[0]) - 1
        fig=plt.figure()
        ax=fig.add_subplot(111)
        if circle_type == 'bg':
            c = np.meshgrid(np.arange(2 * n_pad + n_bins), np.arange(2 * n_pad + n_bins))
            c = np.sqrt(((c[0] - ((2 * n_pad + n_bins) / 2)) ** 2) + ((c[1] - ((2 * n_pad + n_bins) / 2)) ** 2)) < (0.95 * ((2 * n_pad + n_bins) / 2))
            pc=ax.pcolormesh(np.pad(d, [n_pad, n_pad]) + c, vmin=1, vmax=1 + vlim[1], cmap=cmap, shading='gouraud', alpha=1)
            #plot colorbar
            if colorbar_plotting:
                cbar = plt.colorbar(pc)
        elif circle_type == 'arch':
            c = (n_bins / 2)
            ax.add_artist(plt.Circle((c + n_pad, c + n_pad), 0.95 * (c + n_pad), color='black', fill=False))
            ax.pcolormesh(np.pad(d, [n_pad, n_pad]), vmin=-vlim[1], vmax=vlim[1], cmap=cmap, shading='gouraud', alpha=1)
        else:
            ax.pcolormesh(np.pad(d, [n_pad, n_pad]), vmin=0, vmax=vlim[1], cmap=cmap, shading='gouraud', alpha=1)

        if box_off is True:
            [ax.spines[sp].set_visible(False) for sp in ax.spines]
            ax.set(xticks=[], yticks=[])
        plt.savefig(saving_path,dpi = dpi)
        plt.close()

def plot_2d_density_wrapper(features,attention,saving_path):
    """
    wrapper for plot_2d_density
    features: [n_sample,n_cell,2]
    attention: [n_sample,n_cell]
    
    """
    if isinstance(features,tf.RaggedTensor):
        features = features.flat_values.numpy()
        attention = attention.flat_values.numpy()
    elif isinstance(features,tf.Tensor):
        features = features.numpy()
        attention = attention.numpy()
    plot_2d_density(X=features[:,0],Y=features[:,1],w=attention[:,0],saving_path = saving_path)

# wrapper for multi-channel plotting
def plot_2d_density_wrapper_nhead(features,attention,saving_path,mode=1,cmap='viridis',dpi = 300,channel_names = None,bin_range = None,fix_vlim = None,colorbar_plotting=False,bins=1000,hist_collection = False,gaussian_sigma=0.5):
    """
    wrapper for plot_2d_density
    features: [n_sample,n_cell,2]
    attention: [n_sample,n_head,n_cell]
    modes: 1 for weighted attention plotting where attention is used for weighted average,
    2 use attention to sample subset of cells where attention is treated as a 0/1 matrix.

    """
    
    if isinstance(features,tf.RaggedTensor):
        features = features.flat_values.numpy()
        attention = attention.flat_values.numpy()
    elif isinstance(features,tf.Tensor):
        features = features.numpy()
        attention = attention.numpy()
    if len(attention.shape)==1:
        attention = attention[:,np.newaxis]
    if channel_names is None:
        channel_names = [str(i) for i in attention.shape[1]]
    hist_results = []
    for i in range(attention.shape[1]):

        if fix_vlim is not None:
            current_vlim = fix_vlim[i]
        else:
            current_vlim = None
        if mode==1:
            hist_result = plot_2d_density(X=features[:,0],Y=features[:,1],w=attention[:,i],saving_path = saving_path+channel_names[i]+'.png',bins=bins,gaussian_sigma=gaussian_sigma,
                            cmap=cmap,dpi = dpi,bin_range = bin_range,fix_vlim = current_vlim,colorbar_plotting=colorbar_plotting,return_matrix=hist_collection)
            hist_results.append(hist_result)
        if mode==2:
            sampled_features = features[attention[:,i] == 1]
            if features.shape[0]>0:
                hist_result = plot_2d_density(X=sampled_features[:,0],Y=sampled_features[:,1],saving_path = saving_path+channel_names[i]+'.png',bins=bins,gaussian_sigma=gaussian_sigma,
                            cmap=cmap,dpi=dpi,bin_range = bin_range,fix_vlim = current_vlim,colorbar_plotting=colorbar_plotting,return_matrix=hist_collection)
                hist_results.append(hist_result)
        
    if hist_collection:
        return hist_results
        
def plot_2d_density_with_clustering_model(features,saving_path,clustering_mean, clustering_std,cmap='viridis',dpi = 300,colorbar_plotting=False):
    """
    wrapper for plot_2d_density
    features: [n_sample,n_cell,2]
    attention: [n_sample,n_head,n_cell]
    
    """
    
    if isinstance(features,tf.RaggedTensor):
        features = features.flat_values.numpy()
    elif isinstance(features,tf.Tensor):
        features = features.numpy()

    # density_map = plot_2d_density(X=features[:,0],Y=features[:,1],w=None,bins=1000,gaussian_sigma=0.5,cmap=cmap,dpi = dpi,colorbar_plotting=colorbar_plotting,
    #                 return_matrix=True)
    fig,ax = plt.subplots()
    plt.scatter(features[:,0],features[:,1],alpha = 0.5,s= 1,color='y')
    plt.scatter(clustering_mean[:,0],clustering_mean[:,1],alpha = 1,s= 5,color='b')
    for nc in range(len(clustering_mean)):
        circle = plt.Circle(clustering_mean[nc], clustering_std[nc]/5, edgecolor='blue', facecolor='none', linewidth=1)
        ax.add_patch(circle)
    plt.axis('off')
    plt.savefig(saving_path)
    fig.clf()
 
def plot_mrd_prediction_curve(prediction,ground_truth,saving_path):
    """
    plotting function for mrd prediction curve
    prediction: [n_sample]
    ground_truth: [n_sample]
    """
    plt.figure()
    plt.scatter(prediction,ground_truth,alpha=0.1)
    plt.xlabel('prediction')
    plt.ylabel('ground_truth')
    plt.title('mrd prediction curve')
    plt.savefig(saving_path)
    
def plot_mrd_prediction_curve_wrapper(dataset,model,saving_path):
    """
    wrapper for plot_mrd_prediction_curve
    dataset: tf.data.Dataset
    model: tf.keras.Model
    """
    predictions = []
    ground_truths = []
    for features,labels in dataset:
        predictions.append(model.predict(features))
        ground_truths.append(labels)
    predictions = [x[:] for x in predictions]
    ground_truths = [x.numpy().squeeze()  for x in ground_truths]
    prediction = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truths)
    plot_mrd_prediction_curve(prediction,ground_truth,saving_path)

def plot_with_cell_label(features, cell_labels,attention, saving_path):
    features = features.flat_values.numpy()
    attention = attention.flat_values.numpy()
    cell_labels = cell_labels.flat_values.numpy()
    
    #randomly select 500 cells with label 0
    idx_0 = np.where(cell_labels==0)[0]
    idx_0 = np.random.choice(idx_0,500,replace=False)
    features_0 = features[idx_0,:]
    attention_0 = attention[idx_0,:]
    #randomly select 500 cells with label 1
    idx_1 = np.where(cell_labels==1)[0]
    idx_1 = np.random.choice(idx_1,500,replace=False)
    features_1 = features[idx_1,:]
    attention_1 = attention[idx_1,:]
    #plot
    plt.figure()
    plt.scatter(features_0[:,0],features_0[:,1],c=attention_0[:,0],alpha=0.25,marker='o')
    plt.scatter(features_1[:,0],features_1[:,1],c=attention_1[:,0],alpha=0.25,marker='^')
    plt.colorbar()
    plt.axis('off')
    plt.title('attention on cells learned by network')
    plt.legend(['normal','MRD'])
    plt.savefig(saving_path)
    plt.clf()

def scattering_plot(features, attentions, labels, saving_path,mode='overlapping',sampling=True):
    """
    features: [n_sample,n_cell,2]
    attentions: [n_sample,n_cell,2]
    labels: [n_sample,n_class]
    """
    n_class = labels.shape[1]
    color_list = [[0.75,0,0],[0,0.75,0]]
    if mode == 'overlapping':
        figure = plt.subplots(1,2,figsize=(20,10))
        plt.subplot(1,2,1)
        for i in range(n_class):
            current_class_features = tf.ragged.boolean_mask(features,labels[:,i]==1).flat_values.numpy()
            if sampling:
                idx = np.random.choice(current_class_features.shape[0],10000,replace=False)
                current_class_features = current_class_features[idx,:]
            plt.scatter(current_class_features[:,0],current_class_features[:,1],color=color_list[i],alpha=0.1)
        plt.title('cell distribution')
        plt.subplot(1,2,2)
        for i in range(n_class):
            current_class_attentions = tf.ragged.boolean_mask(attentions,labels[:,i]==1).flat_values.numpy()
            if sampling:
                idx = np.random.choice(current_class_attentions.shape[0],10000,replace=False)
                current_class_attentions = current_class_attentions[idx,:]
            plt.scatter(current_class_attentions[:,0],current_class_attentions[:,1],color=color_list[i],alpha=0.1)
        plt.title('attention distribution')
        plt.savefig(saving_path)
        plt.clf()
    elif mode == 'separated':
        figure = plt.subplots(n_class,2,figsize=(20,10*n_class))
        for i in range(n_class):
            plt.subplot(n_class,2,2*i+1)
            current_class_features = tf.ragged.boolean_mask(features,labels[:,i]==1).flat_values.numpy()
            if sampling:
                idx = np.random.choice(current_class_features.shape[0],10000,replace=False)
                current_class_features = current_class_features[idx,:]
            plt.scatter(current_class_features[:,0],current_class_features[:,1],color=color_list[i],alpha=0.1)
            plt.title('cell distribution for class '+str(i))
            plt.subplot(n_class,2,2*i+2)
            current_class_attentions = tf.ragged.boolean_mask(attentions,labels[:,i]==1).flat_values.numpy()
            if sampling:
                idx = np.random.choice(current_class_attentions.shape[0],10000,replace=False)
                current_class_attentions = current_class_attentions[idx,:]
            plt.scatter(current_class_attentions[:,0],current_class_attentions[:,1],color=color_list[i],alpha=0.1)
            plt.title('attention distribution for class '+str(i))
        plt.savefig(saving_path)
        plt.clf()
    elif mode == 'att_weighted':
        figure = plt.subplots(1,n_class,figsize=(20,10))
        for i in range(n_class):
            plt.subplot(1,n_class,i+1)
            current_class_features = tf.ragged.boolean_mask(features,labels[:,i]==1).flat_values.numpy()
            current_class_attentions = tf.ragged.boolean_mask(attentions,labels[:,i]==1).flat_values.numpy()
            if sampling:
                idx = np.random.choice(current_class_features.shape[0],10000,replace=False)
                current_class_features = current_class_features[idx,:]
                current_class_attentions = current_class_attentions[idx,:]
            plt.scatter(current_class_features[:,0],current_class_features[:,1],c=current_class_attentions[:,i],cmap='viridis',alpha=0.1)
            plt.colorbar()
            plt.title('cell distribution for class '+str(i))
        plt.savefig(saving_path)
        plt.clf()
        
def v3_model_visualization(features,outputs,labels,saving_path,n_clusters = 5,mode = 'each'):
    for i in range(len(labels)//4):
        current_features = features[i].numpy()
        current_similarity = outputs[-1][-1][i].numpy()
        if mode == 'each':
            figure = plt.subplots(2,3,figsize=(20,10))
            for j in range(5):
                plt.subplot(2,3,j+1)
                plt.scatter(current_features[:,0],current_features[:,1],c=current_similarity[:,j],cmap='viridis',alpha=0.1)
                plt.colorbar()
                plt.title('attention distribution for cluster '+str(j))
            plt.savefig(saving_path+str(i)+'.png')
            plt.close()
        elif mode == 'all':
            color_list = ['skyblue','orange','purple','brown','pink','gray','olive','cyan','green','red',]
            figure = plt.figure(figsize=(20,10))
            for j in range(5):
                # pick the instances with current cluster
                current_cluster  = np.where(np.argmax(current_similarity,axis=1)==j)[0]
                plt.scatter(current_features[current_cluster,0],current_features[current_cluster,1],c=color_list[j],alpha=1,label='cluster '+str(j))
            # plt.colorbar()
            plt.legend()
            plt.title('attention distribution for all clusters')
            plt.savefig(saving_path+str(i)+'.png')
            plt.close()

def gmm_model_visualization(features,outputs,labels,saving_path,mode='hard',model=None,draw_dist=True):
    features = features.flat_values.numpy()
    outputs = outputs.flat_values.numpy()
    color_list = ['skyblue','orange','purple','brown','pink','gray','olive','cyan','green','red',
                  'skyblue','orange','purple','brown','pink','gray','olive','cyan','green','red',]

    if mode == 'hard':
        selected_idx = np.random.choice(features.shape[0],10000,replace=False)
        features = features[selected_idx,:]
        outputs = outputs[selected_idx,:]
        outputs = np.argmax(outputs,axis=1)
        figure = plt.figure(figsize=(20,10))
        plt.scatter(features[:,0],features[:,1],c=outputs,alpha=0.1)
        plt.colorbar()
        plt.title('gmm model visualization')
        if draw_dist:
            centroids = model.get_layer('gaussian_radial_basis_layer').l.numpy().T
            # centroids = np.array([[0,0],[-5,0],[5,5],[5,0],[-5,5]])
            std = model.get_layer('gaussian_radial_basis_layer').s.numpy().T
            for i in range(centroids.shape[0]):
                plt.scatter(centroids[i,0],centroids[i,1],c='red',s=100)
                #draw ellipse
                theta = np.linspace(0, 2*np.pi, 100)
                x = std[i,0] * np.cos(theta) + centroids[i,0]
                y = std[i,1] * np.sin(theta) + centroids[i,1]
                plt.plot(x, y,c=color_list[i])
        plt.savefig(saving_path)
        plt.close()

def gmm_mlp_model_visualization(features,outputs,labels,saving_path,mode='hard',model=None,draw_dist=True,centroids=None,std = None):
    if isinstance(features,tf.RaggedTensor):
        features = features.flat_values.numpy()
    else:
        features = features.numpy()
    if isinstance(outputs,tf.RaggedTensor):
        outputs = outputs.flat_values.numpy()
    else:
        outputs = outputs.numpy()
    color_list = ['skyblue','orange','purple','brown','pink','gray','olive','cyan','green','red']
    color_list = color_list*10
    
    if mode == 'hard':
        selected_idx = np.random.choice(features.shape[0],10000,replace=False)
        features = features[selected_idx,:]
        outputs = outputs[selected_idx,:]
        outputs = np.argmax(outputs,axis=1)
        figure = plt.figure(figsize=(20,10))
        plt.scatter(features[:,0],features[:,1],c=outputs,alpha=0.1)
        plt.colorbar()
        plt.title('gmm model visualization')
        if draw_dist:
            for i in range(centroids.shape[0]):
                plt.scatter(centroids[i,0],centroids[i,1],c='red',s=100)
                #draw ellipse
                theta = np.linspace(0, 2*np.pi, 100)
                x = std[i,0] * np.cos(theta) + centroids[i,0]
                y = std[i,1] * np.sin(theta) + centroids[i,1]
                plt.plot(x, y,c=color_list[i])
        plt.savefig(saving_path)
        plt.close()
        
def channel_wise_visualization(features,clusters,saving_path =None,mode='concat',sampling=True):
    
    if isinstance(features,tf.RaggedTensor):
        features = features.flat_values.numpy()
        clusters = clusters.flat_values.numpy()
    elif isinstance(features,tf.Tensor):
        features = features.numpy()
        clusters = clusters.numpy()
    if mode == 'concat':
        features_concat = np.concatenate([features,clusters],axis=1)
        features_concat = (features_concat - np.min(features_concat,axis=0,keepdims=True))/(np.max(features_concat,axis=0,keepdims=True)-np.min(features_concat,axis=0,keepdims=True))
        figure = plt.figure(figsize=(20,10))
        plt.imshow(features_concat.T,aspect='auto',cmap='hot', interpolation='nearest')
        channel_names = ['feature_'+str(i) for i in range(features.shape[1])]+['cluster_'+str(i) for i in range(clusters.shape[1])]
        plt.yticks(np.arange(len(channel_names)),channel_names)
        plt.savefig(saving_path)
        plt.clf()
    if mode == 'separated':
        nf_x = int(clusters.shape[1]**(0.5))
        nf_y = int(np.ceil(clusters.shape[1]/int(clusters.shape[1]**(0.5))))
        figure = plt.subplots(nf_x,nf_y,figsize = (20*nf_y,10*nf_x))
        for i in range(clusters.shape[1]):
            plt.subplot(nf_x,nf_y,i+1)
            #plot for samples with cluster i
            current_cluster = np.argmax(clusters,axis=1)==i
            plt.imshow(features[current_cluster,:].T,aspect='auto',cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('cluster_'+str(i))
        plt.savefig(saving_path)
        plt.clf()

# channel info loading
def channel_info_loading(dataset):
    print('current version of script only works for complete channels for the datasets')
    if dataset == 'CRCLC':
        channel_list = np.array(['CCR7','CD103','CD127','CD14','CD161','CD25','CD27',
                                'CD28','CD3','CD39','CD4','CD45','CD56','CD57','CD69','CD8',
                                'HLA-DR','ICOS','PD-1','TCRgd','TIGIT','Va7-2'])
    elif dataset == 'hivnh':
        channel_list =np.array(['KI67','CD28','CD45RO','CD8','CD4','CD57','CCR5','CD27','CCR7','CD127']) 
    elif dataset == 'aml_2015':
        channel_list = np.array( ['pZap70-Syk', 'HLA-DR', 'CD7', 'DNA1', 'pAKT', 'CD34', 'CD123',
                         'BC2', 'BC6', 'CD19', 'CD11b', 'BC4', 'BC5', 'CD44',
                         'CD45', 'CD47', 'pSTAT1', 'pc-Cbl', 'CD38', 'BC3', 'pAMPK', 'CD33',
                         'cCaspase3', 'pP38', 'Viability', 'pErk1-2', 'DNA2', 'Cell_length',
                         'pCREB', 'pS6', 'BC1', 'pSTAT5', 'pRb', 'CD117', 'CD41', 'CD64', 
                         'p4EBP1', 'pPLCg2', 'CD15', 'pSTAT3', 'CD3'])
    elif dataset == 'COVID':
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
        channel_list = {1:BDC1_channels,2:BDC2_channels,3:TNK1_channels,4:TNK2_channels}
    else:
        raise NotImplementedError('other datasets not supported yet')
    return channel_list

# phenotype info loading
def phenotype_info_loading(dataset):
    dataset = dataset.upper()
    if dataset == 'CRCLC':
        phenotype_names = ['LC','CRC']
    elif dataset == 'HIVNH':
        phenotype_names = ['Poor Outcomes','Good Outcomes']
    elif dataset == 'AML_2015':
        phenotype_names = ['Healthy','CN','CBF']
    elif dataset == 'COVID':
        phenotype_names = ['Healthy','Mild','Severe']
    else:
        raise NotImplementedError('other datasets not supported yet')
    return phenotype_names

def sample_name_preprocessing(dataset,sample_name ):
    if dataset in ['CRCLC','hivnh']:
        sample_name = sample_name.split('.')[0][:-8]
    else:
        raise NotImplementedError('other datasets not supported yet')
    return sample_name
# normalization to make the data just the same as neural network input
def feature_normalization_func(data,dataset = 'CRCLC'):
    if dataset == 'CRCLC':
        fcs_mean = 0.209501
        fcs_std= 0.155944
        data =  flowutils.transforms.logicle(data, t = 16409, m = 4.5, w = 0.25,a=0,channel_indices=None)     
        data = (data - fcs_mean) / fcs_std
    elif dataset == 'HIVNH':
        fcs_mean = np.load('./preprocessing_used_files/hivnh_mean_fcs.npy')
        fcs_std= np.load('./preprocessing_used_files/hivnh_std_fcs.npy')
        data = (data - fcs_mean) / fcs_std
    elif dataset == 'COVID':
        pass
    return data


def _to_numpy_feature(x):
    """x is the 'feature' part (x[0]) of a mid_output entry."""
    if isinstance(x, tf.Tensor):
        return x.numpy()
    if isinstance(x, tf.RaggedTensor):
        return x.flat_values.numpy()
    if isinstance(x, np.ndarray):
        return x.squeeze()
    raise TypeError(f"Unsupported feature type: {type(x)}")

def _to_numpy_att(x):
    """x is the 'attention' part (x[-1]) of a mid_output entry."""
    if isinstance(x, tf.Tensor):
        return x.numpy()
    if isinstance(x, tf.RaggedTensor):
        return x.flat_values
    if isinstance(x, np.ndarray):
        return x.squeeze()
    raise TypeError(f"Unsupported attention type: {type(x)}")

def _safe_decode(bytes_or_str):
    if isinstance(bytes_or_str, (bytes, bytearray)):
        return bytes_or_str.decode("utf-8")
    if isinstance(bytes_or_str, str):
        return bytes_or_str
    raise TypeError(f"Unsupported sample name type: {type(bytes_or_str)}")

# gathering features and attentions
def sample_feature_att_gathering_across_sets(
    loading_dir: str,
    dataset: str,                      
    n_sample: int = 30000,
    current_split: int = 0,
    feature_normalization = True,
    prefix_list:list[str] = ["train", "valid", "test"]
):
    """
    Unified sampler for CRCLC / HIVNH / COVID.
    Returns:
        inst_feat, inst_att, bag_label, inst_label, bag_specific_feature, bag_specific_att,
        sample_names, inst_bag_source, split_size, train_val_test_ind
    """
    dataset = dataset.upper()
    if dataset not in {"CRCLC", "HIVNH", "COVID"}:
        raise ValueError("dataset must be one of {'CRCLC', 'HIVNH', 'COVID'}")

    inst_feature = []
    inst_att = []
    bag_label = []
    sample_names = []
    split_size = [0] 
    train_val_test_ind = []

    for current_prefix in prefix_list:
        pkl_name =  f"{current_prefix}_att_split_{current_split}.pkl"
        with open(os.path.join(loading_dir, pkl_name), "rb") as f:
            mid_outputs = pickle.load(f)
        feats = []
        for x in mid_outputs["mid_output"]:
            raw_feat = _to_numpy_feature(x[0])
            if feature_normalization:
                feature_normalization_func(raw_feat,dataset)
            feats.append(_to_numpy_feature(x[0]))

        inst_feature.extend(feats)

        raw_att = []
        for x in mid_outputs["mid_output"]:
            raw_att.append(_to_numpy_att(x[-1]))
        inst_att.extend(raw_att)

        bag_label.append([x.numpy() for x in mid_outputs["label"]])
        sample_names.extend([_safe_decode(x.numpy()) for x in mid_outputs["sample_names"]])
        split_size.append(len(mid_outputs["mid_output"]))
        train_val_test_ind.append([current_prefix] * len(mid_outputs["mid_output"]))

    bag_label = np.concatenate(bag_label, axis=0)              
    sample_names = np.array(sample_names)                      
    bag_specific_feature = inst_feature                        
    bag_specific_att = inst_att                                 
    split_size = np.array(split_size)                           
    train_val_test_ind = sum(train_val_test_ind, [])           

    uniq = np.unique(bag_label).tolist()
    uniq = sorted(uniq)
    per_class_feats = {c: [] for c in uniq}
    per_class_atts = {c: [] for c in uniq}
    per_class_bag_source = {c: [] for c in uniq}

    for i in range(len(bag_label)):
        c = int(bag_label[i])
        f_i = inst_feature[i]
        a_i = inst_att[i]
        if isinstance(a_i, tf.Tensor):
            a_i = a_i.numpy()
        per_class_feats[c].append(f_i)
        per_class_atts[c].append(a_i)
        per_class_bag_source[c].append(len(f_i) * np.ones(len(f_i)))

    for c in uniq:
        per_class_feats[c] = np.concatenate(per_class_feats[c], axis=0) if len(per_class_feats[c]) else np.empty((0,))
        per_class_atts[c]  = np.concatenate(per_class_atts[c],  axis=0) if len(per_class_atts[c])  else np.empty((0,))
        per_class_bag_source[c] = np.concatenate(per_class_bag_source[c], axis=0) if len(per_class_bag_source[c]) else np.empty((0,))

        if per_class_feats[c].shape[0] < n_sample:
            raise ValueError(
                f"Class {c} has only {per_class_feats[c].shape[0]} instances; "
                f"cannot sample {n_sample} without replacement."
            )

    sampled_feats = []
    sampled_atts = []
    sampled_labels = []
    sampled_bag_sources = []

    rng = np.random.default_rng()
    for c in uniq:
        idx = rng.choice(per_class_feats[c].shape[0], n_sample, replace=False)
        sampled_feats.append(per_class_feats[c][idx])
        sampled_atts.append(per_class_atts[c][idx])
        sampled_bag_sources.append(per_class_bag_source[c][idx])
        sampled_labels.append(np.full(n_sample, c, dtype=float))

    inst_feat = np.concatenate(sampled_feats, axis=0)
    inst_att = np.concatenate(sampled_atts, axis=0)
    inst_label = np.concatenate(sampled_labels, axis=0)
    inst_bag_source = np.concatenate(sampled_bag_sources, axis=0)

    return inst_feat, inst_att,bag_label, inst_label, bag_specific_feature, bag_specific_att, sample_names, inst_bag_source, split_size, train_val_test_ind     
    
def simplest_loading(data_dir, prefix = 'test'):
    mid_outputs = pickle.load(open(data_dir+'/'+prefix+'_att.pkl', 'rb'))
    inst_att = mid_outputs['mid_output'][0][-1].numpy()
    inst_feat = mid_outputs['mid_output'][0][0].numpy()
    inst_label = mid_outputs['inst_label'][0]
    return inst_feat, inst_att, inst_label

def umap_fitting(feature):
    umap_model = UMAP.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='euclidean', random_state=42)
    transformed_feature = umap_model.fit_transform(feature)
    return transformed_feature, umap_model