import pickle
from flowmil_utils.visualization import *
import numpy as np
# from umap import UMAP
import umap.umap_ as UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import sklearn.metrics as metrics
# import scikitplot as skplt
import seaborn as sns
import scipy.stats as stats
import tensorflow as tf
import os
import flowutils
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import joblib
import argparse
import pandas as pd

np.random.seed(42)
parser = argparse.ArgumentParser(description='Visualization after using post-training to extract features')
parser.add_argument('--dataset',type = str, default = 'CRCLC', help='dataset for visualization')
parser.add_argument('--loading_dir',type = str, default = None, help='loading dir for used features')
parser.add_argument('--saving_dir',type = str, default = None, help='saving dir for results')
parser.add_argument('--n_sample_cells',type = int, default = 30000, help='number of sampled cells per phenotype')
parser.add_argument('--picked_split',type = int, default = None,nargs='+', help='picked splits')
parser.add_argument('--COVID_selected_panel',type = int, default = 1, help='selected single panel for COVID dataset')
parser.add_argument('--selected_channels',type = str, default = None,nargs='+', help=' selected channels for plotting')
parser.add_argument('--top_k_cells',type = int, default = 3000, help='top_k selected cells')
parser.add_argument('--umap_attention_plotting', action='store_true', default=False, help='weighted attention UMAP plotting')
parser.add_argument('--umap_channel_plotting', action='store_true', default=False, help='weighted biomarker level UMAP plotting')
parser.add_argument('--umap_phenotype_plotting', action='store_true', default=False, help='phenotype UMAP plotting')
parser.add_argument('--cross_split_plotting', action='store_true', default=False, help='cross split figure plotting')
parser.add_argument('--sample_specific_plotting', action='store_true', default=False, help='sample specific figure plotting')
parser.add_argument('--biomarker_reaction_plotting', action='store_true', default=False, help='biomarker reaction plotting')
parser.add_argument('--channelwise_att_clean', action='store_true', default=False, help='channelwise scatter plotting')
parser.add_argument('--attention_channel_combination', action='store_true', default=False, help='biomarker reaction level plotting')
parser.add_argument('--umap_colorbar_plotting', action='store_true', default=False, help='whether or not plotting the colorbar for UMAP figures')
 
color_list = ["#0072B2",  # Blue
                "#E69F00",  # Orange
                "#56B4E9",  # Light Blue
                "#D55E00",  # Vermillion
                "#CC79A7",  # Reddish Purple
                "#009E73",  # Bluish Green
                "#999999",  # Gray
                "#F0E442",  # Yellow
                "#004488",  # Navy Blue
                "#FF69B4",   # Pink
                "#000000",  # Black
                ]

def visualization_process():
    args = parser.parse_args()
    print(args)
    channel_list = channel_info_loading(args.dataset)
    if args.dataset == 'COVID':
        channel_list = channel_list[args.COVID_selected_panel]
    phenotype_list = phenotype_info_loading(args.dataset)
    args.saving_dir = os.path.join(args.saving_dir,args.dataset)
    if args.dataset in ['aml_2015']:
        sampled_inst_feat, sampled_inst_att, sampled_inst_label = simplest_loading(args.loading_dir)
    elif args.dataset in ['hivnh','COVID','CRCLC']:
        sampled_inst_feat, sampled_inst_att,bag_label, sampled_inst_label, bag_specific_feature, bag_specific_att, sample_names, inst_bag_source, split_size, train_val_test_ind = sample_feature_att_gathering_across_sets(loading_dir = args.loading_dir,dataset = args.dataset, n_sample= args.n_sample_cells,current_split=0,prefix_list= ["train", "valid", "test"])
    
    os.makedirs(args.saving_dir,exist_ok=True)
    if args.umap_attention_plotting + args.umap_channel_plotting + args.umap_phenotype_plotting + args.biomarker_reaction_plotting:
        os.makedirs(os.path.join(args.saving_dir,'umap_sampled_cells'),exist_ok=True)
        transformed_feature, umap_model = umap_fitting(sampled_inst_feat)
        range_x = [np.min(transformed_feature[:,0]),np.max(transformed_feature[:,0])]
        range_y = [np.min(transformed_feature[:,1]),np.max(transformed_feature[:,1])]
        bin_range = [range_x,range_y]
    
    if args.cross_split_plotting:
        if args.picked_split is None:
            args.picked_split = range(5)
        if isinstance(args.picked_split,int):
            args.picked_split = [args.picked_split]
        if args.umap_attention_plotting:
            for i in args.picked_split:
                sampled_inst_feat, sampled_inst_att,_, _,_, _,_, _, _, _ = sample_feature_att_gathering_across_sets(loading_dir = args.loading_dir,dataset = args.dataset, n_sample= args.n_sample_cells,current_split=i,prefix_list= ["train", "valid", "test"])
                transformed_feature = umap_model.transform(sampled_inst_feat)
                plot_2d_density_wrapper_nhead(transformed_feature,sampled_inst_att,os.path.join(args.saving_dir,'umap_sampled_cells','split_'+str(i)+'inst_att_'),cmap = 'gist_earth',channel_names = phenotype_list,bin_range = bin_range,colorbar_plotting = args.umap_colorbar_plotting)
                print('split',i,'done')
        if args.umap_channel_plotting:
            if args.selected_channels is not None:
                assert all(elem in channel_list for elem in args.selected_channels)
                selected_channels = args.selected_channels
            else:
                print('all channels used')
                selected_channels = channel_list
            selected_ind = np.array([np.where(channel_list == x)[0][0] for x in selected_channels])
            transformed_feature = umap_model.transform(sampled_inst_feat)
            rescale_inst_feat = sampled_inst_feat - np.min(sampled_inst_feat,axis=0)
            plot_2d_density_wrapper_nhead(transformed_feature,rescale_inst_feat[:,selected_ind],os.path.join(args.saving_dir,'umap_sampled_cells','channel_'),cmap = 'magma',channel_names = channel_list[selected_ind],bin_range = bin_range,colorbar_plotting = args.umap_colorbar_plotting)
        if args.umap_phenotype_plotting:
            transformed_feature = umap_model.transform(sampled_inst_feat)
            plot_2d_density_wrapper_nhead(transformed_feature,np.eye(len(phenotype_list))[sampled_inst_label.astype(int)],os.path.join(args.saving_dir,'umap_sampled_cells','inst_label_'),cmap = 'viridis',channel_names = phenotype_list,bin_range = bin_range,colorbar_plotting = args.umap_colorbar_plotting)
        if args.biomarker_reaction_plotting:
            os.makedirs(os.path.join(args.saving_dir,'biomarker_reaction'),exist_ok=True)
            if args.selected_channels is not None:
                assert all(elem in channel_list for elem in args.selected_channels)
                selected_channels = args.selected_channels
            else:
                print('all channels used')
                selected_channels = channel_list
            if args.picked_split is None:
                args.picked_split = 0
            else:
                args.picked_split = args.picked_split[0]
            sampled_inst_feat, sampled_inst_att, bag_label, sampled_inst_label, bag_specific_feature, bag_specific_att, sample_names, inst_bag_source, split_size, train_val_test_ind = sample_feature_att_gathering_across_sets(loading_dir = args.loading_dir,dataset = args.dataset, n_sample= args.n_sample_cells,current_split=args.picked_split,prefix_list= ["train", "valid", "test"])
            selected_ind = np.array([np.where(channel_list == x)[0][0] for x in selected_channels])
            transformed_feature = umap_model.transform(sampled_inst_feat)
            rescale_inst_feat = sampled_inst_feat - np.min(sampled_inst_feat,axis=0)
            for ind, lbl in enumerate(np.unique(sampled_inst_label)):
                mask = (sampled_inst_label == lbl)
                transformed_lbl = transformed_feature[mask]
                rescaled_lbl = rescale_inst_feat[mask][:, selected_ind]
                plot_2d_density_wrapper_nhead(transformed_lbl,rescaled_lbl,os.path.join(args.saving_dir,'biomarker_reaction',phenotype_list[ind]+'_channel_'),cmap='plasma',channel_names=[channel_list[k] for k in selected_ind],bin_range=bin_range,
                colorbar_plotting = args.umap_colorbar_plotting)
        if args.channelwise_att_clean:
            if args.selected_channels is not None:
                assert all(elem in channel_list for elem in args.selected_channels)
                selected_channels = args.selected_channels
            else:
                print('all channels used')
                selected_channels = channel_list
            selected_ind = np.array([np.where(channel_list == x)[0][0] for x in selected_channels])
            os.makedirs(os.path.join(args.saving_dir,'channelwise_att_clean'),exist_ok=True)
            outdir = os.path.join(args.saving_dir, 'channelwise_att_clean')
            for split in args.picked_split:
                sampled_inst_feat, sampled_inst_att, bag_label, sampled_inst_label, bag_specific_feature, bag_specific_att, sample_names, inst_bag_source, split_size, train_val_test_ind = sample_feature_att_gathering_across_sets(loading_dir = args.loading_dir,dataset = args.dataset, n_sample= args.n_sample_cells,current_split=split,prefix_list= ["train", "valid", "test"])

                rescale_att = (sampled_inst_att - np.min(sampled_inst_att,axis=0))/(np.percentile(sampled_inst_att,99,axis=0)-np.min(sampled_inst_att,axis=0)) 
                rescale_att = np.clip(rescale_att,0,0.75)

                unique_labels = np.unique(sampled_inst_label)
                n_heads = sampled_inst_att.shape[1] if sampled_inst_att.ndim == 2 else 1

                selected_inst = {
                    lbl: np.argsort(
                        sampled_inst_att[sampled_inst_label == lbl, min(idx, n_heads - 1)]
                        if sampled_inst_att.ndim == 2 else sampled_inst_att[sampled_inst_label == lbl]
                    )[-min(args.top_k_cells, (sampled_inst_label == lbl).sum()):]
                    for idx, lbl in enumerate(unique_labels)}
                for i in range(len(selected_channels)):
                    for j in range(i + 1, len(selected_channels)):
                        xi, yj = selected_ind[i], selected_ind[j]
                        xmin, xmax = np.min(sampled_inst_feat[:, xi]), np.max(sampled_inst_feat[:, xi])
                        ymin, ymax = np.min(sampled_inst_feat[:, yj]), np.max(sampled_inst_feat[:, yj])

                        # -------- AFTER CLEAN: only top-K per class --------
                        for cidx, lbl in enumerate(unique_labels):
                            mask = (sampled_inst_label == lbl)
                            if not mask.any():
                                continue
                            abs_idx = np.flatnonzero(mask)[selected_inst[lbl]]  # map within-class idxs to absolute idxs
                            plt.scatter(
                                sampled_inst_feat[abs_idx, xi],
                                sampled_inst_feat[abs_idx, yj],
                                color=color_list[cidx % len(color_list)],
                                alpha=0.25, s=1
                            )
                        leg = plt.legend(phenotype_list)
                        for lh in leg.legendHandles:
                            lh.set_alpha(1)                        
                        plt.xlabel(channel_list[xi]); plt.ylabel(channel_list[yj])
                        plt.xlim([xmin, xmax]); plt.ylim([ymin, ymax])
                        plt.savefig(os.path.join(
                            outdir, f'after_clean_split_{split}_att_{channel_list[xi]}_{channel_list[yj]}.png'
                        ), dpi=300)
                        plt.close()

                        for cidx, lbl in enumerate(unique_labels):
                            mask = (sampled_inst_label == lbl)
                            if not mask.any():
                                continue
                            plt.scatter(
                                sampled_inst_feat[mask, xi],
                                sampled_inst_feat[mask, yj],
                                color=color_list[cidx % len(color_list)],
                                alpha=0.25, s=1
                            )
                        leg = plt.legend(phenotype_list)
                        for lh in leg.legendHandles:
                            lh.set_alpha(1)
                        plt.xlabel(channel_list[xi]); plt.ylabel(channel_list[yj])
                        plt.xlim([xmin, xmax]); plt.ylim([ymin, ymax])
                        plt.savefig(os.path.join(
                            outdir, f'before_clean_split_{split}_att_{channel_list[xi]}_{channel_list[yj]}.png'
                        ), dpi=300)
                        plt.close()
        if args.attention_channel_combination:

            sampled_inst_feat, sampled_inst_att,bag_label, sampled_inst_label, bag_specific_feature, bag_specific_att, sample_names, inst_bag_source, split_size, train_val_test_ind = sample_feature_att_gathering_across_sets(loading_dir = args.loading_dir,dataset = args.dataset, n_sample= args.n_sample_cells,current_split=0,prefix_list= ["train", "valid", "test"])
            
            n_heads = sampled_inst_att.shape[1] if sampled_inst_att.ndim == 2 else 1
            blocks, labels = [], []
            for ci, name in enumerate(phenotype_list):
                m = (sampled_inst_label == ci)
                if not m.any(): continue
                h = min(ci, n_heads - 1)
                att = (sampled_inst_att[m, h] if sampled_inst_att.ndim == 2 else sampled_inst_att[m])
                k = min(args.top_k_cells, att.size)
                idx = np.flatnonzero(m)[np.argsort(att)[-k:]]
                blocks.append(sampled_inst_feat[idx]); labels += [name]*k
            if not blocks: raise ValueError("No instances found for any classes in phenotype_list.")

            df = pd.DataFrame(np.vstack(blocks), columns=channel_list)
            df["Phenotype"] = labels
            dfm = df.melt(id_vars="Phenotype", var_name="Biomarker", value_name="Value")
            plt.figure(figsize=(16,6))
            sns.boxplot(x="Biomarker", y="Value", hue="Phenotype", data=dfm, showfliers=False, hue_order=phenotype_list)
            plt.xticks(rotation=90)
            ax = plt.gca(); ax.set_ylim(bottom=-1.5); ax.set_ylabel("Relative Biomarker Values", fontsize=20)
            leg = ax.legend(title="Phenotype", loc="best"); [h.set_alpha(1) for h in leg.legendHandles]
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            os.makedirs(os.path.join(args.saving_dir,'biomarker_reaction_on_high_att'), exist_ok=True)
            plt.savefig(os.path.join(args.saving_dir,'biomarker_reaction_on_high_att','M1.png'), dpi=300)
            plt.close()

    if args.sample_specific_plotting:
        os.makedirs(os.path.join(args.saving_dir,'umap_sample'),exist_ok=True)
        if args.picked_split is None:
            args.picked_split = 0
        assert len(args.picked_split) == 1

        if args.umap_attention_plotting:
            sampled_inst_feat, sampled_inst_att,bag_label, sampled_inst_label, bag_specific_feature, bag_specific_att, sample_names, inst_bag_source, split_size, train_val_test_ind = sample_feature_att_gathering_across_sets(loading_dir = args.loading_dir,dataset = args.dataset, n_sample= args.n_sample_cells,current_split=0,prefix_list= ["train", "valid", "test"])
            selected_sample_ind = [i for i in range(len(sample_names)) if len(bag_specific_att[i]) > 5000]
            for i in selected_sample_ind:
                current_sample_name = sample_names[i]
                current_sample_name = sample_name_preprocessing(args.dataset,current_sample_name )
                os.makedirs(os.path.join(args.saving_dir,'umap_sample',current_sample_name),exist_ok=True)
                current_sample_feature = bag_specific_feature[i]
                current_sample_att = bag_specific_att[i]
                current_sample_att = current_sample_att.numpy()
                current_sample_transformed_feature = umap_model.transform(current_sample_feature)
                current_sample_att[-1,:] = 1e-2
                plot_2d_density_wrapper_nhead(current_sample_transformed_feature,current_sample_att,os.path.join(args.saving_dir,'umap_sample',current_sample_name,'split_'+str(args.picked_split)+'_UMAP_att_'),cmap = 'gist_earth',channel_names = phenotype_list,bin_range = bin_range,colorbar_plotting = args.umap_colorbar_plotting)
                print(sample_names[i],' done')
        
if __name__ == '__main__':
    visualization_process()