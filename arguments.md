# Arguments Description for FlowMIL

This is the description document for FlowMIL
This page includes
- [Preprocessing Arguments](#preprocessing-arguments)
- [Training Arguments](#training-arguments)
- [Evaluations and Attention Extraction](#evaluations-and-attention-extraction)
- [Visualization](#visualization)
## Preprocessing Arguments


- **`--raw_data_dir`**  
  Directory containing the raw dataset files to be loaded.

- **`--data_saving_dir`**  
  Output directory where preprocessed dataset files will be stored.

- **`--dataset`**  
  Name of the dataset to preprocess. Choices: `['aml_2015','CRCLC','hivnh','COVID']`.

- **`--used_feature`**  
  List of features to be used during preprocessing. Accepts multiple values (space-separated).

- **`--sup_info_dir`**  
  Directory to save supplementary preprocessing information (e.g., mapping or metadata). Currently is `./preprocessing_used_files`.

- **`--seed`**  
  Random seed for reproducibility.

- **`--CRCLC_channel_info`**  
  Excel file providing channel mapping information for the CRCLC raw data. Currently in `./preprocessing_used_files`.

- **`--HIVNH_demographic`**  
  CSV file containing demographic information for the HIVNH dataset. Currently in `./preprocessing_used_files`.

- **`--HIVNH_stratification`**  
  If `True`, process HIVNH as a classification task; otherwise as a survival regression task. Stratification threshold can be found in `preprocessing.HIVNH_preprocessing`.

- **`--AML2015_gating`**  
  If set, apply gating to extract blasts from original files.

- **`--AML2015_recollection`**  
  If set, recollect blasts from preprocessed files.

- **`--AML2015_simulation_regeneration`**  
  If set, regenerate simulated AML samples.

- **`--AML2015_healthy_raw_dir`**  
  Directory containing raw healthy cell samples.

- **`--AML2015_subsampled_cells`**  
  Number of cells to subsample for simulation.

- **`--AML2015_simulated_n_components`**  
  Number of Gaussian Mixture Model (GMM) kernels used in simulation.

- **`--AML2015_mrd_range`**  
  Range of Minimal Residual Disease (MRD) ratios for data generation.

- **`--AML2015_cell_range`**  
  Range of MRD cell counts for data generation.

- **`--COVID_demographic`**  
  CSV file containing demographic information for the COVID dataset. Currently in `./preprocessing_used_files`.

- **`--COVID_dataset_initialization`**  
  If set, check and initialize channel information for the COVID dataset.

- **`--COVID_compute_stat`**  
  If set, compute mean and statistical summaries of the COVID dataset.

- **`--COVID_h5_generation`**  
  If set, generate an HDF5 file for the COVID dataset.

## Training Arguments

- **`--dataset_dir`**  
  Directory containing the dataset files.

- **`--dataset`**  
  Name of the dataset to use. Choices: `['aml_2015','CRCLC','hivnh','COVID']`.

- **`--model`**  
  Model architecture to use. Choices: `['simple_mlp','mil_v0','abmil_sh','abmil_mh','gmm']`.

- **`--output_dir`**  
  Directory to save training results.

- **`--n_classes`**  
  Number of classes for classification.

- **`--n_split`**  
  Specific data split index for cross validation.

- **`--inst_mlp_setting`**  
  Instance MLP layer configuration (list of hidden units).

- **`--instance_final_act`**  
  Final activation function for the instance network.

- **`--exp_code`**  
  Experiment code to override other specified settings.

- **`--final_act`**  
  Final activation function at the sample level.

- **`--dropout`**  
  Dropout rate applied per layer.

- **`--feature_type`**  
  Type of features to use. Choices: `['cells','mstep_pi','cell_instance','instance_label','mean_inst','mean_inst_feature']`.

- **`--model_loading_dir`**  
  Directory to load a pretrained model.

- **`--validation`**  
  If set, enable train/val/test splitting.

- **`--batch_size`**  
  Mini-batch size.

- **`--task`**  
  Task type. Choices: `['regression','classification']`.

- **`--cv`**  
  Number of splits for cross-validation.

- **`--learning_rate`**  
  Learning rate for training.

- **`--phase`**  
  Training phase. Choices: `['classification','feature_extraction','evaluation','clustering']`.

- **`--instance_aug`**  
  If set, enable instance augmentation in the training process, where cells are sampled randomly each step.

- **`--inst_aug_range`**  
  Range for instance augmentation (suggested used in HIVNH and COVID training sets).

- **`--weight_decay`**  
  Weight decay (L2 regularization) factor.

- **`--n_exp`**  
  Experiment number identifier.

- **`--return_sample_info`**  
  If set, return sample information (only for sample-specific feature extraction).

- **`--pretraining`**  
  If set, enable pretraining mode.

- **`--pretraining_loading_dir`**  
  Directory of pretrained model to load for pretraining.

- **`--inst_att_trainable`**  
  If set, make the instance attention network trainable.

- **`--patience`**  
  Patience for early stopping.

- **`--max_epoch`**  
  Maximum number of training epochs.

---
### MIL-V0 Specific Arguments

- **`--mil_v0_bag_logits_aggr`**  
  Method for aggregating bag logits in the `mil_v0` model.  
  Works identically to `att_normalization` in ABMIL:  
  - `mean` → equivalent to `normalize_based_on_instance`  
  - `sum` → equivalent to `normalize_along_instance`


### ABMIL-Specific Arguments

- **`--abmil_att_encoding`**  
  Attention encoding method for ABMIL.

- **`--inst_att_setting`**  
  Instance attention MLP configuration (list of hidden units).

- **`--instance_att_act`**  
  Final activation function for instance attention.

- **`--att_normalization`**  
  Attention normalization strategy.

- **`--att_regularization_method`**  
  Regularization method for attention.

- **`--loss_weights`**  
  Loss weights for multiple targets, should be only used with `att_regularization_method` provided.

---

### AML-Specific Arguments

- **`--load_cell_label`**  
  If set, load cell-level labels.

- **`--aml_instance_encoder_loading`**  
  If set, load pretrained AML instance encoder.

- **`--aml_inst_encoder_trainable`**  
  If set, make AML instance encoder trainable.

---

### Clustering Arguments

- **`--unit`**  
  Number of kernels in Gaussian Mixture Model (GMM).

---

### Post-Training / Evaluation Arguments

- **`--result_collection`**  
  If set, enable result collection.

- **`--inst_att_extraction`**  
  If set, extract instance attention weights.

- **`--predict_value_collection`**  
  If set, collect predicted values.

---

### COVID-Specific Arguments

- **`--COVID_task`**  
  Prediction target for COVID dataset (e.g., binary classification).

- **`--selected_panels`**  
  Panels selected for COVID dataset.

- **`--ashbin`**  
  Number of extra attention heads (Ashbin attention).

- **`--l2_reg_weight`**  
  L2 regularization weight on model variables (used in `abmil_mh` experiments).

- **`--COVID_single_panel_pretrained_dir`**  
  Directory for pretrained COVID single-panel model.

- **`--COVID_demographic`**  
  Demographic information file for COVID dataset.

- **`--inst_mlp_setting_1`, `--inst_mlp_setting_2`, `--inst_mlp_setting_3`, `--inst_mlp_setting_4`**  
  Instance MLP layer configurations for multi-tube COVID dataset.

- **`--inst_att_setting_1`, `--inst_att_setting_2`, `--inst_att_setting_3`, `--inst_att_setting_4`**  
  Instance attention MLP configurations for multi-tube COVID dataset.

- **`--inst_att_extraction_multi_tube`**  
  If set, extract instance attention for multi-tube models.

- **`--inst_mlp_pretraining_loading`**  
  If set, load pretrained single-tube model for pretraining.

## Evaluations and Attention Extraction
- **`--stat_training_results`**  
  If set, search and collect results from hyperparameter tuning experiments.

- **`--instance_logits_extraction`**  
  If set, extract instance-level features (logits).

- **`--output_dir`**  
  Directory where post-training analysis results will be saved.

- **`--results_collection_dir`**  
  Directory containing results for collection (e.g., pretraining model outputs).

- **`--filtering_dict`**  
  JSON-formatted dictionary used to filter experiments during result collection.
  
- **`--from_filtered_result`**  
  If set, use only the filtered experiment results (based on `--filtering_dict`) for instance logits extraction.

- Arguments for **manual setting** based attention extraction can be found in `flowmil_post_training.py`

## Visualization

- **`--dataset`**  
  Dataset used for visualization (e.g., CRCLC, AML2015, HIVNH, COVID).

- **`--loading_dir`**  
  Directory containing extracted features to be visualized.

- **`--saving_dir`**  
  Directory to save visualization results.

- **`--n_sample_cells`**  
  Number of sampled cells per phenotype for visualization.

- **`--picked_split`**  
  Specific split(s) to visualize. Accepts multiple values.

- **`--COVID_selected_panel`**  
  Selected single panel for the COVID dataset.

- **`--selected_channels`**  
  Channels selected for plotting. Accepts multiple values.

- **`--top_k_cells`**  
  Number of top-ranked cells to plot.

- **`--umap_attention_plotting`**  
  Plot UMAP histograms weighted by attention scores.

- **`--umap_channel_plotting`**  
  Plot UMAP histograms weighted by biomarker levels.

- **`--umap_phenotype_plotting`**  
  Plot phenotype-level UMAP histogram.

- **`--umap_colorbar_plotting`**  
  Add colorbar to UMAP figures.

- **`--cross_split_plotting`**  
  Plot figures across multiple splits.

- **`--sample_specific_plotting`**  
  Generate sample-specific plots.

- **`--biomarker_reaction_plotting`**  
  Visualize biomarker reaction for each phenotype in UMAP space.

- **`--channelwise_att_clean`**  
  Generate channel-wise scatter plots with attention-based gating.

- **`--attention_channel_combination`**  
  Visualize biomarker-level attention levels in 1D histogram.
