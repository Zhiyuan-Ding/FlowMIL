import numpy as np
import os

# here we only provide an example of our searching process

# fixed settings for the dataset, model and framework settings
dataset = 'CRCLC'
model = 'mil_v0'
instance_final_act = 'sigmoid'
mil_v0_bag_logits_aggr = 'sum'

# hyperparameters in the network to search with
dropout_list = [0.0, 0.05, 0.1,0.15, 0.2]
len_block_list = [2, 3, 4]
learning_rate_list = [1e-2,5e-3,2e-3,1e-3,5e-4]
weight_decay_list = [1e-3,1e-4,1e-5, 0]

for n_exp in range(20):
    dropout = np.random.choice(dropout_list)
    len_block = np.random.choice(len_block_list)
    learning_rate = np.random.choice(learning_rate_list)
    weight_decay = np.random.choice(weight_decay_list)
    log_max_neuron = np.random.randint(len_block+1,len_block+4)
    inst_mlp_setting = [str(2**i) for i in range(log_max_neuron,log_max_neuron-len_block,-1)]
    for n_split in range(5):
        prefix = 'nohup python flowmil_main.py --dataset %s --model %s --n_exp %d --n_split %d --dropout %.1f --inst_mlp_setting %s --learning_rate %.5f --weight_decay %.5f --validation --mil_v0_bag_logits_aggr %s --result_collection --instance_final_act %s ' % (dataset, model, n_exp,n_split,dropout,' '.join(inst_mlp_setting),learning_rate,weight_decay,mil_v0_bag_logits_aggr,instance_final_act)
        os.system(prefix)