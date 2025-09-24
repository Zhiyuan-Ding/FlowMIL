import tensorflow as tf
import os
import numpy as np
import json
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from flowmil_utils.models import *

def feature_extraction(dataset, model, saving_dir):
    #gather all samples and labels
    all_features = []
    all_labels = []
    for sample,label in dataset:
        feature = model.predict(sample)
        all_features.append(feature)
        all_labels.append(label)
    all_features = tf.concat(all_features,axis=0)
    all_labels = tf.concat(all_labels,axis=0)
    #save features and labels into one file
    feature_dict = {'features':all_features,'labels':all_labels}
    np.save(saving_dir,feature_dict)

def loglikelihood_debug(mu,sig,data,indices):
    selected_instances = []
    selected_dist = []
    selected_logits = []
    for i in range(indices.shape[0]):
        instance = data[indices[i,0],indices[i,1],:]
        selected_dist.append(np.sum((instance - mu[:,indices[i,2]])**2))
        selected_logits.append( np.sum((instance - mu[:,indices[i,2]])**2 / sig[:,indices[i,2]]**2))
        selected_instances.append(instance)
        print('dist:',selected_dist[-1],'logits:',selected_logits[-1])

def extra_evaluation(model,dataset,args):
    assert args.dataset == 'aml_2015', 'only support aml_2015 dataset'
    bag_preds = []
    ins_preds = []
    bag_labels = []
    ins_labels = []
    for feature, (label, ins_label) in dataset:
        prediction = model.predict(feature)
        bag_preds.append(prediction[0])
        ins_preds.append(prediction[1].flat_values.numpy())
        bag_labels.append(label.numpy())
        ins_labels.append(ins_label.flat_values.numpy())
    
    bag_preds = np.concatenate(bag_preds)
    ins_preds = np.concatenate(ins_preds)
    bag_labels = np.concatenate(bag_labels)
    ins_labels = np.concatenate(ins_labels)
    if args.instance_final_act != 'softmax':
        ins_preds = tf.nn.softmax(ins_preds).numpy()    
    ins_acc = accuracy_score(np.argmax(ins_labels,axis=1), np.argmax(ins_preds,axis = 1))
    ins_f1 = f1_score(np.argmax(ins_labels,axis=1), np.argmax(ins_preds,axis = 1), average='macro')
    ins_auc = roc_auc_score(np.argmax(ins_labels,axis=1), ins_preds,multi_class='ovr')
    bag_acc = accuracy_score(np.argmax(bag_labels,axis = 1), np.argmax(bag_preds,axis = 1))
    bag_f1 = f1_score(np.argmax(bag_labels,axis = 1), np.argmax(bag_preds,axis = 1), average='macro')
    bag_auc = roc_auc_score(np.argmax(bag_labels,axis = 1), bag_preds,multi_class='ovr')
    
    return [ins_acc, ins_f1, ins_auc, bag_acc, bag_f1, bag_auc]

# gather the predicted metrics for the best model
def sample_level_result_collection(args, model, train_dataset, valid_dataset, test_dataset):
    if args.phase == 'evaluation':
        model.load_weights(os.path.join(args.model_loading_dir,'model'))
    else:
        model.load_weights(os.path.join(args.output_dir,'model'))
    test_best_results = model.evaluate(test_dataset)
    train_best_results = model.evaluate(train_dataset)
    if args.dataset == 'aml_2015' and args.load_cell_label:
        test_best_results.extend(extra_evaluation(model,test_dataset,args))
        train_best_results.extend(extra_evaluation(model,train_dataset,args))
    if args.validation:
        valid_best_results = model.evaluate(valid_dataset)
        if args.dataset == 'aml_2015' and args.load_cell_label:
            valid_best_results.extend(extra_evaluation(model,valid_dataset,args))
        best_results = {'test':test_best_results,'train':train_best_results,'valid':valid_best_results}
    else:
        best_results = {'test':test_best_results,'train':train_best_results}
    with open(os.path.join(args.output_dir,'best_results.json'), 'w') as f:
        json.dump(best_results, f, indent=2)

# gather the predicted results per sample for the best model
def sample_level_logits_collection(args, model, train_dataset, valid_dataset, test_dataset):
    if args.phase == 'evaluation':
        model.load_weights(os.path.join(args.model_loading_dir,'model'))
    else:
        model.load_weights(os.path.join(args.output_dir,'model'))
    prefix = ['train','valid','test']
    for (current_prefix,current_dataset) in zip(prefix,[train_dataset,valid_dataset,test_dataset]):
        predictions = []
        labels = []
        for d in current_dataset:
            if args.dataset in ['CRCLC']:
                current_data = tf.RaggedTensor.from_tensor(d[0][tf.newaxis,:,:],ragged_rank=1)
            elif args.dataset in ['hivnh']:
                current_data = d[0]
            elif args.dataset in ['aml_2015']:
                current_data = d[0]
            predictions.append(model.predict(current_data))
            labels.append(d[1])
        f = open(os.path.join(args.output_dir,current_prefix+'_predict.pkl'), 'wb')
        pickle.dump({'predictions':predictions,'label':labels}, f)
        f.close()

# gather the predicted results per instance for the best model
def inst_level_logits_collection(args, model, train_dataset, valid_dataset, test_dataset):
    model.load_weights(os.path.join(args.model_loading_dir,'model'))
    if args.model in[ 'mil_v0']:
        model_layers = grab_mid_layers(model,'map_tensor_function_ragged')
    elif args.model in ['abmil_sh','abmil_mh']:
        if args.abmil_att_encoding == 'attention_dependent':
            model_layers = grab_mid_layers(model,'map_tensor_function_ragged_1')
        elif args.abmil_att_encoding == 'attention_independent':
            model_layers = grab_mid_layers(model,'map_tensor_function_ragged_1')
            model_layers = [layer for layer in model_layers if layer.name != 'map_tensor_function_ragged']

    elif args.model in ['simple_mlp']:
        model_layers = [model]
    if args.dataset in ['hivnh']:
        assert os.path.isfile('./preprocessing_used_files/hivnh_mean_fcs.npy'), 'dir wrong for hivnh preprocessed results'
        mean = np.load('./preprocessing_used_files/hivnh_mean_fcs.npy')
        std = np.load('./preprocessing_used_files/hivnh_std_fcs.npy')
    prefix = ['train','valid','test']
    for (current_prefix,current_dataset) in zip(prefix,[train_dataset,valid_dataset,test_dataset]):
        mid_outputs = []
        labels = []
        sample_names = []
        inst_labels = []
        predictions = []
        if args.dataset in ['CRCLC','hivnh','COVID']:
            for d in current_dataset:
                #transform data into ragged
                if args.dataset in ['CRCLC']:
                    current_data = tf.RaggedTensor.from_tensor(d[0][tf.newaxis,:,:],ragged_rank=1)
                elif args.dataset in ['hivnh']:
                    current_data = (d[0] - mean) / std
                elif args.dataset in ['COVID']:
                    assert len(args.selected_panels) == 1
                    current_data = d[0]

                if (args.model == 'simple_mlp')*(args.feature_type == 'cells'):
                    current_sample_mid_output = grab_mid_outputs(model_layers,current_data.flat_values)
                    sum_output = tf.reduce_sum(current_sample_mid_output[-1],axis=0)
                    current_sample_mid_output.append(sum_output/tf.reduce_sum(sum_output))
                    mid_outputs.append(current_sample_mid_output)
                else: 
                    current_mid_output = grab_mid_outputs(model_layers,current_data)
                    del current_mid_output[0]
                    current_mid_output = [x.numpy() for x in current_mid_output]
                    max_inst_size = int(2e4)
                    if current_mid_output[0].shape[1]>max_inst_size:
                        randomly_selected_instances = np.random.choice(np.arange(current_mid_output[0].shape[1]),max_inst_size)
                        current_mid_output = [x[0,randomly_selected_instances] for x in current_mid_output]
                    mid_outputs.append(current_mid_output)
                predictions.append(model.predict(current_data,verbose = 0))
                labels.append(d[1])
        elif args.dataset in ['aml_2015']:
            current_mid_output = grab_mid_outputs(model_layers,tf.constant(current_dataset))
            del current_mid_output[0]
            mid_outputs.append(current_mid_output)
            if args.dataset in ['CRCLC','hivnh']:
                sample_names.append(d[-1])
            elif args.dataset in ['aml_2015']:
                inst_labels.append(np.concatenate([0*np.ones(len(current_dataset)//3),1*np.ones(len(current_dataset)//3),2*np.ones(len(current_dataset)//3)],axis = 0))            
        if args.dataset == 'aml_2015':
            f = open(os.path.join(args.output_dir,current_prefix+'_att.pkl'), 'wb')
        else:
            f = open(os.path.join(args.output_dir,current_prefix+'_att_split_'+str(args.n_split)+'.pkl'), 'wb')
        if args.dataset in ['aml_2015']:
            pickle.dump({'mid_output':mid_outputs,'inst_label':inst_labels}, f)
        else:
            pickle.dump({'mid_output':mid_outputs,'label':labels,'sample_names':sample_names,'prediction':predictions}, f)
        f.close()

# gather the predicted results per instance for the best model, multi tube
def inst_level_logits_collection_multi_tube(args, model, train_dataset, valid_dataset, test_dataset):
    model.load_weights(os.path.join(args.model_loading_dir,'model'))
    assert args.model in ['mil_v0']
    if args.model in ['mil_v0']:
        tube_layers = grad_mid_layers_multi_tube(model,'map_tensor_function_ragged',len(args.selected_panels))
    assert args.dataset in ['COVID']
    prefix = ['train','valid','test']
    max_inst_size = 30000
    for (current_prefix,current_dataset) in zip(prefix,[train_dataset,valid_dataset,test_dataset]):
        att = {f'tube_{i}':[] for i in range(1,len(args.selected_panels)+1)}
        raw_data = {f'tube_{i}':[] for i in range(1,len(args.selected_panels)+1)}
        labels = []
        sample_names = []
        predictions = []
        if args.dataset in ['COVID']:
            for d in current_dataset:
                for nt,current_layer in enumerate(tube_layers):
                    current_tube_data = d[nt]
                    current_tube_att = current_layer[0](current_tube_data)
                    if len(current_tube_data.flat_values)>max_inst_size:
                        rand_ind = np.random.choice(np.arange(len(current_tube_data.flat_values)),max_inst_size)
                        current_tube_data = current_tube_data.flat_values.numpy()[rand_ind,:]
                        current_tube_att = current_tube_att.flat_values.numpy()[rand_ind,:]
                    else:
                        current_tube_data = current_tube_data.flat_values.numpy()
                        current_tube_att = current_tube_att.flat_values.numpy()
                    att['tube_'+str(nt+1)].append(current_tube_att)
                    raw_data['tube_'+str(nt+1)].append(current_tube_data)
                if args.return_sample_info:
                    labels.append(d[-2])
                    sample_names.append(d[-1])
                    predictions.append(model.predict(d[0:-2]))
                else:
                    labels.append(d[-1])
                    predictions.append(model.predict(d[0:-1]))
            f = open(os.path.join(args.output_dir,current_prefix+'_att_split_'+str(args.n_split)+'.pkl'), 'wb')
            pickle.dump({'inst_data':raw_data,'inst_att':att,'label':labels,'sample_names':sample_names,'prediction':predictions}, f)
            f.close()


