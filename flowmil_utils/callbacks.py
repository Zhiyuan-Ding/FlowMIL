import tensorflow as tf

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data,task,dataset_type='CRCLC',feature_type = 'cells', using_unsupervised_loss_target=False ):
        self.test_data = test_data
        self.task = task
        self.dataset_type = dataset_type
        self.feature_type = feature_type
        self.using_unsupervised_loss_target = using_unsupervised_loss_target
    def on_epoch_end_old(self, epoch, logs):
        if self.task == 'classification':
            if self.dataset_type in ['aml_2015'] and self.feature_type == 'cells':
                weighted_loss, bag_loss, ins_loss = self.model.evaluate(self.test_data,verbose=2)
                logs['test_weighted_loss'] = weighted_loss
                logs['test_bag_loss'] = bag_loss
                logs['test_ins_loss'] = ins_loss
            else:
                loss, auc, f1_score, acc = self.model.evaluate(self.test_data,verbose=2)
                logs['test_loss'] = loss
                logs['test_auc'] = auc
                logs['test_f1_score'] = f1_score
                logs['test_acc'] = acc
        elif self.task == 'regression':
            loss, mse, mae = self.model.evaluate(self.test_data)
            logs['test_loss'] = loss
            logs['test_mse'] = mse
            logs['test_mae'] = mae
    def on_epoch_end(self, epoch, logs):
        if self.task == 'classification':
            if self.using_unsupervised_loss_target:
                loss,supervised_loss,unsupervised_loss, auc, f1_score, acc = self.model.evaluate(self.test_data,verbose=2)
                logs['test_loss'] = loss
                logs['test_auc'] = auc
                logs['test_f1_score'] = f1_score
                logs['test_acc'] = acc
                logs['test_supervised_loss'] = supervised_loss
                logs['test_unsupervised_loss'] = unsupervised_loss
            else:
                loss, auc, f1_score, acc = self.model.evaluate(self.test_data,verbose=2)
                logs['test_loss'] = loss
                logs['test_auc'] = auc
                logs['test_f1_score'] = f1_score
                logs['test_acc'] = acc
        elif self.task == 'regression':
            loss, mse, mae = self.model.evaluate(self.test_data,verbose=2)
            logs['test_loss'] = loss
            logs['test_mse'] = mse
            logs['test_mae'] = mae
class LearningRateLoggingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch,logs):
        lr = self.model.optimizer.lr
        logs['learning_rate'] = lr

def loglearningratecallback():
    result_dic = {"epochs": []}
    json_logging_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs:
                    result_dic["epochs"].append({
                        'epoch': epoch + 1, 
                        'learning_rate': str(logs['learning_rate']),
                    }))
    return json_logging_callback

def logcallback(task,dataset_type = 'CRCLC',feature_type = 'cells',using_unsupervised_loss_target=False):
    result_dic = {"epochs": []}
    if task == 'classification':
        # if dataset_type in ['aml_2015'] and feature_type == 'cells':
        #     json_logging_callback = tf.keras.callbacks.LambdaCallback(
        #                     # on_epoch_begin=lambda epoch, logs: [learning_rate],
        #                     on_epoch_end=lambda epoch, logs:
        #                     result_dic["epochs"].append({
        #                         'epoch': epoch + 1, 
        #                         'test_weighted_loss': str(logs['test_weighted_loss']),
        #                         'test_bag_loss': str(logs['test_bag_loss']),
        #                         'test_ins_loss': str(logs['test_ins_loss']),
        #                     }))
        # else:
        if using_unsupervised_loss_target:
            json_logging_callback = tf.keras.callbacks.LambdaCallback(
                            # on_epoch_begin=lambda epoch, logs: [learning_rate],
                            on_epoch_end=lambda epoch, logs:
                            result_dic["epochs"].append({
                                'epoch': epoch + 1, 
                                'test_loss': str(logs['test_loss']),
                                'test_auc': str(logs['test_auc']),
                                'test_f1_score': str(logs['test_f1_score']),
                                'test_acc': str(logs['test_acc']),
                                'test_supervised_loss':str(logs['test_supervised_loss']),
                                'test_unsupervised_loss':str(logs['test_unsupervised_loss'])
                            }))
        else:
            json_logging_callback = tf.keras.callbacks.LambdaCallback(
                            # on_epoch_begin=lambda epoch, logs: [learning_rate],
                            on_epoch_end=lambda epoch, logs:
                            result_dic["epochs"].append({
                                'epoch': epoch + 1, 
                                'test_loss': str(logs['test_loss']),
                                'test_auc': str(logs['test_auc']),
                                'test_f1_score': str(logs['test_f1_score']),
                                'test_acc': str(logs['test_acc']),

                            }))
    elif task == 'regression':
        json_logging_callback = tf.keras.callbacks.LambdaCallback(
                    # on_epoch_begin=lambda epoch, logs: [learning_rate],
                    on_epoch_end=lambda epoch, logs:
                    result_dic["epochs"].append({
                        'epoch': epoch + 1, 
                        'test_loss': str(logs['test_loss']),
                        'test_mse': str(logs['test_mse']),
                        'test_mae': str(logs['test_mae']),
                    }))
    return json_logging_callback