import tensorflow as tf

def custom_training_process(model,custom_layers, epochs, train_dataset, val_dataset,callbacks,losses,optimizer,tuned_epochs=100):
    """
    custom training process for mrd model
    model: tf.keras.Model
    custom_layers: list of layers
    epoches: int
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    optimizer: tf.keras.optimizers
    loss_fn: tf.keras.losses
    saving_path: str
    saving_name: str
    mode: int
    """
    other_layers = [x for x in model.layers if x not in custom_layers]
    #warm up training for the whole model
    warmup_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(optimizer=warmup_optimizer,loss=losses)
    model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=callbacks)
    
    model.compile(optimizer=optimizer,loss=losses)
    for n_ep in range(epochs // tuned_epochs):
        for layer in custom_layers:
            layer.trainable = False
        for layer in other_layers:
            layer.trainable = True
        model.fit(train_dataset, epochs=tuned_epochs//10, validation_data=val_dataset, callbacks=callbacks)
        #set all the other layers to be trainable
        for layer in custom_layers:
            layer.trainable = True
        for layer in other_layers:
            layer.trainable = False
        model.fit(train_dataset, epochs=tuned_epochs, validation_data=val_dataset, callbacks=callbacks)
        

def iterative_fitting_process(model, selected_layers, train_dataset,valid_dataset,  losses, optimizer,callbacks,epochs, tuned_epochs = 10):
    other_layers = [x for x in model.layers if x not in selected_layers]
    model.compile(optimizer=optimizer,loss=losses,loss_weights=[0.5,0.5])
    for n_ep in range(epochs // (2*tuned_epochs)):
        for layer in selected_layers:
            layer.trainable = True
        for layer in other_layers:
            layer.trainable = False
        model.fit(train_dataset, epochs=tuned_epochs+n_ep*tuned_epochs*2, validation_data=valid_dataset, callbacks=callbacks,initial_epoch=n_ep*tuned_epochs*2,verbose=2)
        for layer in selected_layers:
            layer.trainable = False
        for layer in other_layers:
            layer.trainable = True
        model.fit(train_dataset, epochs=n_ep*tuned_epochs*2+2*tuned_epochs, validation_data=valid_dataset, callbacks=callbacks,initial_epoch=n_ep*tuned_epochs*2+tuned_epochs,verbose=2)

def evaluation_with_dropout(model,dataset,losses,metrics):
    
    if isinstance(losses,dict):
        losses = [losses[x] for x in losses]
    preds = []
    labels = []
    for d in dataset:
        if d[0].flat_values.shape[0] <5:
            continue
        pred = model(d[0],training=True)
        label = d[1]
        preds.extend(pred)
        labels.extend(label)
    preds = tf.stack(preds,axis=0)
    labels = tf.stack(labels,axis=0)
    results = []
    results.extend([losses[i](labels,preds).numpy() for i in range(len(losses))])
    results.extend([metrics[i](labels,preds).numpy() for i in range(len(metrics))]) 
    
    return results
        
        
        
    