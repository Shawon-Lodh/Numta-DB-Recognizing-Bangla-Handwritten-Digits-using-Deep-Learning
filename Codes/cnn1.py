from sklearn.model_selection import KFold
import gc
import keras.backend as K
import numpy as np


from Data_Processing import data_Fetching,get_data,get_key,create_submission,data_aug
from Create_Model import create_model,callback




paths_train_all,path_label_train_all,paths_test_all=data_Fetching()

img_size = 64####################################################################################
X_train_all,y_train_all=get_data(paths_train_all,path_label_train_all,resize_dim=img_size)
print (X_train_all.shape)
print (y_train_all.shape)


X_test_all=get_data(paths_test_all,resize_dim=img_size)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)##################
cvscores = []
Fold = 1
for train, val in kfold.split(X_train_all, y_train_all):
    gc.collect()
    K.clear_session()
    print('Fold: ', Fold)

    X_train = X_train_all[train]
    X_val = X_train_all[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    y_train = y_train_all[train]
    y_val = y_train_all[val]

    # Data Augmentation and Normalization(OPTIONAL) UNCOMMENT THIS FOR AUGMENTATION !!
    # batch_size = 16
    # train_batch, val_batch = data_aug(X_train,X_val,y_train,y_val, batch_size, batch_size)

    batch_size = 16############################
    train_batch, val_batch = data_aug(X_train,X_val,y_train,y_val, batch_size, batch_size)

    # Data Normalization only - COMMENT THIS OUT FOR DATA AUGMENTATION
    #print(X_train)
    #X_train /= 255
    #print(X_train)
    #X_val /= 255   #########################################

    # If model checkpoint is used UNCOMMENT THIS
    #model_name = 'cnn_keras_Fold_'+str(Fold)+'.h5'

    #cb = callback() #####################################################################

    # create model
    model = create_model(img_size, 3)

    # Fit the model for without Data Augmentation - COMMENT THIS OUT FOR DATA AUGMENTATION
    #batch_size = 16 ###############################################
    batch_size=16
    epochs = 5######################################
    #epochs=20


    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,)##############
    ######################### #callback chilo

    # Fit generator for Data Augmentation - UNCOMMENT THIS FOR DATA AUGMENTATION
    # batch_size = 16
    # epochs = 5
    model.fit_generator(train_batch, validation_data=val_batch, epochs=epochs, validation_steps= X_val.shape[0] // batch_size,
                        steps_per_epoch= X_train.shape[0] // batch_size)

    # Save each fold model
    #model_name = 'cnn_keras_aug_Fold_' + str(Fold) + '.h5'########################################3
    #model.save(model_name)#################################################

    # evaluate the model
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    # save the probability prediction of each fold in separate csv file
    proba = model.predict(X_test_all, batch_size=None, steps=1)
    labels = [np.argmax(pred) for pred in proba]
    keys = [get_key(path) for path in paths_test_all]
    csv_name = 'Test1_Fold' + str(Fold) + '.csv'
    create_submission(predictions=labels, keys=keys, path=csv_name)

    Fold = Fold + 1

print("%s: %.2f%%" % ("Mean Accuracy: ", np.mean(cvscores)))
print("%s: %.2f%%" % ("Standard Deviation: +/-", np.std(cvscores)))