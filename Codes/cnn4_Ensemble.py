import numpy as np
import keras
from keras import Model
from keras.optimizers import Adam
from keras.layers import Input
from sklearn.model_selection import train_test_split

from Create_Model import create_model
from Data_Processing import get_key, create_submission,data_Fetching,get_data
#from cnn4 import X_train_all, y_train_all, X_test_all, paths_test_all


def ensemble(models, model_input):
    Models_output=[ model(model_input) for model in models]
    Avg = keras.layers.average(Models_output)

    modelEnsemble = Model(inputs=model_input, outputs=Avg, name='ensemble')
    modelEnsemble.summary()
    modelEnsemble.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return modelEnsemble

img_size=32

model_1 = create_model(img_size,3)
model_2 = create_model(img_size,3)
model_3 = create_model(img_size,3)
model_4 = create_model(img_size,3)
model_5 = create_model(img_size,3)

models = []

# Load weights
print("Load Weights")

model_1.load_weights('cnn_keras_aug_Fold_1.h5')
model_1.name = 'model_1'
models.append(model_1)

model_2.load_weights('cnn_keras_aug_Fold_2.h5')
model_2.name = 'model_2'
models.append(model_2)

model_3.load_weights('cnn_keras_aug_Fold_3.h5')
model_3.name = 'model_3'
#models.append(model_3)

model_4.load_weights('cnn_keras_aug_Fold_4.h5')
model_4.name = 'model_4'
models.append(model_4)

model_5.load_weights('cnn_keras_aug_Fold_5.h5')
model_5.name = 'model_5'
#models.append(model_5)

print("Model input: "+str(models[0].input_shape[1:]))
model_input = Input(shape=models[0].input_shape[1:])
ensemble_model = ensemble(models, model_input)


#########################################################################
paths_train_all,path_label_train_all,paths_test_all=data_Fetching()

img_size = 32####################################################################################
X_train_all,y_train_all=get_data(paths_train_all,path_label_train_all,resize_dim=img_size)
print (X_train_all.shape)
print (y_train_all.shape)


X_test_all=get_data(paths_test_all,resize_dim=img_size)

import pickle
with open("Xy_Train_Test.pickle", "wb") as f:
    pickle.dump((X_train_all,y_train_all,X_test_all), f)

with open("Xy_Train_Test.pickle", "rb") as f:
    X1_train_all, y1_train_all, X1_test_all = pickle.load(f)


#############################################################################

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
scores = ensemble_model.evaluate(X_val, y_val, verbose=0)
print("%s: %.2f%%" % (ensemble_model.metrics_names[1], scores[1]*100))

model_name = 'cnn4_keras_ensebmle_only_2.h5'
ensemble_model.save(model_name)

##############################################################3
from keras.models import load_model
ensemble_model=load_model('cnn4_keras_ensebmle.h5')
######################################################
proba = ensemble_model.predict(X_test_all,batch_size=None,steps=1)
labels=[np.argmax(pred) for pred in proba]
keys=[get_key(path) for path in paths_test_all ]
csv_name= 'submission_CNN4_keras_ensemble.csv'
create_submission(predictions=labels,keys=keys,path=csv_name)