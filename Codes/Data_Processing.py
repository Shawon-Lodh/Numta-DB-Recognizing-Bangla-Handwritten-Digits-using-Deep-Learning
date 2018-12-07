import glob
import os
import cv2
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator



def data_Fetching():
    data_dir = os.path.join('..', 'input')

    #arr_train = ['a']
    arr_train = ['a', 'b', 'c', 'd', 'e']
    iterator_train = len(arr_train)
    print(iterator_train)
    paths_train_all = []

    for i in range(iterator_train):
        print (arr_train[i])
        dirx = 'training-' + arr_train[i]
        paths_train_x = glob.glob(os.path.join(dirx, '*.png'))
        paths_train_all = paths_train_all + paths_train_x

    #arr_test = ['a']
    #arr_test = ['a', 'b', 'c', 'd', 'e', 'f']
    arr_test = ['a', 'b', 'c', 'd', 'e', 'f', 'auga', 'augc']
    iterator_test = len(arr_test)
    paths_test_all = []

    for i in range(iterator_test):
        dirx = 'testing-' + arr_test[i]
        paths_test_x = glob.glob(os.path.join(dirx, '*.png'))
        paths_test_all = paths_test_all + paths_test_x
        if arr_test[i] == 'f':
            paths_test_f = glob.glob(os.path.join(dirx, '*.JPG'))
            paths_test_all = paths_test_all + paths_test_f

    path_label_train_all = []
    for i in range(iterator_train):
        dirx = 'training-' + arr_train[i] + '.csv'
        paths_label_train = glob.glob(dirx)

        path_label_train_all = path_label_train_all + paths_label_train
    print(path_label_train_all)

    return paths_train_all,path_label_train_all,paths_test_all


def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key


def get_data(paths_img, path_label=None, resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    X = []  # initialize empty list for resized images
    for i, path in enumerate(paths_img):
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # images loaded in color (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize_dim is not None:
            img = cv2.resize(img, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)  # resize image to 28x28
        # X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        X.append(img)  # expand image to 28x28x1 and append to the list
        # display progress
        if i == len(paths_img) - 1:
            end = '\n'
        else:
            end = '\r'
        print('processed {}/{}'.format(i + 1, len(paths_img)), end=end)

    X = np.array(X)  # tranform list to numpy array
    if path_label is None:
        return X
    else:

        # Concatenate all data into one DataFrame
        df = pd.DataFrame()
        l = []
        for file_ in path_label:
            df_x = pd.read_csv(file_, index_col=None, header=0)
            l.append(df_x)
        df = pd.concat(l)

        # df = pd.read_csv(path_label[i]) # read labels
        df = df.set_index('filename')
        y_label = [df.loc[get_key(path)]['digit'] for path in paths_img]  # get the labels corresponding to the images
        y = to_categorical(y_label, 10)  # transfrom integer value to categorical variable

        return X, y

def create_submission(predictions,keys,path):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
        )
    result.index.name='key'
    result.to_csv(path, index=True)


def data_aug(X_train,X_test,y_train,y_test,train_batch_size,test_batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    train_batch = train_datagen.flow(X_train,y_train,batch_size=train_batch_size)
    test_batch = test_datagen.flow(X_test,y_test,batch_size=test_batch_size)
    return (train_batch,test_batch)
