from keras import callbacks
from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,Dropout
from keras.optimizers import Adam,SGD


def create_model(img_size=32, channels=3):
    model = Sequential()
    input_shape = (img_size, img_size, channels)
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    #model.add(Dense(64))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    ################################################
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    ################################################
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=.0001,), loss='categorical_crossentropy', metrics=['accuracy'])
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    model.summary()

    return model


def callback(model_name,tf_log_dir_name='./tf-log/',patience_lr=10,):
    cb = []
    """
    Tensorboard log callback
    """
    tb = callbacks.TensorBoard(log_dir=tf_log_dir_name, histogram_freq=0)
    cb.append(tb)

    """
    Model-Checkpoint
    """
    m = callbacks.ModelCheckpoint(filepath=model_name,monitor='val_loss',mode='auto',save_best_only=True)
    cb.append(m)

    """
    Reduce Learning Rate
    """
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    cb.append(reduce_lr_loss)

    """
    Early Stopping callback
    """
    # Uncomment for usage
    early_stop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
    cb.append(early_stop)



    return cb