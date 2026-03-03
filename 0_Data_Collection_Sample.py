from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from tensorflow.keras import backend as K
from itertools import combinations
from scipy.spatial import distance
import sys
from tensorflow.keras.constraints import unit_norm
from itertools import combinations
import h5py
import random
from tensorflow.keras.layers import Input, Dense, Permute, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
import random
import os

sys.setrecursionlimit(10000)
nb_classes = 10

def SimpleNet():
    inputs = Input((32, 32, 3,))

    x1 = Conv2D(8, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.), name='conv1')(inputs)
    x = MaxPooling2D((2, 2))(x1)

    x2 = Conv2D(16, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.), name='conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x2)

    x3 = Conv2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.), name='conv5')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x3)

    x4 = Conv2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.), name='conv8')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x4)

    x = Flatten()(x)
    dense = Dense(64, activation='relu', name='dense1')(x)
    x = Dense(10, name='dense5')(dense)
    x = Activation("softmax")(x)


    model = Model(inputs, [x])
    return model
    
for trial in range(5):    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    start = True
    rate = np.random.uniform(0.3, 0.9, (1,))[0]
    LR = np.random.uniform(0.00001, 0.004, (1,))[0]
    BS = int(np.random.uniform(32, 50000*(rate**6), (1,))[0])
    
    for Sample in range(7):
        print(rate)
        Idx = list(range(x_train.shape[0]))
        random.shuffle(Idx)
        length = int(50000*(rate**float(Sample)))
        print(length)
        Idx=Idx[:length]
        x_train=x_train[Idx,:,:,:]
        y_train=y_train[Idx]
        print(x_train.shape)

        input_shape = x_train.shape[1:]

        lll = 0.0
        datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.,
                zoom_range=0.3,
                fill_mode='nearest',
                cval=0.,
        )

        model = SimpleNet()
        if start==True:
            model.save_weights('Temp_Init.h5')
            start=False
        else:
            model.load_weights('Temp_Init.h5')

        adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        batch_size = BS

        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])	

        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=int(x_train.shape[0] // batch_size),epochs=1, verbose=1, workers=1)

        dict = {'start': 0}

        lcnt=0
        for ll in model.layers:
            temp = ll.get_weights()
            print(ll.name)
            if len(temp)>0:
                if ll.name[0:4] == 'conv':
                    dict['weights'+ str(lcnt)]  = temp[0]
                    if len(temp)>1:
                        dict['bias'+ str(lcnt)]  = temp[1]
                    lcnt = lcnt+1
                if len(temp[0].shape)>2:
                    dict['weights'+ str(lcnt)]  = temp[0]
                    if len(temp)>1:
                        dict['bias'+ str(lcnt)]  = temp[1]
                    lcnt = lcnt+1
                if ll.name[0:5] == 'dense':
                    dict['weights'+ str(lcnt)]  = temp[0]
                    if len(temp)>1:
                        dict['bias'+ str(lcnt)]  = temp[1]
                    lcnt = lcnt+1

        sio.savemat('./Save/SiImpleCNN_CIFAR10_Weights_Trial_'+str(trial)+'_SamplingRate_'+str(rate)+'_Power_'+str(Sample)+'_LR_'+str(LR)+'_BS_'+str(BS)+'.mat', mdict = dict)

        K.clear_session()
