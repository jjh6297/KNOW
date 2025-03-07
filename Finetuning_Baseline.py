from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import np_utils
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate, ZeroPadding2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
# from tensorflow.keras.utils.training_utils import multi_gpu_model
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
# from TrActLayer import trAct_1D_Exp, trAct_2D_Exp
import os
import random
import os

sys.setrecursionlimit(10000)
for trial in range(1,5):
    nb_classes = 10
    NumLength=7
        
    def identity_block(input_tensor, kernel_size, filters, stage, block):

        filters1, filters2 = filters

        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (3, 3),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a',padding='same')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def conv_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2)):

        filters1, filters2 = filters
        bn_axis = 3
     
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (3, 3), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a',padding='same')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        shortcut = Conv2D(filters2, (1, 1), strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1',padding='same')(input_tensor)
        shortcut = BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x


    def ResNet18():

        img_input = Input(shape=(32,32,3))

        bn_axis = 3


        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = Conv2D(64, (7, 7),
                          strides=(1, 1),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)

        x = identity_block(x, 3, [64, 64], stage=2, block='a')
        x = identity_block(x, 3, [64, 64], stage=2, block='b')

        x = conv_block(x, 3, [128, 128], stage=3, block='a', strides=(1, 1))
        x = identity_block(x, 3, [128, 128], stage=3, block='b')

        x = conv_block(x, 3, [256, 256], stage=4, block='a')
        x = identity_block(x, 3, [256, 256], stage=4, block='b')

        x = conv_block(x, 3, [512, 512], stage=5, block='a')
        x = identity_block(x, 3, [512, 512], stage=5, block='b')

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(nb_classes, activation='softmax', name='fc1000')(x)
        model = Model(img_input, x, name='resnet18')


        return model
        
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255.
    print('Training Size: ',x_train.shape)
    x_test = x_test.astype('float32') / 255.
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    start = True
    rate = 0.5
    LR = 0.001
    BS = 128 
    datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.,
            zoom_range=0.3,
            fill_mode='nearest',
            cval=0.,
    )


    def cos_decay(epoch):
        init = 4e-4
        lr = 0.5*init*(1.+np.cos(np.pi*epoch/100))
        print('Learning rate: ', lr)
        return lr
    model = ResNet18()
    lr_scheduler = LearningRateScheduler(cos_decay)
    adam = tf.keras.optimizers.legacy.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    batch_size = BS

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])	
    Idx = list(range(x_train.shape[0]))
    random.shuffle(Idx)
    x_train=x_train[Idx,:,:,:]
    y_train=y_train[Idx]

    model.load_weights('Weights_StartFrom_0_CurrentSample_0.h5', skip_mismatch=True, by_name=True)
    for ll in model.layers:
        WW = ll.get_weights()
        if len(WW)<4:
            print(ll.name)
            ll.trainable=False
    model.get_layer('fc1000').trainable=True  
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])	
    model.summary()              
    hist = model.fit_generator(datagen.flow(x_train[:,:,:,:], y_train[:,:], batch_size=batch_size),steps_per_epoch=int(x_train[:,:,:,:].shape[0] / batch_size),epochs=10, verbose=1, workers=1,validation_data = (x_test, y_test) )



    ii=0
    for ll in model.layers:
        WW = ll.get_weights()
        if len(WW)<4:
            print(ll.name)
            ll.trainable=True
    model.get_layer('fc1000').trainable=True  
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])	
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=int(x_train.shape[0] / batch_size),epochs=100, verbose=1, workers=1,validation_data = (x_test, y_test) , callbacks=[lr_scheduler])
    sio.savemat('Save_Baseline_Finetuned_Trial'+str(trial)+'.mat', mdict = {'Loss':np.array(hist.history['loss']), 'ValLoss':np.array(hist.history['val_loss']), 'Acc':np.array(hist.history['accuracy']), 'ValAcc':np.array(hist.history['val_accuracy'])})
    model.save_weights('Weights_Baseline_Finetuned_Trial_'+str(trial)+'.h5')

    K.clear_session()
