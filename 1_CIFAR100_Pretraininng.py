from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
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
from tensorflow.keras.layers import Input, Dense, Permute, Reshape, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate, LeakyReLU, Input, Activation, Conv2D, Dense, Dropout, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
import os
import random
import os
sys.setrecursionlimit(10000)
nb_classes = 100
NumLength=5

   
    
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
    x = Dense(100, activation='softmax', name='fc1000')(x)
    model = Model(img_input, x, name='resnet18')


    return model
    
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32') / 255.
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

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 330:
        lr *= 0.5e-3
    elif epoch > 500:
        lr *= 1e-3
    elif epoch > 160:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

for StartSample in range(0,1):

    model = ResNet18()
    lr_scheduler = LearningRateScheduler(lr_schedule)

    adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    batch_size = BS

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])	
    Idx = list(range(x_train.shape[0]))
    random.shuffle(Idx)
    x_train=x_train[Idx,:,:,:]
    y_train=y_train[Idx]

    lcnt=0
    for ll in model.layers:
        temp = ll.get_weights()
        if len(temp)==2:
            exec('Weights'+ str(lcnt) + '= np.zeros( temp[0].shape + (NumLength,) )')    
            exec('Bias'+ str(lcnt) + '= np.zeros( temp[1].shape + (NumLength,) )')
            lcnt = lcnt+1 
    ii=NumLength-1
    for Sample in range(StartSample,StartSample+5):

        
        length = int(50000*(rate**float(Sample)))
        print(length)

        model.fit_generator(datagen.flow(x_train[0:length,:,:,:], y_train[0:length,:], batch_size=batch_size),steps_per_epoch=int(length // batch_size),epochs=1, verbose=1, workers=1, callbacks=[lr_scheduler])
        model.save_weights('Weights_StartFrom_'+str(StartSample)+'_CurrentSample_'+str(Sample)+'.h5')
        lcnt=0
        for ll in model.layers:
            temp = ll.get_weights()
            if len(temp)==2:
                if len(temp[0].shape)>2:
                    exec('Weights'+ str(lcnt) + '[:,:,:,:,ii]  = temp[0]')
                    exec('Bias'+ str(lcnt) + '[:,ii]  = temp[1]')
                else:
                    exec('Weights'+ str(lcnt) + '[:,:,ii]  = temp[0]')
                    exec('Bias'+ str(lcnt) + '[:,ii]  = temp[1]')
                lcnt = lcnt+1
        ii=ii-1
    dictionary={'init':0}
    lcnt=0
    for ll in model.layers:
        temp = ll.get_weights()
        if len(temp)==2:
            exec("dictionary['Weights'+ str(lcnt)]=Weights"+ str(lcnt))
            exec("dictionary['Bias'+ str(lcnt)]=Bias"+ str(lcnt))

            lcnt = lcnt+1
    sio.savemat('Save_ResNet18_CIFAR100.mat', mdict = dictionary)

    K.clear_session()
