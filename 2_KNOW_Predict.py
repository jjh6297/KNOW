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
import random
import os
import os

sys.setrecursionlimit(10000)
nb_classes = 10
NumLength=5
from KNOWN import *


    
    
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
    
    
  
        
        
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model2 = KNOWN(5)
model2.compile(loss='mae', optimizer=adam,metrics=['mae'])



model = ResNet18()

step=1.0
ftr = sio.loadmat('Save_ResNet18_CIFAR100.mat')
lcnt=0
for ll in model.layers:
    temp = ll.get_weights()
    if len(temp)==2:
        exec("Weights"+ str(lcnt) + "=ftr['Weights"+ str(lcnt) + "'][...,:]")
        exec("Bias"+ str(lcnt) + "=ftr['Bias"+ str(lcnt) + "'][...,:]")
        lcnt = lcnt+1        

for loop in range(3):
    lcnt=0
    for ll in model.layers:
        temp = ll.get_weights()
        if len(temp)==2:
            if len(temp[0].shape)>2:
                model2.load_weights('KNOWN_Conv.h5')
                exec('NewWeight = get_ConvLayer_pred(Weights'+ str(lcnt) + ',model2,NumLength=NumLength)')
                model2.load_weights('KNOWN_Bias.h5')
                exec('NewBias = get_BiasLayer_pred(Bias'+ str(lcnt) + ',model2,NumLength=NumLength)')

            else:
                model2.load_weights('KNOWN_FC.h5')
                exec('NewWeight = get_FCLayer_pred(Weights'+ str(lcnt) + ',model2,NumLength=NumLength)')
                model2.load_weights('KNOWN_Bias.h5')
                exec('NewBias = get_BiasLayer_pred(Bias'+ str(lcnt) + ',model2,NumLength=NumLength)')
            ll.set_weights([NewWeight, NewBias])   
            exec("Weights"+ str(lcnt) + "[...,0:4]=Weights"+ str(lcnt)+ "[...,1:5]")
            exec("Weights"+ str(lcnt) + "[...,4]=NewWeight")

            exec("Bias"+ str(lcnt) + "[...,0:4]=Bias"+ str(lcnt)+ "[...,1:5]")
            exec("Bias"+ str(lcnt) + "[...,4]=NewBias")
            lcnt = lcnt+1 
    model.save_weights('KNOWN_Predicted_Recurrence_'+str(loop)+'.h5')
    
    
    
K.clear_session()
