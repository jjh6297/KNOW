from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.optimizers import  Adam
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from itertools import combinations
from scipy.spatial import distance
import sys
from tensorflow.keras.constraints import unit_norm
import h5py
import random
from tensorflow.keras.layers import Input, Dense, Permute, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import initializers



def trAct_1D_Exp(x,compressed, numExp):

	xShape2=x.shape
	x2 = Dense(compressed, activation='relu')(x)
	temp = Dense(xShape2[1])(x2)
	for jj in range(2,numExp):
		temp = Add()([Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),Lambda(lambda x: K.exp(x))(Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),x]))]),temp])

	return temp	


    
def trAct_1D_Exp(x,compressed, numExp):

	xShape2=x.shape
	x2 = Dense(compressed, activation='relu')(x)
	temp = Dense(xShape2[1])(x2)
	for jj in range(2,numExp):
		temp = Add()([Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),Lambda(lambda x: K.exp(x))(Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),x]))]),temp])

	return temp	



reg = 1e-6
reg2 = 0.1

def KNOWN(NumLength=5):
    inputs = Input(shape=(NumLength,))
    inputs2 = Input(shape=(NumLength-1,))

    fc = Dense(64, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), bias_initializer=initializers.Zeros())(inputs)
    fc = trAct_1D_Exp(fc,8,3)

    fc = Dense(32, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), bias_initializer=initializers.Zeros())(fc)
    fc = LeakyReLU()(fc)



    fc2 = Dense(64, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), bias_initializer=initializers.Zeros())(inputs2)
    fc2 = trAct_1D_Exp(fc2,8,3)

    fc2 = Dense(32, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), bias_initializer=initializers.Zeros())(fc2)
    fc2= LeakyReLU()(fc2)



    fc = Dense(1,activation='tanh')(Concatenate()([fc,fc2]))

    network = Model([inputs,inputs2], fc)
    return network
    

def get_ConvLayer_pred(layer, predictor,NumLength=5):
  
    Layer = np.reshape(layer,[layer.shape[0]*layer.shape[1]*layer.shape[2]*layer.shape[3],layer.shape[4]]) 
    diff = predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0] 
    return np.reshape(Layer[:,NumLength-1] + 1.*diff, [layer.shape[0], layer.shape[1], layer.shape[2], layer.shape[3]]    )



def get_FCLayer_pred(layer, predictor,NumLength=5):
    Layer = np.reshape(layer,[layer.shape[0]*layer.shape[1], layer.shape[2]])                        
    return np.reshape(Layer[:,NumLength-1] + 1.*predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0], [layer.shape[0], layer.shape[1]])
    
    
    
def get_BiasLayer_pred(layer, predictor,NumLength=5):
    return layer[:,NumLength-1] + 1.*predictor.predict([layer, layer[:,1:NumLength] - layer[:,0:NumLength-1]], batch_size = 100000)[:,0]

