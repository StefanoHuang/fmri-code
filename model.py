#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import scipy.io as sio
import gc

from keras.layers import Dense, Input, Conv1D, MaxPooling1D, concatenate, LSTM
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import backend as K
#import ABIDEParser as Reader
from keras.optimizers import Adam
from keras.regularizers import l2

#from kegra.layers.graph import GraphConvolution
#from kegra.utils import *


def DNN_model(input_dim, numClass):
    print("******DNN model****")
    inputs = Input(shape = (input_dim,), name='inputs-layer')
    x = Dense(128,kernel_initializer=initializers.glorot_normal(), name='hidden-layer1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(32, kernel_initializer=initializers.glorot_normal(),name='hidden-layer2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU()(x)
    x= Dropout(0.5, name='dropout2')(x)
    predictions = Dense(units=numClass, activation='softmax', name='predict-layer')(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def Concat_DNN_model(input_dim1,input_dim2,numClass):
    print("******DNN model****")
    inputs1 = Input(shape = (input_dim1,), name='inputs-layer')
    x = Dense(64,kernel_initializer=initializers.glorot_normal(), name='hidden-layer1')(inputs1)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(16, kernel_initializer=initializers.glorot_normal(),name='hidden-layer2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU()(x)
    x= Dropout(0.5, name='dropout2')(x)
    x = Model(inputs=inputs1, outputs=x)


    inputs2 = Input(shape = (input_dim2,), name='inputs-layer_2')
    y = Dense(64,kernel_initializer=initializers.glorot_normal(), name='hidden-layer1_2')(inputs2)
    y = BatchNormalization(name='bn1_2')(y)
    y = LeakyReLU()(y)
    y = Dropout(0.5, name='dropout1_2')(y)
    y = Dense(16, kernel_initializer=initializers.glorot_normal(),name='hidden-layer2_2')(y)
    y = BatchNormalization(name='bn2_2')(y)
    y = LeakyReLU()(y)
    y= Dropout(0.5, name='dropout2_2')(y)
    y = Model(inputs=inputs2, outputs=y)

    combined = concatenate([x.output, y.output])
    predictions = Dense(units=numClass, activation='softmax', name='predict-layer')(combined)
    model = Model(inputs=[x.input, y.input], outputs=predictions)

    return model

def multi_DNN_model(input_dim1,input_dim2,input_dim3, numClass):
    print("******DNN model****")
    inputs1 = Input(shape = (input_dim1,), name='inputs-layer')
    x = Dense(32,kernel_initializer=initializers.glorot_normal(), name='hidden-layer1')(inputs1)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(16, kernel_initializer=initializers.glorot_normal(),name='hidden-layer2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU()(x)
    x= Dropout(0.5, name='dropout2')(x)
    x = Dense(8, kernel_initializer=initializers.glorot_normal(),name='hidden-layer3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout3')(x)
    x = Model(inputs=inputs1, outputs=x)


    inputs2 = Input(shape = (input_dim2,), name='inputs-layer_2')
    y = Dense(32,kernel_initializer=initializers.glorot_normal(), name='hidden-layer1_2')(inputs2)
    y = BatchNormalization(name='bn1_2')(y)
    y = LeakyReLU()(y)
    y = Dropout(0.5, name='dropout1_2')(y)
    y = Dense(16, kernel_initializer=initializers.glorot_normal(),name='hidden-layer2_2')(y)
    y = BatchNormalization(name='bn2_2')(y)
    y = LeakyReLU()(y)
    y= Dropout(0.5, name='dropout2_2')(y)
    y = Dense(8, kernel_initializer=initializers.glorot_normal(),name='hidden-layer3_2')(y)
    y = BatchNormalization(name='bn3_2')(y)
    y = LeakyReLU()(y)
    y = Dropout(0.5, name='dropout3_2')(y)
    y = Model(inputs=inputs2, outputs=y)

    inputs3 = Input(shape = (input_dim3,), name='inputs-layer_3')
    z = Dense(32,kernel_initializer=initializers.glorot_normal(), name='hidden-layer1_3')(inputs3)
    z = BatchNormalization(name='bn1_3')(z)
    z = LeakyReLU()(z)
    z = Dropout(0.5, name='dropout1_3')(z)
    z = Dense(16, kernel_initializer=initializers.glorot_normal(),name='hidden-layer2_3')(z)
    z = BatchNormalization(name='bn2_3')(z)
    z = LeakyReLU()(z)
    z = Dropout(0.5, name='dropout2_3')(z)
    z = Dense(8, kernel_initializer=initializers.glorot_normal(),name='hidden-layer3_3')(z)
    z = BatchNormalization(name='bn3_3')(z)
    z = LeakyReLU()(z)
    z = Dropout(0.5, name='dropout3_3')(z)
    z = Model(inputs=inputs3, outputs=z)

    combined = concatenate([x.output, y.output, z.output])
    predictions = Dense(units=numClass, activation='softmax', name='predict-layer')(combined)
    model = Model(inputs=[x.input, y.input,z.input], outputs=predictions)

    return model


def DNN_seq_model(input_dim, numClass):
    #print("******DNN model****")
    model = Sequential()
    model.add(Dense(32,input_dim=input_dim,kernel_initializer=initializers.glorot_normal(), name='hidden-layer1'))
    model.add(BatchNormalization(name='bn1'))
    model.add(LeakyReLU())
    model.add(Dense(16, kernel_initializer=initializers.glorot_normal(),name='hidden-layer2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(LeakyReLU())
    model.add(Dense(8, kernel_initializer=initializers.glorot_normal(),name='hidden-layer3'))
    model.add(BatchNormalization(name='bn3'))
    model.add(LeakyReLU())
    model.add(Dense(units=numClass, activation='softmax', name='predict-layer'))
    #predictions = Dense(units=numClass, activation='softmax', name='predict-layer')(x)
    return model


'''
def GCN_model(input_dim, numClass,support,G):
    print("******GCN model****")
    inputs = Input(shape = (input_dim,), name='inputs-layer')
    H = Dropout(0.5)(inputs)
    H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H] + G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(numClass, support, activation='softmax')([H] + G)

    model = Model(inputs=[inputs] + G, outputs=Y)
    return model
'''
def DS_CNN1D_model(input_dim1, step, fea,numClass):
    inputs = Input(shape=(input_dim1,), name='inputs-static-layer')
    main_input = Input(shape=(step, fea,), name='main_input')
    x = Dense(32, kernel_initializer=initializers.glorot_normal(), name='hidden-layer1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(16, kernel_initializer=initializers.glorot_normal(), name='hidden-layer2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(8, kernel_initializer=initializers.glorot_normal(), name='hidden-layer3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout3')(x)

    x = Model(inputs=inputs, outputs=x)


    y = Conv1D(filters=500,kernel_size=3,name='conv1',kernel_initializer=initializers.glorot_normal())(main_input)
    y = BatchNormalization(name='bn4')(y)
    y = LeakyReLU()(y)
    y = Conv1D(filters=128,kernel_size=3,name='conv2',kernel_initializer=initializers.glorot_normal())(y)
    y = BatchNormalization(name='bn5')(y)
    y = LeakyReLU()(y)
    y = MaxPooling1D()(y)
    y = Flatten()(y)
    y = Dense(128, name='dense1',kernel_initializer=initializers.glorot_normal())(y)
    y = BatchNormalization(name='bn6')(y)
    y = LeakyReLU()(y)
    y = Dropout(0.5,name='dropout4')(y)

    y = Dense(4, name='dense2', kernel_initializer=initializers.glorot_normal())(y)
    y = BatchNormalization(name='bn7')(y)
    y = LeakyReLU()(y)
    y = Dropout(0.5,name='dropout5')(y)

    y = Model(inputs=main_input, outputs=y)

    combined = concatenate([x.output,y.output])
    z = Dense(2,activation="relu",name="combine1")(combined)
    z = Dense(numClass, activation='softmax', name='predict-layer')(z)
    model = Model(inputs=[x.input,y.input],outputs=z)

    return model

def DS_LSTM_model(input_dim1, step, fea,numClass):
    inputs = Input(shape=(input_dim1,), name='inputs-static-layer')
    main_input = Input(shape=(step, fea,), name='main_input')
    x = Dense(32, kernel_initializer=initializers.glorot_normal(), name='hidden-layer1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(16, kernel_initializer=initializers.glorot_normal(), name='hidden-layer2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(8, kernel_initializer=initializers.glorot_normal(), name='hidden-layer3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5, name='dropout3')(x)

    x = Model(inputs=inputs, outputs=x)


    y = LSTM(500,activation='tanh', name='lstm1',kernel_initializer=initializers.glorot_normal(),dropout=0.5, recurrent_dropout=0.5,return_sequences=True)(main_input)
    y = LSTM(500,activation='tanh',name='lstm2',kernel_initializer=initializers.glorot_normal(),dropout=0.25, recurrent_dropout=0.25,return_sequences=False)(y)
    y = Dense(128, name='dense',kernel_initializer=initializers.glorot_normal(),activation="tanh")(y)
    y = Dropout(0.5,name='dropout')(y)

    y = Dense(4, name='dense2', kernel_initializer=initializers.glorot_normal())(y)
    y = BatchNormalization(name='bn7')(y)
    y = LeakyReLU()(y)
    y = Dropout(0.5,name='dropout5')(y)

    y = Model(inputs=main_input, outputs=y)

    combined = concatenate([x.output,y.output])
    z = Dense(2,activation="relu",name="combine1")(combined)
    z = Dense(numClass, activation='softmax', name='predict-layer')(z)
    model = Model(inputs=[x.input,y.input],outputs=z)

    return model