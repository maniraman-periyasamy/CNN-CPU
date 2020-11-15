
from __future__ import print_function
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import constant, HeNormal
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split


import time
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)




class basicCNN:
    def __init__(self, optimizer = "Adam", weightInitializer = "He", biasInitializer="Constant",biasConstant = 0.0,
                 weightsConstant = 1.0):
        self.optimizer = optimizer
        self.weightInitalizer = weightInitializer
        self.biasInitializer = biasInitializer
        self.biasConstant = biasConstant
        self.weightConstant = weightsConstant

        self.data_Layer = None

    def createModel(self,modelDict):

        self.model = Sequential()

        if self.weightInitalizer == "Constant":
            weightInitializer = constant(self.weightConstant)
        else:
            weightInitializer = "HeNormal"

        if self.biasInitializer == "Constant":
            biasInitializer = constant(self.biasConstant)
        else:
            biasInitializer = "HeNormal"

        firstLayer = modelDict['layers'][0]

        if firstLayer['name'] == 'CNN':

            self.model.add(Conv2D(filters=firstLayer['filters'],input_shape=self.data_Layer.get_input_shape(), kernel_size=firstLayer['kernel_size'],strides=firstLayer['strides'],
                           padding=firstLayer['padding'], activation=firstLayer['activation'],
                             kernel_initializer=weightInitializer, bias_initializer=biasInitializer))
        elif firstLayer['name'] == 'FC':
            self.model.add(Dense(input_shape=firstLayer['input_shape'],units=firstLayer['outputSize'], activation=firstLayer['activation'],
                            kernel_initializer=weightInitializer, bias_initializer=biasInitializer))

        for layer in modelDict['layers'][1:]:

            if layer['name'] == 'CNN':
                self.model.add(Conv2D(filters=layer['filters'],kernel_size=layer['kernel_size'],
                                 strides=layer['strides'],
                                 padding=layer['padding'], activation=layer['activation'],
                                 kernel_initializer=weightInitializer, bias_initializer=biasInitializer))
            elif layer['name'] == 'FC':
                self.model.add(Dense(units=layer['outputSize'], activation=layer['activation'],
                                kernel_initializer=weightInitializer, bias_initializer=biasInitializer))
            elif layer['name'] == 'Dropout':
                self.model.add(Dropout(layer['dropoutRate']))

            elif layer['name'] == 'pool':
                self.model.add(MaxPooling2D(strides=layer['strides'],pool_size=layer['pool_size']))
            else:
                self.model.add(Flatten())

        self.model.compile(loss=categorical_crossentropy, optimizer=self.optimizer,
                      metrics=['accuracy'])


    def train(self,epochs):

        log_dir = "logs/"

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        train = self.data_Layer.get_train_set()
        val = self.data_Layer.get_val_set()
        history = self.model.fit(train[0],train[1],
                            batch_size=self.data_Layer.get_batch_size(), epochs=epochs, verbose=1,
                            validation_data=(val[0], val[1]),callbacks=[tensorboard_callback])
        return history.history

    def test(self):
        test = self.data_Layer.get_test_set()
        result = self.model.evaluate(test[0],test[1])
        return result

