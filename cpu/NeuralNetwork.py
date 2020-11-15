from cpu.Layers import *
from cpu.Optimization import *

import copy
import numpy as np
import time

class NeuralNetwork:

    def __init__(self, optimizer,weights_initializer,bias_initializer,image_format = "channel_last"):
        self.optimizer = optimizer
        self.loss = []
        self.trainLoss = []
        self.trainAcc = []
        self.valLoss = []
        self.valAcc = []
        self.layers=[]
        self.data_layer = []
        self.loss_layer = []
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.image_type = image_format
        self.__phase = None

    def forward(self):

        self.input_tensor, self.lable_tensor = self.data_layer.forward()
        for i in range(len(self.layers)):
            self.input_tensor = self.layers[i].forward(self.input_tensor)
        loss = self.loss_layer.forward(self.input_tensor,self.lable_tensor)
        self.trainCorrectCount = self.trainCorrectCount+self.calculate_accuracy(self.input_tensor, self.lable_tensor, train = True)

        return loss

    def backward(self):
        self.error_tensor = self.loss_layer.backward(self.lable_tensor)
        for i in range(len(self.layers)-1,-1,-1):
            self.error_tensor = self.layers[i].backward(self.error_tensor)

    def createModel(self, modelDict):

        if self.weights_initializer == "Constant":
            self.weights_initializer = Initializers.Constant()

        elif self.weights_initializer == "He":
            self.weights_initializer = Initializers.He()

        elif self.weights_initializer == "Xavier":
            self.weights_initializer = Initializers.Xavier()

        if self.bias_initializer == "Constant":
            self.bias_initializer = Initializers.Constant()

        elif self.bias_initializer == "He":
            self.bias_initializer = Initializers.He()

        elif self.bias_initializer == "Xavier":
            self.weights_initializer = Initializers.Xavier()

        if self.optimizer == "Adam":
            self.optimizer = Optimizers.Adam()
        elif self.optimizer == "SGD":
            self.optimizer = Optimizers.Sgd()
        elif self.optimizer == "SGD_with_momentum":
            self.optimizer = Optimizers.SgdWithMomentum()


        for layer in modelDict['layers']:

            if layer['name'] == 'CNN':
                self.add(Conv.Conv(num_kernels=layer['filters'],convolution_shape=layer['kernel_size'],
                                 stride_shape=layer['strides'],
                                 convo_type=layer['padding'], image_type=self.image_type,
                                   image_channels=layer['image_channels']),trainable=True)
                if layer['activation'] == "relu":
                    self.add(ReLU.ReLU(),trainable=False)
                elif layer['activation'] == "sigmoid":
                    self.add(Sigmoid.Sigmoid(),trainable=False)
                elif layer['activation'] == "tanh":
                    self.add(TanH.TanH(),trainable=False)
                elif layer['activation'] == "softmax":
                    self.add(SoftMax.SoftMax(),trainable=False)

            elif layer['name'] == 'FC':
                self.add(FullyConnected.FullyConnected(input_size=layer['input_shape'],output_size=layer['outputSize'])
                         ,trainable=True)
                if layer['activation'] == "relu":
                    self.add(ReLU.ReLU(),trainable=False)
                elif layer['activation'] == "sigmoid":
                    self.add(Sigmoid.Sigmoid(),trainable=False)
                elif layer['activation'] == "tanh":
                    self.add(TanH.TanH(),trainable=False)
                elif layer['activation'] == "softmax":
                    self.add(SoftMax.SoftMax(),trainable=False)

            elif layer['name'] == 'Dropout':
                self.add(Dropout.Dropout(probability=layer['dropoutRate']), trainable=False)

            elif layer['name'] == 'pool':
                self.add(Pooling.Pooling(stride_shape=layer['strides'],pooling_shape=layer['pool_size'],
                                         image_type=self.image_type,), trainable=False)
            else:
                self.add(Flatten.Flatten(),trainable=False)
        self.loss_layer = Loss.CrossEntropyLoss()




    def add(self,layer, trainable=True):
        if trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer,self.bias_initializer)
        self.layers.append(layer)

    def calculate_accuracy(self, results, labels, train = False):

        index_maximum = np.argmax(results, axis=1)
        one_hot_vector = np.zeros_like(results)
        for i in range(one_hot_vector.shape[0]):
            one_hot_vector[i, index_maximum[i]] = 1

        correct = 0.
        wrong = 0.
        for column_results, column_labels in zip(one_hot_vector, labels):
            if column_results[column_labels > 0.].all() > 0.:
                correct += 1.
            else:
                wrong += 1.
        if train:
            return correct
        return correct / (correct + wrong)

    def train(self,iterations):
        self.phase = "train"
        num_batches = self.data_layer.get_num_batches()
        for i in range(iterations):
            print("Epoch :", i+1, " Out of :", iterations)
            self.trainCorrectCount = 0
            self.data_layer.shuffleIndex()
            epochTrainLoss = 0
            for batch in range(num_batches):
                start = time.time()
                currLoss = self.forward()
                epochTrainLoss+=currLoss
                #self.loss.append(currLoss)
                self.backward()
                end = time.time()
                print("Batch :", batch+1," Out of :", num_batches, " in seconds :", end-start, " loss :", currLoss)
            self.trainLoss.append(epochTrainLoss/num_batches)
            self.trainAcc.append(self.trainCorrectCount / (self.data_layer.get_batch_size()*num_batches))

            valImage, valLabel = self.data_layer.get_val_set()
            # validation process
            for j in range(len(self.layers)):
                valImage = self.layers[j].forward(valImage)
            self.valLoss.append(self.loss_layer.forward(valImage, valLabel))
            self.valAcc.append(self.calculate_accuracy(valImage,valLabel))
            print("epoch : ",i, "\tvalLoss : ", self.valLoss[i], "\tvalAccuracy : ", self.valAcc[i])

        hist_dict = {"loss":self.trainLoss,"acc":self.trainAcc,"val_loss":self.valLoss,"val_acc":self.valAcc}
        return hist_dict



    def test(self):
        self.phase = "test"
        valImage, valLabel = self.data_layer.get_test_set()
        for i in range(len(self.layers)):
            valImage = self.layers[i].forward(valImage)
        loss = self.loss_layer.forward(valImage, valLabel)
        accuracy = self.calculate_accuracy(valImage, valLabel)
        print("test Loss : ", loss, "\ttest Accuracy : ", accuracy)
        return (loss,accuracy)


    @property
    def phase(self):
        return self.__phase

    @phase.setter
    def phase(self,phase):
        self.__phase = phase
        for layer in self.layers:
            layer.phase = phase

