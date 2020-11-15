
# Cpu imports.

from cpu.Layers import *
from cpu.Optimization import *
import cpu.NeuralNetwork as net

import DataHandler
from Gpu import basicCNN

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os



dataPath = 'data/mnist.pkl.gz'

def CNN_Gpu():

    model = basicCNN()
    model.data_Layer = DataHandler.MnistData(batch_size=batch_size, num_classes=num_classes, path=dataPath,
                                             image_format="channel_last",input_shape=(28,28,1))
    model.createModel(architecture)
    start = time.time()
    hist_dict = model.train(epochs)
    end = time.time()
    print("total training time = ", end - start)
    testResult = model.test()

    return hist_dict, [testResult[0], testResult[1], end - start]


def CNN_Cpu():
    model = net.NeuralNetwork(optimizer="Adam",weights_initializer="He",bias_initializer="Constant", image_format = "channel_last")
    model.data_layer = DataHandler.MnistData(batch_size=batch_size, num_classes=num_classes, path=dataPath,
                                             image_format="channel_last", input_shape=(28,28,1))
    model.createModel(architecture)
    start = time.time()
    hist_dict = model.train(epochs)
    end = time.time()
    print("total training time = ", end - start)
    testResult = model.test()

    return hist_dict, [testResult[0], testResult[1], end - start]

def FC_Cpu():
    model = net.NeuralNetwork(optimizer="Adam",weights_initializer="He",bias_initializer="Constant", image_format = "channel_last")
    model.data_layer = DataHandler.MnistData(batch_size=batch_size, num_classes=num_classes, path=dataPath,
                                             image_format="channel_last", input_shape=(784,))
    model.createModel(FullyConnected)
    start = time.time()
    hist_dict = model.train(epochs)
    end = time.time()
    print("total training time = ", end - start)
    testResult = model.test()

    return hist_dict, [testResult[0], testResult[1], end - start]


batch_size = 128
num_classes = 10
epochs = 12

#structue of the architecture - used for both CPU and GPU
architecture = {
        "layers": [
            {
                "name": "CNN",
                "filters":5,
                "kernel_size": (5,5),
                "strides": (2,2),
                "padding": "valid",
                "activation": "relu",
                "input_shape": (28,28,1),
                "image_channels":1
            },
            {
                "name": "CNN",
                "filters":12,
                "kernel_size": (3,3),
                "strides": (1,1),
                "padding": "valid",
                "activation": "relu",
                "input_shape": (1,28,28),
                "image_channels":1

            },
            {
                "name": "pool",
                "pool_size": (2,2),
                "strides": (1,1),
                "input_shape": (1,28,28)
            },
            {
                "name": "dropout",
                "dropoutRate": 0.25
            },
            {
                "name": "Flatten"
            },
            {
                "name": "FC",
                "input_shape": 972,
                "outputSize": 128,
                "activation": "relu"
            },
            {
                "name": "dropout",
                "dropoutRate": 0.5
            },
            {
                "name": "FC",
                "input_shape": 128,
                "outputSize": num_classes,
                "activation": "softmax"
            },
        ],
    }



FullyConnected = {
        "layers": [
            {
                "name": "FC",
                "input_shape": 784,
                "outputSize": 720,
                "activation": "relu"
            },

            {
                "name": "FC",
                "input_shape": 720,
                "outputSize": 1200,
                "activation": "relu"
            },

            {
                "name": "dropout",
                "dropoutRate": 0.25
            },

            {
                "name": "FC",
                "input_shape": 1200,
                "outputSize": 128,
                "activation": "relu"
            },
            {
                "name": "dropout",
                "dropoutRate": 0.5
            },
            {
                "name": "FC",
                "input_shape": 128,
                "outputSize": num_classes,
                "activation": "softmax"
            },
        ],
    }


resultsFolder = "results/"
if not os.path.exists(resultsFolder):
    os.makedirs(resultsFolder)

fnLisit = [CNN_Gpu,CNN_Cpu,FC_Cpu]
MetricDict = {}
for fn in fnLisit:
    hist_dict, metric = fn()

    MetricDict[fn.__name__] = metric
    fig = plt.figure()
    fig.gca().plot(np.arange(epochs),hist_dict["loss"],'X-', label='training loss', linewidth=1.0)
    fig.gca().plot(np.arange(epochs),hist_dict["val_loss"],'o-', label='validation loss', linewidth=1.0)
    fig.gca().set_xlim(right = epochs+1)
    fig.gca().grid(which='minor', linestyle='--')
    fig.gca().set_xlabel('epoch')
    fig.gca().set_ylabel('loss')
    fig.gca().legend(loc = "upper right", fontsize = 18)
    fig.gca().set_title(fn.__name__, fontsize = 20)
    fig.gca().minorticks_on()
    fig.gca().grid(which='minor', linestyle='--')
    fig.tight_layout()
    fig.savefig(resultsFolder+fn.__name__+".png",dpi = 300)

index = ["Loss", "Accuracy", "Time"]
df = pd.DataFrame(data=MetricDict, index=index)
df = df.T
df.to_csv(resultsFolder+"testResults.csv")
