from Layers import *
from Optimization import *
import numpy as np
import NeuralNetwork
import time
import pandas as pd
import matplotlib.pyplot as plt




batch_size = 512
num_classes = 10
epochs = 12

adam = Optimizers.Adam(learning_rate=0.001, mu=0.9, rho=0.999)
model = NeuralNetwork.NeuralNetwork(adam,Initializers.He(),Initializers.Constant(0.0))

model.data_layer = DataHandler.MnistData(batch_size=batch_size,num_classes=num_classes,path='data/mnist.pkl.gz',image_format="channel_first")

model.add(Conv.Conv(stride_shape=(2,2), convolution_shape=(1,4,4), num_kernels=20, convo_type="valid"), trainable=True)
model.add(ReLU.ReLU(),trainable=False)
model.add(Conv.Conv(stride_shape=(2,2), convolution_shape=(1,4,4), num_kernels=40,convo_type="valid"), trainable=True)
model.add(ReLU.ReLU(),trainable=False)
model.add(Pooling.Pooling(stride_shape=(1,1),pooling_shape=(2,2)),trainable=False)

# Below piece of code is manually calculate input shape for fully connected.
poolingInput = (1, 5, 5)
pooling_shape = (2,2)
poolingOut = (40,1+poolingInput[1]-pooling_shape[0],1+poolingInput[2]-pooling_shape[1])
fcl_1_input_size = np.prod(poolingOut)

model.add(Dropout.Dropout(probability=0.25),trainable=False)
model.add(Flatten.Flatten(),trainable=False)
model.add(FullyConnected.FullyConnected(input_size=fcl_1_input_size, output_size=128), trainable=True)
model.add(Dropout.Dropout(probability=0.5),trainable=False)
model.add(FullyConnected.FullyConnected(input_size=128, output_size=num_classes), trainable=True)
model.add(SoftMax.SoftMax(), trainable=False)

model.loss_layer = Loss.CrossEntropyLoss()
start = time.time()
hist_dict = model.train(epochs)
end = time.time()
print("total training time = ", end-start)
df = pd.DataFrame(hist_dict)
df.to_csv("results.csv")

fig = plt.figure()
fig.gca().plot(np.arange(epochs),hist_dict["loss"],'X-', label='training loss', linewidth=1.0)
fig.gca().plot(np.arange(epochs),hist_dict["valLoss"],'o-', label='validation loss', linewidth=1.0)
fig.gca().set_xlim(right = epochs+1)
fig.gca().grid(which='minor', linestyle='--')
fig.gca().set_xlabel('epoch')
fig.gca().set_ylabel('loss')
fig.gca().legend(loc = "upper right", fontsize = 18)
fig.gca().set_title("CPU Code", fontsize = 20)
fig.gca().minorticks_on()
fig.gca().grid(which='minor', linestyle='--')
fig.tight_layout()
fig.savefig("plots/cpu.png",dpi = 300)

model.test()

