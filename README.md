# CNN-CPU
This is a simple CNN CPU, GPU implementation to classify MNIST data where the model can be submitted as a python Dictionary for the CPU and GPU version.

CPU version is done in Layer abstraction method so that any "Sequential" archticture can be constructed with the layers implemented.

CPU layers :
  1. CNN layer
  2. Fully Connected Layer
  3. Pooling Layer
  4. Initializers Layer (He, Xavier, Constant and Uniform)
  5. Dropout Layer
  6. TanH Layer
  7. Sigmoid Layer
  8. SoftMax Layer
  9. ReLU Layer
  10. Flatten Layer
  
Architecture of the model can be sent as a Dictionary with the parameters for each layer as shown below.

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

To benchmark this against a Keras-GPU implementation a wrapper for Keras is written in GPU.py file so that the same architecture dictionary can be used for CPU and GPU.

