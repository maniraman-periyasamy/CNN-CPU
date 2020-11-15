"""
A stand Alone implementation of Convolution layer which can perform 1D and 2D convolution
"""

import scipy as sp
from scipy.signal import convolve2d
import copy
import math
from .Base import *


class Conv(Base):

    def __init__(self, stride_shape, convolution_shape,image_channels, num_kernels, convo_type="same", image_type="channel_first"):
        Base.__init__(self)
        self.stride_shape = stride_shape
        self.num_kernels = num_kernels
        self.convolution_shape = convolution_shape
        self.image_channels = image_channels
        self.convo_type = convo_type
        self.image_type = image_type
        self.pad_Dim = (0, 0)


        self.convolution_shape = self.convolution_shape + (self.image_channels,)

        self.weights = np.random.uniform(0., 1., (self.convolution_shape + (num_kernels,)))

        if self.convo_type == "same":
            self.pad_Dim = ((np.asarray(self.convolution_shape[:-1]) - 1) / 2.)  # (K-1)/2

        self.__gradient_weights = np.zeros(self.weights.shape)
        self.__gradient_bias = np.zeros(num_kernels)
        self.bias = np.random.randn(num_kernels)

        self.__optimizer = None
        self.optimizerWeigths = None
        self.optimizerBias = None

    def forward(self, input_tensor):

        if self.image_type == "channel_first":
            input_tensor = input_tensor.transpose(0, 2, 3, 1)


        self.input_tensor = input_tensor

        XYShape = (int((self.input_tensor.shape[1] - self.convolution_shape[0] + int(2 * self.pad_Dim[0])) /
                       self.stride_shape[0] + 1), int((self.input_tensor.shape[2] - self.convolution_shape[1]
                                                       + int(2 * self.pad_Dim[1])) / self.stride_shape[1] + 1))

        output = np.zeros(((input_tensor.shape[0],) + XYShape + (self.num_kernels,)), dtype=input_tensor.dtype)
        Input_padded = self.padding(self.input_tensor)

        for i in range(0, output.shape[1]):
            for j in range(0, output.shape[2]):
                output[:, i, j, :] = np.sum(
                    Input_padded[:,
                    i * self.stride_shape[0]:i * self.stride_shape[0] + self.weights.shape[0],
                    j * self.stride_shape[1]:j * self.stride_shape[1] + self.weights.shape[1],
                    :, np.newaxis] * self.weights, axis=(1, 2, 3))

        output += self.bias

        if self.image_type == "channel_first":
            output = output.transpose(0, 3, 1, 2)

        return output

    def backward(self, error_tensor):

        if self.image_type == "channel_first":
            error_tensor = error_tensor.transpose(0, 2, 3, 1)

        error_upsampling_shape = (self.input_tensor.shape[:-1] + (error_tensor.shape[-1],))
        output = np.zeros(self.padding(self.input_tensor).shape)
        paddedInput = self.padding(self.input_tensor)
        self.__gradient_weights = np.zeros(self.weights.shape)

        error_Upsampled = np.zeros(error_upsampling_shape)
        error_Upsampled[:,
                        :error_tensor.shape[1] * self.stride_shape[0]:self.stride_shape[0],
                        :error_tensor.shape[2] * self.stride_shape[1]:self.stride_shape[1],:] = error_tensor

        for i in range(error_tensor.shape[1]):
            for j in range(error_tensor.shape[2]):
                output[:,
                i * self.stride_shape[0]:i * self.stride_shape[0] + self.weights.shape[0],
                j * self.stride_shape[1]:j * self.stride_shape[1] + self.weights.shape[1],
                :] += np.sum(self.weights[np.newaxis, :, :, :, :] * error_tensor[:, i:i + 1, j:j + 1, np.newaxis, :],
                             axis=4)

        for i in range(self.weights.shape[3]):
            for j in range(self.weights.shape[2]):
                self.__gradient_weights[:, :, j, i] = sp.signal.correlate(paddedInput[:, :, :, j],
                                                                          error_Upsampled[:, :, :, i], 'valid')

        for i in range(self.__gradient_bias.shape[0]):
            self.__gradient_bias[i] = np.sum(error_tensor[:, :, :, i])


        if not self.__optimizer is None:
            self.weights = self.optimizerWeigths.calculate_update(self.weights, self.__gradient_weights)
            self.bias = self.optimizerBias.calculate_update(self.bias, self.__gradient_bias)

        if self.image_type == "channel_first":
            output = output.transpose(0, 3, 1, 2)
            output = output[:, :, int(self.pad_Dim[0]):int(self.pad_Dim[0]) + self.input_tensor.shape[1],
               int(self.pad_Dim[1]):int(self.pad_Dim[1]) + self.input_tensor.shape[2]]
        else:
            output = output[:, int(self.pad_Dim[0]):int(self.pad_Dim[0]) + self.input_tensor.shape[1],
                     int(self.pad_Dim[1]):int(self.pad_Dim[1]) + self.input_tensor.shape[2],:]

        return output

    def padding(self, input_tensor):

        if (len(input_tensor.shape) == 3):
            return np.pad(input_tensor, ((0, 0), (math.ceil(self.pad_Dim[0]), int(self.pad_Dim[0])),
                                         (math.ceil(self.pad_Dim[1]), int(self.pad_Dim[1]))))
        return np.pad(input_tensor, (
        (0, 0), (math.ceil(self.pad_Dim[0]), int(self.pad_Dim[0])), (math.ceil(self.pad_Dim[1]), int(self.pad_Dim[1])),
        (0, 0)))

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,
                                                      np.prod(self.convolution_shape),
                                                      self.num_kernels * self.convolution_shape[0] *
                                                      self.convolution_shape[1])
        self.bias = bias_initializer.initialize(self.bias.shape,
                                                np.prod(self.convolution_shape),
                                                np.prod(self.convolution_shape))

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
        self.optimizerWeigths = copy.deepcopy(self.__optimizer)
        self.optimizerBias = copy.deepcopy(self.__optimizer)

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.__gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self.__gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self.__gradient_bias = gradient_bias

