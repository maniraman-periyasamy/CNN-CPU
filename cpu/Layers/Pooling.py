import numpy as np
from .Base import *

class Pooling(Base):
    def __init__(self, stride_shape, pooling_shape, image_type = "channel_last"):
        Base.__init__(self)
        self.pooling_shape = pooling_shape
        self.stride_shape = stride_shape
        self.image_type = image_type
        self.cache = {}

    def forward(self, input_tensor):

        if self.image_type == "channel_first":
            input_tensor = input_tensor.transpose(0, 2, 3, 1)

        self.input_tensor = np.array(input_tensor, copy=True)

        output = np.zeros((input_tensor.shape[0],
                           1 + (input_tensor.shape[1] - self.pooling_shape[0]) // self.stride_shape[0],
                           1 + (input_tensor.shape[2] - self.pooling_shape[1]) // self.stride_shape[1],
                           input_tensor.shape[3]))

        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                input_tensor_slice = input_tensor[:,
                                            i * self.stride_shape[0]:i * self.stride_shape[0]+self.pooling_shape[0],
                                            j * self.stride_shape[1]:j * self.stride_shape[1]+self.pooling_shape[1],
                                            :]
                self.save_mask(x=input_tensor_slice, cords=(i, j))
                output[:, i, j, :] = np.max(input_tensor_slice, axis=(1, 2))


        if self.image_type == "channel_first":
            output = output.transpose(0, 3, 1, 2)

        return output


    def backward(self, error_tensor):

        if self.image_type == "channel_first":
            error_tensor = error_tensor.transpose(0, 2, 3, 1)

        output = np.zeros_like(self.input_tensor)


        for i in range(error_tensor.shape[1]):
            for j in range(error_tensor.shape[2]):

                output[:,
                        i * self.stride_shape[0]:i * self.stride_shape[0]+self.pooling_shape[0],
                        j * self.stride_shape[1]:j * self.stride_shape[1]+self.pooling_shape[1],
                        :] += \
                    error_tensor[:, i:i + 1, j:j + 1, :] * self.cache[(i, j)]

        if self.image_type == "channel_first":
            output = output.transpose(0, 3, 1, 2)

        return output

    def save_mask(self, x, cords):
        mask = np.zeros_like(x)
        shape = x.shape
        x = x.reshape(shape[0], shape[1] * shape[2], shape[3])
        idx = np.argmax(x, axis=1)
        n_idx, c_idx = np.indices((shape[0], shape[3]))
        mask.reshape((shape[0], shape[1] * shape[2],shape[3]))[n_idx, idx, c_idx] = 1
        self.cache[cords] = mask
        


