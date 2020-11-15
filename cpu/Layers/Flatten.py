import numpy as np
from .Base import *

class Flatten(Base):

    def __init__(self):
        Base.__init__(self)
        pass

    def forward(self,input_tensor):
        self.input_tensor_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0],-1)

    def backward(self,error_tensor):
        error_tensor = error_tensor.reshape(self.input_tensor_shape)
        return error_tensor


