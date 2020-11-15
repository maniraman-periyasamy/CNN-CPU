import numpy as np
from .Base import *

class Dropout(Base):

    def __init__(self,probability):
        Base.__init__(self)
        self.probability = probability

    def forward(self,input_tensor):
        if self.phase == "train":

            self.random_index = np.random.binomial(1,1-self.probability,input_tensor.shape).astype(bool)
            input_tensor[self.random_index] = 0
            input_tensor = input_tensor*(1.0/self.probability)

            return input_tensor
        else:
            return input_tensor

    def backward(self,error_tensor):
        error_tensor[self.random_index] = 0

        #   error_tensor= error_tensor*self.random_index
        return error_tensor


