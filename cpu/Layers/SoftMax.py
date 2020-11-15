import numpy as np
from .Base import *

class SoftMax(Base):

    def __init__(self):
        Base.__init__(self)
        pass

    def forward(self, input_tensor):
        input_tensor = input_tensor - input_tensor.max()
        denom = np.repeat(np.sum(np.exp(input_tensor),axis=1),input_tensor.shape[1])
        denom = denom.reshape(input_tensor.shape)
        self.predict = np.exp(input_tensor)/denom
        return self.predict

    def backward(self,label_tensor):

        E_nj = np.repeat(np.sum(np.multiply(label_tensor,self.predict),axis=1),self.predict.shape[1])
        E_nj = E_nj.reshape(self.predict.shape)
        label_tensor_new = label_tensor-E_nj
        return self.predict * label_tensor_new