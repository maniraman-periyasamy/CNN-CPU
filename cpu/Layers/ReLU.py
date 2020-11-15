import numpy as np
from .Base import *

class ReLU(Base):

    def __init__(self):
        Base.__init__(self)

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        activated_input = input_tensor.clip(min=0)
        return activated_input

    def backward(self,error_tensor):
        return np.where(self.input_tensor<=0,0,error_tensor)