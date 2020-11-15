import numpy as np

class Sigmoid:

    def forward(self,input_tensor):
        self.output = 1 / (1 + np.exp(-input_tensor))
        return self.output

    def backward(self,error_tensor):
        derivative = self.output*(1 - self.output)
        return derivative*error_tensor
