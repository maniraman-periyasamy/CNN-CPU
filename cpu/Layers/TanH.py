import numpy as np

class TanH:

    def forward(self,input_tensor):
        self.output = np.tanh(input_tensor)
        return self.output

    def backward(self,error_tensor):
        derivative = 1 - self.output**2
        return derivative*error_tensor
