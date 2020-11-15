import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self,input_tensor,label_tensor):
        self.input_tensor = input_tensor
        result = np.sum(-np.log(input_tensor[np.where(label_tensor == 1)]+ np.finfo(float).eps))/input_tensor.shape[0]
        return result

    def backward(self,label_tensor):
        result = -label_tensor/self.input_tensor
        return result