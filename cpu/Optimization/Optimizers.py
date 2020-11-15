import numpy as np

class baseOptimizer:

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self,regularizer):
        self.regularizer = regularizer

class Sgd(baseOptimizer):

    def __init__(self, learning_rate=0.01):
        baseOptimizer.__init__(self)
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor,gradient_tensor):
        if self.regularizer != None:
            updated_weight = weight_tensor - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor) - self.learning_rate*gradient_tensor
        else:
            updated_weight = weight_tensor - self.learning_rate*gradient_tensor
        return updated_weight

class SgdWithMomentum(baseOptimizer):

    def __init__(self,learning_rate=0.001, momentum_rate=0.9):
        baseOptimizer.__init__(self)
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.k = 1
        self.v_k = 0

    def calculate_update(self,weight_tensor,gradient_tensor):

        if self.k == 1:
            self.v_k = - self.learning_rate*gradient_tensor
        else:
            self.v_k = self.momentum_rate*self.v_k - self.learning_rate*gradient_tensor
        self.k = self.k + 1

        if self.regularizer != None:
            updated_weight = weight_tensor - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor) + self.v_k
        else:
            updated_weight = weight_tensor + self.v_k
        return updated_weight

class Adam(baseOptimizer):

    def __init__(self,learning_rate=0.001,mu=0.9,rho=0.999):
        baseOptimizer.__init__(self)
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 1
        self.v_k = 0
        self.r_k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.k == 1:
            self.v_k = (1-self.mu)*gradient_tensor
            self.r_k = np.multiply((1-self.rho)*gradient_tensor,gradient_tensor)
        else:
            self.v_k = self.mu*self.v_k + (1-self.mu)*gradient_tensor
            self.r_k = self.rho*self.r_k + np.multiply((1-self.rho)*gradient_tensor,gradient_tensor)

        bias_vk = self.v_k / (1 - self.mu**self.k)
        bias_rk = self.r_k / (1 - self.rho**self.k)
        self.k = self.k + 1

        if self.regularizer != None:
            updated_weight = weight_tensor - self.learning_rate*((bias_vk + np.finfo(float).eps) / (np.sqrt(bias_rk) + np.finfo(float).eps))- self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
        else:
            updated_weight = weight_tensor - self.learning_rate*((bias_vk + np.finfo(float).eps) / (np.sqrt(bias_rk) + np.finfo(float).eps))

        return updated_weight


