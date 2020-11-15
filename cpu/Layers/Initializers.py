import numpy as np
from .Base import *

class Constant(Base):

    def __init__(self,contsant_value = 0.0):
        Base.__init__(self)
        self.constant_value = contsant_value

    def initialize(self,weights_shape,fan_in, fan_out):
        constant_weights = np.ones(weights_shape)*self.constant_value
        return constant_weights


class UniformRandom(Base):

    def __init__(self):
        Base.__init__(self)
        pass

    def initialize(self,weights_shape,fan_in, fan_out):
        return np.random.uniform(0,1,weights_shape)

class Xavier(Base):

    def __init__(self):
        Base.__init__(self)
        pass

    def initialize(self,weights_shape,fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in + fan_out))
        return np.random.normal(0,sigma,weights_shape)


class He(Base):

    def __init__(self):
        Base.__init__(self)
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        return np.random.normal(0,sigma,weights_shape)

