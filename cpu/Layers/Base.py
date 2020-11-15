import numpy as np

class Phase:

    def __init__(self):
        self.train = "train"
        self.test = "test"

class Base:

    def __init__(self):
        self.phase = "train"
        self.regularizer = None
