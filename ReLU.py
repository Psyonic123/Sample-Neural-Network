# Rectified Linear Function
import numpy as np


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        # Remember input values
        self.inputs = inputs

    def backward(self, dvalues):
        # make a copy of values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative
        self.dvalues[self.inputs <= 0] = 0
