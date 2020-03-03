#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def sigmod(x):
    return 1 / (1 + np.exp(-x))

class neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def feedforward(self, x):
        return sigmod(np.dot(self.w, x) + self.b)


class ourNeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        self.h1 = neuron(weights, bias)
        self.h2 = neuron(weights, bias)
        self.o1 = neuron(weights, bias)

    def feedforward(self, x):
        h1Out = self.h1.feedforward(x)
        h2Out = self.h2.feedforward(x)

        return self.o1.feedforward(np.array([h1Out, h2Out]))

def mess_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

