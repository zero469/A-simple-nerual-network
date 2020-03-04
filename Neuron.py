#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def sigmod(x):
    return 1 / (1 + np.exp(-x))
def deriv_sigmod(x):
    return sigmod(x) * (1 - sigmod(x))

def mess_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def update(self, w, b):
        self.w = w
        self.b = b
    def feedforward(self, x):
        return sigmod(np.dot(self.w, x) + self.b)


class ourNeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.b1 = np.random.normal()
        self.h1 = neuron(np.array([self.w1, self.w2]), self.b1)

        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.b2 = np.random.normal()
        self.h2 = neuron(np.array([self.w3, self.w4]), self.b2)

        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b3 = np.random.normal()
        self.o1 = neuron(np.array([self.w5, self.w6]), self.b3)
    def feedforward(self, x):
        h1Out = self.h1.feedforward(x)
        h2Out = self.h2.feedforward(x)
        return self.o1.feedforward(np.array([h1Out, h2Out]))
    def updata(self):
        self.h1.update(np.array([self.w1, self.w2]), self.b1)
        self.h2.update(np.array([self.w3, self.w4]), self.b2)
        self.o1.update(np.array([self.w5, self.w6]), self.b3)


    def train(self, data, y_trues, learn_rate = 0.1, train_times = 1000):
        for i in range(train_times):
            for x, y_true in zip(data, y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmod(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmod(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmod(sum_o1)

                y_pred = o1

                dL_dpred = -2 * (y_true - y_pred)

                dpred_dw5 = h1 * deriv_sigmod(sum_o1)
                dpred_dw6 = h2 * deriv_sigmod(sum_o1)
                dpred_db3 = deriv_sigmod(sum_o1)
                dpred_dh1 = self.w5 * deriv_sigmod(sum_o1)
                dpred_dh2 = self.w6 * deriv_sigmod(sum_o1)

                dh1_db1 = deriv_sigmod(sum_h1)
                dh1_dw1 = x[0] * deriv_sigmod(sum_h1)
                dh1_dw2 = x[1] * deriv_sigmod(sum_h1)

                dh2_db2 = deriv_sigmod(sum_h2)
                dh2_dw3 = x[0] * deriv_sigmod(sum_h2)
                dh2_dw4 = x[1] * deriv_sigmod(sum_h2)


                self.w1 = self.w1 - learn_rate * dL_dpred * dpred_dh1 * dh1_dw1
                self.w2 = self.w2 - learn_rate * dL_dpred * dpred_dh1 * dh1_dw2
                self.b1 = self.b1 - learn_rate * dL_dpred * dpred_dh1 * dh1_db1

                self.w3 = self.w3 - learn_rate * dL_dpred * dpred_dh2 * dh2_dw3
                self.w4 = self.w4 - learn_rate * dL_dpred * dpred_dh2 * dh2_dw4
                self.b2 = self.b2 - learn_rate * dL_dpred * dpred_dh2 * dh2_db2

                self.w5 = self.w5 - learn_rate * dL_dpred * dpred_dw5
                self.w6 = self.w6 - learn_rate * dL_dpred * dpred_dw6
                self.b3 = self.b3 - learn_rate * dL_dpred * dpred_db3

                self.updata()
            if i % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mess_loss(y_trues, y_preds)
                print(loss)


data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

network = ourNeuralNetwork()
network.train(data, all_y_trues)

emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M