# coding:utf-8
'''
Created on 2017/11/2

@author: sunyihuan
'''

test = "Hello World"

print("test:" + test)

import math
import time
import numpy as np


def basic_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    s = basic_sigmoid(x)
    ds = s * (1 - s)

    return ds


# print(basic_sigmoid(3))

x = np.array([1, 2, 3])

# print(sigmoid_derivative(x))
# print(x.shape[0])


def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v


image = np.array([[[0.6899898, 0.67899], [0.243543363, 0.52619387], [0.3243535, 0.377478848]],
                  [[0.18273, 0.09777], [0.6890, 0.132453], [0.366373773, 0.373737]],
                  [[0.23883, 0.17263], [0.473838, 0.2636738], [0.243536, 0.4374663]]
                  ])
# print(image2vector(image))


def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x


x0 = np.array([[0, 3, 4],
               [2, 6, 4]])

# print(normalizeRows(x0))


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


x1 = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])
# print(softmax(x1))

x1 = [9, 2, 3, 0, 0, 0, 5, 7, 3, 0, 9, 8, 7, 0, 0, 9]
x2 = [8, 8, 0, 2, 1, 3, 4, 5, 6, 7, 9, 6, 3, 2, 5, 7]
tic = time.process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print('dot=' + str(dot) + "\n--------Computation time=" + str(toc - tic) + "s")


def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss


yhat = np.array([.9, .2, .1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print(L1(yhat, y))


def L2(yhat, y):
    loss = np.dot((y - yhat), (y - yhat).T)
    return loss


print(L2(yhat, y))
