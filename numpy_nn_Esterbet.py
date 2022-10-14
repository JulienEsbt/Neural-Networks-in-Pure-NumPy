#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Julien ESTERBET and Hugo LEGEARD
"""

#%%
import struct
import numpy as np
import time
import matplotlib.pyplot as plt

# provided function for reading idx files
def read_idx(filename):
    '''Reads an idx file and returns an ndarray'''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Task 1: reading the MNIST files into Python ndarrays
x_test = read_idx('/Users/julienesbt/Documents/Etudes/M1/BDA/TP/TP2/mnist/t10k-images.idx3-ubyte')
y_test = read_idx('/Users/julienesbt/Documents/Etudes/M1/BDA/TP/TP2/mnist/t10k-labels.idx1-ubyte')
x_train = read_idx('/Users/julienesbt/Documents/Etudes/M1/BDA/TP/TP2/mnist/train-images.idx3-ubyte')
y_train = read_idx('/Users/julienesbt/Documents/Etudes/M1/BDA/TP/TP2/mnist/train-labels.idx1-ubyte')
# print(x_test, y_test, x_train, y_train)
        
# Task 2: visualize a few bitmap images
# plt.imshow(x_test[0])
# plt.imshow(x_test[1])
# plt.imshow(x_test[2])
# plt.imshow(x_train[0])
# plt.imshow(x_train[1])
# plt.imshow(x_train[2])
        
# Task 3: input pre-preprocessing    
x_train_flattened = x_train.reshape(60000, 784)
# print(x_train_flattened)
x_test_flattened = x_test.reshape(10000, 784)
# print(x_test_flattened)
normal_x_train_flattened = x_train_flattened/255
# print(normal_x_train_flattened)
normal_x_test_flattened = x_test_flattened/255
# print(normal_x_test_flattened)

# Task 4: output pre-processing
def outputPreProcessing(filename) :
    res = np.zeros((len(filename), 10), dtype=int)
    res[np.arange(len(filename)), filename] = 1
    return res
        
y_train_preprocessing = outputPreProcessing(y_train)
y_test_preprocessing = outputPreProcessing(y_test)
        
# Task 5-6: creating and initializing matrices of weights
def layer_weights(m, n) :
    """returns the set of weights between two layers as a matrix with shape (m,n), where m is the number of neurons in the first layer, and n is the number of neurons in the second layer. The weights must be initialized according to the standard normal distribution, and scaled by the factor 1/√n."""
    return np.random.normal(0, 1/np.sqrt(n), (m, n))

layer = layer_weights(784, 128)

def weightsOfMatrices(input, layer1, layer2, output) :
    w1 = layer_weights(input, layer1)
    w2 = layer_weights(layer1, layer2)
    w3 = layer_weights(layer2, output)
    return w1, w2, w3

def sigmoid(x) : 
    return 1/(1+np.exp(-x))

def softmax(x) : 
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

def derivative_sigmoid(x) :
    return sigmoid(x)*(1-sigmoid(x))



# Task 7: defining functions sigmoid, softmax, and sigmoid'
        
# Task 8-9: forward pass
        
# Task 10: backpropagation
        
# Task 11: weight updates
        
# Task 12: computing error on test data
        
# Task 13: error with initial weights
        
# Task 14-15: training

# Task 16-18: batch training