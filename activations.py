__author__ = 'h_hack'
import numpy,math
# this module defines thee various activation functions
# 1. sigmoid


def sigmoid(x):
    return 1/(1+numpy.exp(-x))


def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))