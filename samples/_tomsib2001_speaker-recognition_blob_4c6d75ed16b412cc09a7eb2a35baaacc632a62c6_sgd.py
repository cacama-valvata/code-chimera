# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import sys
import numpy as np
import random
import os, pickle
import plot
#import kernel_perceptron as kp

# http://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy-machine-learning

def testDef(x):
    return False

class LossFun:
     def __init__(self, lf, grad):
         self.lossFun = lf
         self.grad = grad

def hinge(x):
    return max(0,1-x)

def HingeLoss(xi,yi,w): # b est la dernière coordonnée de w
    return hinge(yi * (np.dot(w[:-1],xi) + w[-1]))
    
        
def HLgrad(xi,yi,w,eps):
    evalfxi = yi * (np.dot(w[:-1],xi) + w[-1])
    delta = evalfxi - 1
    if delta > eps:
        res = np.zeros(shape=len(w))
    elif delta < -eps:
        res = (-yi)*(np.concatenate([xi,np.array([1])]))
    else:    
        res = (-yi/2.)*(np.concatenate([xi,np.array([1])]))
    return(res)

L = LossFun(HingeLoss,HLgrad)

def sgd(x,y,w,Tmax,eta,L,eps,test=testDef):
    eta1 = eta
    t = 1
    theta = 0
    while(t <= Tmax and not(test(x))):   
        lossGrads = np.array([L.grad(x[i],y[i],w,eps) for i in xrange(len(x))])
        v = np.add (np.concatenate([w[:-1],np.array([0])]),10*lossGrads.sum(axis=0)/(len(x)))
        eta = eta1 / np.sqrt(t)
        w = np.subtract(w,eta * v)
        t = t+1
        theta = np.add(theta,w)
    res = theta / (t)
    return(res)

