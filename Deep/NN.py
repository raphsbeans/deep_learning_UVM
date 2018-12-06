# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:58:08 2018

@author: Marcos Moreira & Raphael Feijao
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import math as m

'''
A general topology neural network
1)Here we consider that the input and the output will always have dimension 1
2)That's the version we are going to use for the stochastic control. We apply 
a sigmoid function to assure that its output will be at the interval [0,1].
3)We use tanh for all hidden units.     
'''

class NN:
    def __init__(self,topology):
        self.topology = topology
        
        self.W = [np.zeros(1)]
        self.dFdW = [np.zeros(1)]
        self.b = [np.zeros(1)]
        self.dFdb = [np.zeros(1)]
        self.A = []
        
        for i in range(topology.size-1):
            V = 4*np.sqrt(6/(topology[i]+topology[i+1]))*npr.uniform(-1,1,(topology[i+1],topology[i]))
            self.W.append(V)
            self.dFdW.append(np.zeros(topology[i+1],topology[i]))
            self.b.append(np.zeros(topology[i+1]))
            self.dFdb.append(np.zeros(topology[i+1]))
            self.A.append(np.zeros(topology[i]))
        self.A.append(np.zeros(topology[-1]))
        self.d=copy.deepcopy(self.A)
        if (topology[0] == 1):
            self.W[1] = np.hstack(self.W[1])
            
        if(topology[-1] == 1):
            self.W[-1] = np.hstack(self.W[-1])
                                                        
    
        
    def sig(self,x):
        return 1/(1+np.exp(-x))
    
    def feed_forward(self,x):
        self.A[0] = np.array(x)
        self.A[1] = np.hstack(np.tanh(self.W[1]*self.A[0] + np.hstack(self.b[1])))
        for i in range(1,self.topology.size-2):
            self.A[i+1] = np.hstack(np.tanh(self.W[i+1]@self.A[i]+ self.b[i+1]))
        n = self.topology.size-1
        self.A[n] = np.hstack(self.W[n]@self.A[n-1] + self.b[n])
        self.j = self.sig(self.A[-1])
        return self.j
    '''
    Performs the backpropagation of the neural network.
    '''   
    def back_propagation(self):
        self.d[-1] = self.sig(self.A[-1])*(1-self.sig(self.A[-1]))
        self.dFdW[-1] = self.d[-1]*self.A[-2]
        self.dFdb[-1] = self.d[-1]
        self.d[-2] = self.d[-1]*self.W[-1]*(1 - self.A[-2]**2)
        
        for i in range(self.topology.size-2):
            n = self.topology.size-2-i #The indice of the W that we are going to work with
            self.dFdW[n] = np.outer(self.d[n],self.A[n-1])
            self.dFdb[n] = self.d[n]
            self.d[n-1]  = (self.W[n].T@self.d[n])*(1-self.A[n-1]**2)
        
        if(self.topology[0] == 1):
            self.dFdW[1] = np.hstack(self.dFdW[1])
            
        return self.d[-1]
    '''
    Evaluates the derivative of the output with respect to the input.
    It must be called after the backpropagation function.
    '''       
    def der_output(self):
        return self.d[0]/(1-self.A[0]**2)
        
    def evaluate(self,x):
        A = np.hstack(np.tanh(self.W[1]*x+ np.hstack(self.b[1])))
        for i in range(1,self.topology.size-2):
            A = np.hstack(np.tanh(self.W[i+1]@A+ self.b[i+1]))
        n = self.topology.size-1
        A = np.hstack(self.W[n]@A + self.b[n])
        return self.sig(A)

