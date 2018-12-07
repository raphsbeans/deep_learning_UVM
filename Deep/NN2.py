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
Neural network with 2 inputs and 2 outputs
All hidden layers have dimension >= 2   
'''

class NN:
    def __init__(self,topology,sig_mode):
        '''
        if sig_mode == true, we apply the sigmoid function to the output of the NN  
        '''
        self.topology = topology
        self.sig_mode = sig_mode
        
        self.W = [np.zeros(1)]
        self.dFdW = [np.zeros(1)]
        self.dF2dW = [np.zeros(1)]
        self.b = [np.zeros(1)]
        self.dFdb = [np.zeros(1)]
        self.dF2db = [np.zeros(1)]
        self.A = []
        
        for i in range(topology.size-1):
            V = 4*np.sqrt(6/(topology[i]+topology[i+1]))*npr.uniform(-1,1,(topology[i+1],topology[i]))
            self.W.append(V)
            self.dFdW.append(np.zeros(topology[i+1],topology[i]))
            self.dF2dW.append(np.zeros(topology[i+1],topology[i]))
            self.b.append(np.zeros(topology[i+1]))
            self.dFdb.append(np.zeros(topology[i+1]))
            self.dF2db.append(np.zeros(topology[i+1]))
            self.A.append(np.zeros(topology[i]))
        self.A.append(np.zeros(topology[-1]))
        self.d=copy.deepcopy(self.A)
        if (topology[0] == 1):
            self.W[1] = np.hstack(self.W[1])
            
        if(topology[-1] == 1):
            self.W[-1] = np.hstack(self.W[-1])
                                                        
    def sig(self,x):
        return 1/(1+np.exp(-x))
    
    def sig_derivate (self,x):
        return self.sig(x) * (1 - self.sig(x))
    
    def feed_forward(self,x):
        self.A[0] = np.array(x)
        self.A[1] = np.hstack(np.tanh(self.W[1]@self.A[0] + self.b[1]))
            
        for i in range(1,self.topology.size-2):
            self.A[i+1] = np.hstack(np.tanh(self.W[i+1]@self.A[i]+ self.b[i+1]))
        n = self.topology.size-1
        self.A[n] = np.hstack(self.W[n]@self.A[n-1] + self.b[n])
        
        if(self.sig_mode):
            return self.sig(self.A[-1])
        else:
            return self.A[-1]
        
    def back_propagation(self):
        #dF
        if(self.sig_mode):
            self.d[-1][0] = self.sig_derivate(self.A[-1][0])
            self.d[-1][1] = 0
        else:
            self.d[-1][0] = 1
            self.d[-1][1] = 0
        
        self.dFdb[-1] = self.d[-1]
        self.dFdW[-1] = np.outer(self.d[-1],self.A[-2])
        self.d[-2] = (self.W[-1].T@self.d[-1])*(1 - self.A[-2]**2)
            
        for i in range(self.topology.size-2):
            n = self.topology.size-2-i #The indice of the W that we are going to work with
            self.dFdW[n] = np.outer(self.d[n],self.A[n-1])
            self.dFdb[n] = self.d[n]
            self.d[n-1]  = (self.W[n].T@self.d[n])*(1-self.A[n-1]**2)
        #dF2
        if(self.sig_mode):
            self.d[-1][0] = 0
            self.d[-1][1] = self.sig_derivate(self.A[-1][1])
        else:
            self.d[-1][0] = 0
            self.d[-1][1] = 1
        
        self.dF2db[-1] = self.d[-1]
        self.dF2dW[-1] = np.outer(self.d[-1],self.A[-2])
        self.d[-2] = (self.W[-1].T@self.d[-1])*(1 - self.A[-2]**2)
            
        for i in range(self.topology.size-2):
            n = self.topology.size-2-i #The indice of the W that we are going to work with
            self.dF2dW[n] = np.outer(self.d[n],self.A[n-1])
            self.dF2db[n] = self.d[n]
            self.d[n-1]  = (self.W[n].T@self.d[n])*(1-self.A[n-1]**2)        
            
            
    def back_propagation_data_test(self,y,t):
        if(self.sig_mode):
            self.d[-1] = self.sig_derivate(self.A[-1])*(y-t)
        else:
            self.d[-1] = y-t
        
        #The case we have more then one end
        self.dFdb[-1] = self.d[-1]
        self.dFdW[-1] = np.outer(self.d[-1],self.A[-2])
        self.d[-2] = (self.W[-1].T@self.d[-1])*(1 - self.A[-2]**2)
            
        for i in range(self.topology.size-2):
            n = self.topology.size-2-i #The indice of the W that we are going to work with
            self.dFdW[n] = np.outer(self.d[n],self.A[n-1])
            self.dFdb[n] = self.d[n]
            self.d[n-1]  = (self.W[n].T@self.d[n])*(1-self.A[n-1]**2)
            
        return self.d[-1]
        
    def evaluate(self,x):
        #Case dimention of x is more than 1
        if (self.topology[0] == 1):
            A = np.hstack(np.tanh(self.W[1]*x + np.hstack(self.b[1])))
        else:
            A = np.hstack(np.tanh(self.W[1]@x + self.b[1]))
            
        for i in range(1,self.topology.size-2):
            A = np.hstack(np.tanh(self.W[i+1]@A + self.b[i+1]))
        n = self.topology.size-1
        A = np.hstack(self.W[n]@A + self.b[n])
        return self.sig(A)
    
    def training_test(self,N,X,T,e):   
        for i in range(N):
            y = self.feed_forward(X[i])
            self.back_propagation_data_test(y,T[i])
            for j in range(1,self.topology.size):
                self.W[j] = self.W[j] - e*self.dFdW[j] 
                self.b[j] = self.b[j] - e*self.dFdb[j] 

if __name__ == "__main__":
    import time
    np.random.seed(0)
    
    N = 1000000
    X = np.zeros((N,2))
    X[:,0] = npr.uniform(0,2*m.pi,N) 
    X[:,1] = npr.uniform(0,2*m.pi,N) 
    
    Z = np.sum(X,1)/2
    
    T = np.zeros((N,2))
    T[:,0] = np.sin(Z)
    T[:,1] = np.cos(Z)
    
    top = np.array([2,5,5,2])
    nn = NN(top,False)
    start_time = time.time()
    nn.training_test(N,X,T,0.1)
    
    
    # Tests the NN with multidimensional inputs/outputs
    M = 20000
    X = np.zeros((M,2))
    X[:,0] = npr.uniform(0,2*m.pi,M) 
    X[:,1] = npr.uniform(0,2*m.pi,M)
    test = np.zeros((M,2))
    for i in range(M):
        test[i] = nn.feed_forward(X[i])
    
    X_test = npr.uniform(0,2*m.pi,M)
    plt.scatter(np.sin(X_test),np.cos(X_test))
    plt.scatter(test[:,0],test[:,1])
    