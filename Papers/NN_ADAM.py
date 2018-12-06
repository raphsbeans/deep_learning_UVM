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

class NN:
    '''
    We establish that the output has always dimension 1 
    '''
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
    
    def sig_derivate (self,x):
        #In this part we will consider that x is already the sig
        return x * (1 - x)
    
    def tanh_derivate (self, x):
        #In this part we will consider that x is already the tanh
        return 1 - x**2    
    
    def feed_forward(self,x):
        self.A[0] = np.array(x)
        self.A[1] = np.hstack(np.tanh(self.W[1]*self.A[0] + np.hstack(self.b[1])))
        for i in range(1,self.topology.size-2):
            self.A[i+1] = np.hstack(np.tanh(self.W[i+1]@self.A[i] + self.b[i+1]))
        n = self.topology.size-1
        #self.A[n] = np.hstack(self.sig(self.W[n]@self.A[n-1] + self.b[n]))
        self.A[n] = np.hstack(self.W[n]@self.A[n-1] + self.b[n])
        return self.A[-1]
    
    def back_propagation(self):
        #self.d[-1] = self.sig_derivate(self.A[-1])
        self.d[-1] = self.A[-1]/self.A[-1]
        self.dFdW[-1] = self.d[-1]*self.A[-2]
        self.dFdb[-1] = self.d[-1]
        self.d[-2] = self.d[-1]*self.W[-1]*self.tanh_derivate(self.A[-2])
        
        for i in range(self.topology.size-2):
            n = self.topology.size-2-i #The indice of the W that we are going to evaluate
            self.dFdW[n] = np.outer(self.d[n],self.A[n-1])
            self.dFdb[n] = self.d[n]
            self.d[n-1]  = (self.W[n].T@self.d[n])*self.tanh_derivate(self.A[n-1])
        
        if(self.topology[0] == 1):
            self.dFdW[1] = np.hstack(self.dFdW[1])
   
        
    
    def training_test_sen(self,N,X,T,e):   
        for i in range(N):
            y = self.feed_forward(X[i])
            self.back_propagation()
            for j in range(1,self.topology.size):
                self.W[j] = self.W[j] - e*self.dFdW[j] * (y - T[i])
                self.b[j] = self.b[j] - e*self.dFdb[j] * (y - T[i])

    def training_ADAM (self, N, X, T):
        '''
        This is a function that will train our NN with ADAM's algorithm
        '''
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        alpha = 1e-3
        
        V_w = copy.deepcopy(self.dFdW)
        M_w = copy.deepcopy(self.dFdW)
        V_b = copy.deepcopy(self.dFdb)
        M_b = copy.deepcopy(self.dFdb)
        
        for i in range(N):
            y = self.feed_forward(X[i])
            self.back_propagation()
            for j in range(1, self.topology.size):
                M_w[j] = beta_1 * M_w[j] + (1-beta_1) * (self.dFdW[j]* (y - T[i]))
                M_b[j] = beta_1 * M_b[j] + (1-beta_1) * (self.dFdb[j]* (y - T[i]))
                
                V_w[j] = beta_2 * V_w[j] + (1-beta_2) * (self.dFdW[j] * (y - T[i]))**2
                V_b[j] = beta_2 * V_b[j] + (1-beta_2) * (self.dFdb[j] * (y - T[i]))**2
                
                m_chapeu_w = M_w[j]/(1 - beta_1**(i+1))
                m_chapeu_b = M_b[j]/(1 - beta_1**(i+1))
                
                v_chapeu_w = V_w[j]/(1 - beta_2**(i+1))
                v_chapeu_b = V_b[j]/(1 - beta_2**(i+1))
                
                self.W[j] = self.W[j] - alpha*m_chapeu_w/(np.sqrt(v_chapeu_w + epsilon))
                self.b[j] = self.b[j] - alpha*m_chapeu_b/(np.sqrt(v_chapeu_b + epsilon))



np.random.seed(0)

N = 50000
X = npr.uniform(0,1.3,N) 
T = np.tan(X)

top = np.array([1,5,5,1])
nn = NN(top)

nn.training_ADAM(N,X,T) 
#nn.training_test_sen(N,X,T,0.1)

M = 20
Y = np.linspace(0,1.3,M)
test = np.zeros(M)
for i in range(M):
    test[i] = nn.feed_forward(Y[i])

plt.scatter(Y,test)
plt.plot(Y,np.tan(Y))
plt.title("Using Adam")
plt.show()

np.random.seed(0)
X = npr.uniform(0,1.3,N) 
T = np.tan(X)

nn2 = NN(top)
nn2.training_test_sen(N,X,T,0.1)
M = 20
Y = np.linspace(0,1.3,M)
test = np.zeros(M)
for i in range(M):
    test[i] = nn2.feed_forward(Y[i])

plt.scatter(Y,test)
plt.plot(Y,np.tan(Y))
plt.title("using SGD")

