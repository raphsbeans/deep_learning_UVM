# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:23:45 2018

@author: Marcos Moreira & Raphael Feijao
"""
import copy as cp
import numpy as np
import numpy.random as npr
import NN as NN


class deep:
     def __init__(self,topology,N):
         #Creates network
         self.topology = topology
         self.N = N
         self.networks = []
         for i in range(N):
             self.networks.append(NN.NN(topology)) 
         #Parameters of the UVM model
         self.S = np.zeros(self.N+1)
         self.S[0] = 100
         self.sinf = 0.1
         self.ssup = 0.2
         self.T = 1
         self.B = np.zeros(self.N)#brownian mouvement
         self.payoff = 0
         
     #payoff
     def g(self,S):
         return np.maximum(S-100,0)
         
     def feed_forward(self):
        self.B = npr.normal(0,np.sqrt(self.T/self.N),self.N)
        for i in range(self.N):
            vol = (self.ssup - self.sinf)*self.networks[i].feed_forward(self.S[i])+self.sinf
            #self.S[i+1] = self.S[i] + self.S[i]*vol*self.B[i]
            self.S[i+1] = self.S[i]*np.exp(vol*self.B[i]-vol**2/2*self.T/self.N)
            self.networks[i].back_propagation()
        self.payoff = self.g(self.S[-1])
     '''
     Evaluates the stochastic control for the current neural network
     '''
     def evaluate(self,Nmc):
         S = self.S[0]
         sum_payoff = 0
         for i in range(Nmc):
             S = self.S[0]
             B = npr.normal(0,np.sqrt(self.T/self.N),self.N)
             for j in range(self.N):
                 vol = (self.ssup - self.sinf)*self.networks[j].feed_forward(S)+self.sinf
                 #S = S + S*vol*B[j]
                 S = S*np.exp(vol*B[j]-vol**2/2*self.T/self.N)
             sum_payoff += self.g(S)
         payoff = sum_payoff/Nmc
         return payoff   

     
     def evaluate_mean(self,Nmc):
         mean_vol = (self.ssup+self.sinf)/2
         sum_payoff = 0
         for i in range(Nmc):
             S = self.S[0]
             B = npr.normal(0,np.sqrt(self.T/self.N),self.N)
             for j in range(self.N):
                 S = S*np.exp(mean_vol*B[j]-mean_vol**2/2*self.T/self.N)
             sum_payoff += self.g(S)
         payoff = sum_payoff/Nmc
         return payoff
     """
     Performs the derivative of the payoff with respect to the control k 
     where k belongs to {0,1,...,self.N-1}
     """
     def dg_dvol(self,k):
         e = 0.0001
         #f(x+e)
         S = self.S[k]
         vol = (self.ssup - self.sinf)*self.networks[k].evaluate(S)+self.sinf + e
         #S = S + S*vol*self.B[k]
         S = S*np.exp(vol*self.B[k]-vol**2/2*self.T/self.N)
         for i in range(k+1,self.N):
                vol = (self.ssup - self.sinf)*self.networks[i].evaluate(S)+self.sinf
                #S = S + S*vol*self.B[i]
                S = S*np.exp(vol*self.B[i]-vol**2/2*self.T/self.N)
         Gplus = self.g(S)
         
         return (Gplus - self.payoff)/e
    
     '''
     Trains the deep neural network
     Ngd : Number of iterations of the gradient descent algorithm
     alfa : learning rate
     '''
    
     def train(self,Ngd,alfa):
         #Keeps the sum of the derivatives of the output of each network 
         #with respect to its parameters, W and b for each MC iteration
         dadW = []
         dadb = []
         for i in range(self.N):
             dadW.append(cp.deepcopy(self.networks[i].dFdW)) 
             dadb.append(cp.deepcopy(self.networks[i].dFdb))

         for i in range(Ngd):
            print(i)
            #Simulates the current process 
            self.feed_forward()
            for j in range(self.N):
                dg = self.dg_dvol(j)
                for k in range(1,self.topology.size):
                    dadW[j][k] = dg*(self.ssup - self.sinf)*self.networks[j].dFdW[k]
                    dadb[j][k] = dg*(self.ssup - self.sinf)*self.networks[j].dFdb[k]
            #Updates the network's parameters
            for j in range(self.N):
                for k in range(1,self.topology.size):
                    self.networks[j].W[k] += alfa * dadW[j][k]  
                    self.networks[j].b[k] += alfa * dadb[j][k]     
                    
    
     

if __name__ == "__main__":           
    import time      
    top = np.array([1,4,4,4,4,1])
    N = 12
    first_deep = deep(top,N)  
    start_time = time.time()
    first_deep.train(10000,0.1)
    print ("Tempo de treino " + repr(time.time() - start_time ))
    
    start_time = time.time()
    print(first_deep.evaluate(int(2**16)))
    
    print("tempo monte carlo:" + str((time.time()-start_time)))
        