# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:23:45 2018

@author: Marcos Moreira & Raphael Feijao
"""
import copy as cp
import numpy as np
import numpy.random as npr
import NN2 as NN

class deep:
     def __init__(self, topology, N):
         '''
         Initialise this class
         '''
         #Let's see if the test is not dumb
         for i in topology:
             if i < 2:
                 print ("You have to have a better topology!!")
                 print ("No values can be smaller than 2")
                 raise ValueError
         #Creates network
         self.topology = topology
         self.N = N
         self.networks = []
         
         for i in range(N):
             self.networks.append(NN.NN(topology,True)) 
             
         #Parameters of the UVM model
         self.S1 = np.zeros(self.N+1)
         self.S2 = np.zeros(self.N+1)
         self.S1[0] = 100
         self.S2[0] = 100
         
         self.sinf = 0.1
         self.ssup = 0.2
         self.T = 1
         
         #Brownian - The idea is to use 
         #B2 = pho*B1 - sqrt(1-pho**2)*Z
         self.pho = 0#-0.5
         self.B1 = np.zeros(self.N)#brownian mouvement of asset 1
         self.Z = np.zeros(self.N) #Auxiliar brownian independent of 1
         self.B2 = np.zeros(self.N)#Brownian mouvement of asset 2
         
         self.payoff = 0
         
     def g(self, S1, S2):
         '''
         Payoff of this option
         '''
         return np.maximum(S1-S2,0)
         
     def feed_forward(self):
        '''
        Make a feed forward in each Neural Net
        '''
        self.B1 = npr.normal(0,np.sqrt(self.T/self.N),self.N)
        self.Z = npr.normal(0,np.sqrt(self.T/self.N),self.N)
        self.B2 = self.pho*self.B1 + np.sqrt(1 - self.pho**2)*self.Z
        
        for i in range(self.N):
            S = np.array([self.S1[i], self.S2[i]])
            vol1, vol2 = (self.ssup - self.sinf)*self.networks[i].feed_forward(S)+self.sinf
            
            #Feed Forward
            self.S1[i+1] = self.S1[i]*np.exp(vol1*self.B1[i]-vol1**2/2*self.T/self.N)
            self.S2[i+1] = self.S2[i]*np.exp(vol2*self.B2[i]-vol2**2/2*self.T/self.N)
            
            self.networks[i].back_propagation()
            
        self.payoff = self.g(self.S1[-1], self.S2[-1])
     
     def evaluate(self,Nmc):
         '''
         Evaluates the stochastic control for the current neural network
         '''
         S = np.array((self.S1[0], self.S2[0]))
         sum_payoff = 0
         for i in range(Nmc):
             S = np.array((self.S1[0], self.S2[0]))
             B1 = npr.normal(0,np.sqrt(self.T/self.N),self.N)
             Z = npr.normal(0,np.sqrt(self.T/self.N),self.N)
             B2 = self.pho*B1 + np.sqrt(1 - self.pho**2)*Z
             
             for j in range(self.N):
                 vol1, vol2 = (self.ssup - self.sinf) * self.networks[j].feed_forward(S) + self.sinf
                 S[0] = S[0]*np.exp(vol1*B1[j]-vol1**2/2*self.T/self.N)
                 S[1] = S[1]*np.exp(vol2*B2[j]-vol2**2/2*self.T/self.N)
             sum_payoff += self.g(S[0], S[1])
         payoff = sum_payoff/Nmc
         return payoff   

     
     def dg_dvol(self,k):
         """
         Performs the derivative of the payoff with respect to the both volatilities at k 
         where k belongs to {0,1,...,self.N-1}
         """
         e = 0.0001   
         #f(x+e) - 1
         S = np.array((self.S1[k], self.S2[k]))
         
         vol1, vol2 = (self.ssup - self.sinf)*self.networks[k].evaluate(S)+self.sinf
         vol1 += e
         
         S[0] = S[0]*np.exp(vol1*self.B1[k]-vol1**2/2*self.T/self.N)
         S[1] = S[1]*np.exp(vol2*self.B2[k]-vol2**2/2*self.T/self.N)
         
         for i in range(k+1,self.N):
                vol1, vol2 = (self.ssup - self.sinf)*self.networks[i].evaluate(S)+self.sinf
                S[0] = S[0]*np.exp(vol1*self.B1[i]-vol1**2/2*self.T/self.N)
                S[1] = S[1]*np.exp(vol2*self.B2[i]-vol2**2/2*self.T/self.N)
         
         Gplus1 = self.g(S[0],S[1])
         
         #f(x+e) - 2
         S = np.array((self.S1[k], self.S2[k]))
         
         vol1, vol2 = (self.ssup - self.sinf)*self.networks[k].evaluate(S)+self.sinf
         vol2 += e
         
         S[0] = S[0]*np.exp(vol1*self.B1[k]-vol1**2/2*self.T/self.N)
         S[1] = S[1]*np.exp(vol2*self.B2[k]-vol2**2/2*self.T/self.N)
         
         for i in range(k+1,self.N):
                vol1, vol2 = (self.ssup - self.sinf)*self.networks[i].evaluate(S)+self.sinf
                S[0] = S[0]*np.exp(vol1*self.B1[i]-vol1**2/2*self.T/self.N)
                S[1] = S[1]*np.exp(vol2*self.B2[i]-vol2**2/2*self.T/self.N)
         
         Gplus2 = self.g(S[0],S[1])
         
         return (Gplus1 - self.payoff)/e, (Gplus2 - self.payoff)/e
    
     def train(self,Ngd,alfa):
         dadW = []
         dadb = []
         for i in range(self.N):
             dadW.append(cp.deepcopy(self.networks[i].dFdW)) 
             dadb.append(cp.deepcopy(self.networks[i].dFdb))

         for i in range(Ngd):
            #Simulates the current process 
            self.feed_forward()
            for j in range(self.N):
                dg1, dg2 = self.dg_dvol(j)
                for k in range(1,self.topology.size):
                    dadW[j][k] = dg1*(self.ssup - self.sinf)*self.networks[j].dFdW[k] + dg2*(self.ssup - self.sinf)*self.networks[j].dF2dW[k]
                    dadb[j][k] = dg1*(self.ssup - self.sinf)*self.networks[j].dFdb[k] + dg2*(self.ssup - self.sinf)*self.networks[j].dF2db[k]
            #Updates the network's parameters
            for j in range(self.N):
                for k in range(1,self.topology.size):
                    self.networks[j].W[k] += alfa * dadW[j][k]  
                    self.networks[j].b[k] += alfa * dadb[j][k]     
                                            
     

        