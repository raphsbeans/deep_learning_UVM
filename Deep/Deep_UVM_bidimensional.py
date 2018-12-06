# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:23:45 2018

@author: Marcos Moreira & Raphael Feijao
"""
import copy as cp
import numpy as np
import numpy.random as npr
import NN as NN


class deep_bidmensional:
     def __init__(self,topology,N):
         #Creates network
         self.topology = topology
         self.N = N
         self.networks1 = []
         self.networks2 = []
         for i in range(N):
             self.networks1.append(NN.NN(topology)) 
             self.networks2.append(NN.NN(topology)) 
             
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
         self.pho = -0.5
         self.B1 = np.zeros(self.N)#brownian mouvement of asset 1
         self.Z = np.zeros(self.N) #Auxiliar brownian independent of 1
         self.B2 = np.zeros(self.N)#Brownian mouvement of asset 2
         
         self.payoff = 0
         
     def g(self,S1, S2):
         '''
         The payoff of this option
         '''
         return np.maximum(S1-S2,0)
         
     def feed_forward(self):
        #Brownian Calculus
        self.B1 = npr.normal(0,np.sqrt(self.T/self.N),self.N)
        self.Z = npr.normal(0,np.sqrt(self.T/self.N),self.N)
        self.B2 = self.pho*self.B1 + np.sqrt(1 - self.pho**2)*self.Z
        
        for i in range(self.N):
            #Volatility calculus
            vol1 = (self.ssup - self.sinf)*self.networks1[i].feed_forward(self.S1[i])+self.sinf
            vol2 = (self.ssup - self.sinf)*self.networks2[i].feed_forward(self.S2[i])+self.sinf
            #Feed Forward
            self.S1[i+1] = self.S1[i]*np.exp(vol1*self.B1[i]-vol1**2/2*self.T/self.N)
            self.S2[i+1] = self.S2[i]*np.exp(vol2*self.B2[i]-vol2**2/2*self.T/self.N)
            #BackProp
            self.networks1[i].back_propagation()
            self.networks2[i].back_propagation()
        #Payoff
        self.payoff = self.g(self.S1[-1], self.S2[-1])

     def evaluate(self,Nmc):
         '''
         Evaluates the stochastic control for the current neural network
         '''
         S1 = self.S1[0]
         S2 = self.S2[0]
         sum_payoff = 0
         for i in range(Nmc):
             S1 = self.S1[0]
             S2 = self.S2[0]
             
             B1 = npr.normal(0,np.sqrt(self.T/self.N),self.N)
             Z = npr.normal(0,np.sqrt(self.T/self.N),self.N)
             B2 = self.pho*B1 + np.sqrt(1 - self.pho**2)*Z
             
             for j in range(self.N):
                 vol1 = (self.ssup - self.sinf)*self.networks1[j].feed_forward(S1)+self.sinf
                 vol2 = (self.ssup - self.sinf)*self.networks2[j].feed_forward(S2)+self.sinf
                 S1 = S1*np.exp(vol1*B1[j]-vol1**2/2*self.T/self.N)
                 S2 = S2*np.exp(vol2*B2[j]-vol2**2/2*self.T/self.N)
             sum_payoff += self.g(S1, S2)
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

     def dg_dvol(self,k):
         """
         Performs the derivative of the payoff with respect to the control k 
         where k belongs to {0,1,...,self.N-1}
         """
         e = 0.0001
         #f(x+e)
         S1 = self.S1[k]
         S2 = self.S2[k]
         S1_before = self.S1[k]
         S2_before = self.S2[k]
         
         vol1 = (self.ssup - self.sinf)*self.networks1[k].evaluate(S1)+self.sinf + e
         vol1_before = (self.ssup - self.sinf)*self.networks1[k].evaluate(S1)+self.sinf
         vol2 = (self.ssup - self.sinf)*self.networks2[k].evaluate(S2)+self.sinf + e
         vol2_before = (self.ssup - self.sinf)*self.networks2[k].evaluate(S2)+self.sinf
         
         S1 = S1*np.exp(vol1*self.B1[k]-vol1**2/2*self.T/self.N)
         S1_before = S1_before*np.exp(vol1_before*self.B1[k]-vol1_before**2/2*self.T/self.N)
         S2 = S2*np.exp(vol2*self.B2[k]-vol2**2/2*self.T/self.N)
         S2_before = S2_before*np.exp(vol2_before*self.B2[k]-vol2_before**2/2*self.T/self.N)
         
         for i in range(k+1,self.N):
                vol1 = (self.ssup - self.sinf)*self.networks1[i].evaluate(S1)+self.sinf
                S1 = S1*np.exp(vol1*self.B1[i]-vol1**2/2*self.T/self.N)
                vol2 = (self.ssup - self.sinf)*self.networks2[i].evaluate(S2)+self.sinf
                S2 = S2*np.exp(vol2*self.B2[i]-vol2**2/2*self.T/self.N)
                
                vol1_before = (self.ssup - self.sinf)*self.networks1[i].evaluate(S1_before)+self.sinf
                S1_before = S1_before*np.exp(vol1_before*self.B1[i]-vol1_before**2/2*self.T/self.N)
                vol2_before = (self.ssup - self.sinf)*self.networks2[i].evaluate(S2_before)+self.sinf
                S2_before = S2_before*np.exp(vol2_before*self.B2[i]-vol2_before**2/2*self.T/self.N)
                
         Gplus1 = self.g(S1, S2_before)
         Gplus2 = self.g(S1_before, S2)
         
         return (Gplus1 - self.payoff)/(2*e), (Gplus2 - self.payoff)/(2*e)
    

     '''
     Trains the deep neural network
     Ngd : Number of iterations of the gradient descent algorithm
     alfa : learning rate
     '''
    
     def train_ADAM(self,Ngd):
         #Keeps the sum of the derivatives of the output of each network 
         #with respect to its parameters, W and b for each MC iteration
         dadW = []
         dadb = []
         
         for i in range(self.N):
             dadW.append(cp.deepcopy(self.networks[i].dFdW)) 
             dadb.append(cp.deepcopy(self.networks[i].dFdb))
         
         beta_1 = 0.9
         beta_2 = 0.999
         epsilon = 1e-8
         alpha = 1e-3       
             
         V_w = cp.deepcopy(dadW)
         M_w = cp.deepcopy(dadW)
         V_b = cp.deepcopy(dadb)
         M_b = cp.deepcopy(dadb)

         for i in range(Ngd):
            #Simulates the current process 
            self.feed_forward()
            for j in range(self.N):
                dg = self.dg_dvol(j)
                for k in range(1,self.topology.size):
                    #ADAM
                    dadW[j][k] = dg*(self.ssup - self.sinf)*self.networks[j].dFdW[k]
                    dadb[j][k] = dg*(self.ssup - self.sinf)*self.networks[j].dFdb[k]
                    
                    M_w[j][k] = beta_1 * M_w[j][k] + (1-beta_1) * (dadW[j][k])
                    M_b[j][k] = beta_1 * M_b[j][k] + (1-beta_1) * (dadb[j][k])
                    
                    V_w[j][k] = beta_2 * V_w[j][k] + (1-beta_2) * (dadW[j][k])**2
                    V_b[j][k] = beta_2 * V_b[j][k] + (1-beta_2) * (dadb[j][k])**2
                    
                    m_chapeu_w = M_w[j][k]/(1 - beta_1**(i+1))
                    m_chapeu_b = M_b[j][k]/(1 - beta_1**(i+1))
                    
                    v_chapeu_w = V_w[j][k]/(1 - beta_2**(i+1))
                    v_chapeu_b = V_b[j][k]/(1 - beta_2**(i+1))
                    
                    #Updates the network's parameters
                    e_w =  alpha*m_chapeu_w/(np.sqrt(v_chapeu_w)+ epsilon)
                    e_b =  alpha*m_chapeu_b/(np.sqrt(v_chapeu_b) + epsilon)

                    self.networks[j].W[k] += e_w
                    self.networks[j].b[k] += e_b     
        
     def train(self,Ngd,alfa):
         #Keeps the sum of the derivatives of the output of each network 
         #with respect to its parameters, W and b for each MC iteration
         dadW = []
         dadb = []
         for i in range(self.N):
             dadW.append(cp.deepcopy(self.networks1[i].dFdW)) 
             dadb.append(cp.deepcopy(self.networks1[i].dFdb))

         for i in range(Ngd):
            #Simulates the current process 
            print(i)
            self.feed_forward()
            for j in range(self.N):
                dg1, dg2 = self.dg_dvol(j)
                for k in range(1,self.topology.size):
                    dadW[j][k] = dg1*(self.ssup - self.sinf)*self.networks1[j].dFdW[k] + dg2*(self.ssup - self.sinf)*self.networks2[j].dFdW[k]
                    dadb[j][k] = dg1*(self.ssup - self.sinf)*self.networks1[j].dFdb[k] + dg2*(self.ssup - self.sinf)*self.networks2[j].dFdb[k]
            #Updates the network's parameters
            for j in range(self.N):
                for k in range(1,self.topology.size):
                    self.networks1[j].W[k] += alfa * dadW[j][k]  
                    self.networks1[j].b[k] += alfa * dadb[j][k]    
                    self.networks2[j].W[k] += alfa * dadW[j][k]  
                    self.networks2[j].b[k] += alfa * dadb[j][k]      
                                            
     

        