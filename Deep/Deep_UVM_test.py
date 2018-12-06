#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:05:56 2018

@author: marcos.saraiva
"""
import numpy as np
import Deep_UVM_bidimensional as deep
import time


#Paramters of the test             
top = np.array([1,4,4,1])
N = 10
Ntraining = 100000
print('Amount of NN:'+str(N))
print('Ntraining:'+str(Ntraining))
print('Forward difference')
print('e = 10^-4')

#SGD 
print('SGD Test')
first_deep = deep.deep_bidmensional(top,N) 
start_time = time.time()
first_deep.train(Ntraining,0.1)
print("SGD time in minutes:" + str((time.time()-start_time)/60))
start_time = time.time()
print(first_deep.evaluate(65000))
print("MC time in minutes:" + str((time.time()-start_time)/60))

'''
#ADAM
print('ADAM Test')
first_deep = deep.deep(top,N) 
start_time = time.time()
first_deep.train_ADAM(Ntraining)
print("ADAM time in minutes:" + str((time.time()-start_time)/60))
start_time = time.time()
print(first_deep.evaluate(65000))
print("MC time in minutes:" + str((time.time()-start_time)/60))
'''