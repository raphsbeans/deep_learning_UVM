#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:05:56 2018

@author: marcos.saraiva
"""
import numpy as np
import Deep_UVM as deep
import time

start_time = time.time()
                    
top = np.array([1,4,4,1])
N = 12
first_deep = deep.deep(top,N) 

print(first_deep.evaluate_mean(100000))

''' 
first_deep.train(10000,0.1)

print("SGD time in minutes:" + str((time.time()-start_time)/60))

start_time = time.time()
print(first_deep.evaluate(65000))
print("MC time in minutes:" + str((time.time()-start_time)/60))
'''