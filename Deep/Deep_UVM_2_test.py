#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:18:44 2018

@author: marcos.saraiva
"""
import numpy as np
import time
import Deep_UVM_2 as D2

#Parameters of our deep neural network
Nnn = 12
top = np.array([2,4,4,4,4,2])
deep2 = D2.deep(top,Nnn)

#Parameters of the test

Ngd = 100000
Nmc = 65000
print('Ngd:' + str(Ngd))
print('Nmc:' + str(Nmc))

#SGD
start_time = time.time()

deep2.train(Ngd,0.1)
print("SGD time in minutes:" + str((time.time()-start_time)/60))
#MC
start_time = time.time()
print(deep2.evaluate(Nmc))
print("MC time in minutes:" + str((time.time()-start_time)/60))