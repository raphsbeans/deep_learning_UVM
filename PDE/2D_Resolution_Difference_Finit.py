#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:56:40 2018

@author: raphael.guimaraes-feijao
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#Payoff function for an array os stock values
def g(S):
    return np.maximum(S-90,0) - np.maximum(S-110,0)

T =    1       #maturity
N =    100    #parameter of discretization
dt =   T/N     #time step
S0 =   100     #Initial price of the underlying asset
sinf = 0.1   #Volatility interval
ssup = 0.2
Smax = 200
dx = Smax/(2*N+1)
pho = -0.5
gamma = np.zeros((2,2))

W = np.zeros((N,2*N+1))
J = np.linspace(0,Smax, 2*N+1) 
W[-1,:] = g(J)
Wjm, Wjp = np.zeros(2*N+1), np.zeros(2*N+1)


def Phi (pho, sig1, sig2, gamma, X1, X2):
    '''
    This function will generate Phi, 
    gamma a 2-dimensional array with 2 elements in each dim
    '''
    first_term = 0.5 * (sig1)**2 * X1**2 * gamma[0,0]
    second_term = 0.5 * (sig2)**2 * X2**2 * gamma[1,1]
    third_term = pho * sig1 * sig2 * X1 * X2 * gamma[0,1]
    
    return first_term + second_term + third_term

def Sig1_given_Sig2 ():
    '''
    In this part we will freeze sig2 and calculate sig1
    '''
    
    