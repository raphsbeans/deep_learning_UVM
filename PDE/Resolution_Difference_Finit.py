# -*- coding: utf-8 -*- 
"""
Created on Tue Nov 20 22:56:33 2018

@author: raphaelfeijao
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

W = np.zeros((N,2*N+1))
J = np.linspace(0,Smax, 2*N+1) 
W[-1,:] = g(J)
Wjm, Wjp = np.zeros(2*N+1), np.zeros(2*N+1)

for i in range(2,N+1):
    n = N-i
    Wj = W[n+1,:]
    
    #Wj-1
    Wjm[1:2*N+1] = Wj[0:2*N]
    Wjm[0] = 0
    
    #Wj+1
    Wjp[0:2*N] = Wj[1:2*N+1]
    Wjp[-1] = 0
    
    #Finding gama 
    Delta_xx = (Wjp + Wjm - 2*Wj)/dx**2
    Gama = ssup * (Delta_xx >= 0) + sinf * (Delta_xx < 0)
    Rj = 0.5 * Gama**2 * J**2 * dt
    
    #Creating the matrix
    Diag_Princ = 1 + 2*Rj
    Diag_Secon_sup = -Rj[0:2*N]
    Diag_Secon_inf = -Rj[1:2*N+1]
    A = np.diag(Diag_Princ) + np.diag(Diag_Secon_sup,1) + np.diag(Diag_Secon_inf,-1)
    
    #Teste prof
    b = np.zeros_like(J)
    b[0] = -Rj[0]*0
    b[-1] = -Rj[-1]*W[-1,-1]
    A_inverse = np.linalg.inv(A)
        
    W[n,:] = A_inverse@(W[n+1,:]-b)
    
Price = interpolate.interp1d(J, W[0,:])
print("Price by the BSB equation: " + str(Price(S0)))

plt.figure()
plt.plot(J, W[0,:], label = "At time 0")
plt.plot(J, W[-1,:], label = "At maturity")
plt.title("Option pricing")
plt.xlabel("S")
plt.ylabel("Price")
plt.legend(loc="best")
plt.show()