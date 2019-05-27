#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# growth factor calculation at redshift 50 using romberg integration
def integ(zlim = 50.):
    omega_m = 0.3
    omega_l = 0.7
    H0 = 70.
    
    # Defining the growth factor function as D(z)
    Hz = H0**(2.) * (omega_m * (1.+zlim)**(3.) + omega_l )
    D = lambda a: (2.5 * omega_m  *Hz * H0**2.) * 1./(a * H0**(3.) * (omega_m * (a)**(-3.) + omega_l ))**3.
    
    # Defining the Romberg integration
    n = 10
    lim1 = 1e-8              # Lower integration limit (very close to zero)
    lim2 = 1./(zlim + 1.)    # Upper integration limit
    h0 = (lim2-lim1)/(2.**n) # The smallest integral step
    
    F = np.tile(np.nan, 2**n + 1)
    for i, F_i in enumerate(F):
        F[i] = D(lim1 + h0*i)
        
    # Making an n*n grid for the Riemann sums
    R = np.tile( np.nan, (n,n))
    for j in range(n):
        # The first column
        if j == 0:
            for i in range(0,n):
                h = h0 * 2**(n-i)
                R[i,j] = h * np.sum(F[0:2**n :2**(n-i)]) - 0.5*h*(F[0]+F[2**n])
        # The other columns 
        else:
            for i in range(j,n):
                R[i,j] = (4.**j * R[i, j-1] - R[i-1, j-1])/(4.**j-1.)
    
    
    return R[n-1,n-1]

value = integ()
print('D(z=50) =', value)
np.savetxt('integral_4a.txt', np.array([value]))