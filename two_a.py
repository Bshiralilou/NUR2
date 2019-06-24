#!/usr/bin/env python
import numpy as np
from one_b import normal_dist
import matplotlib.pyplot as plt
import scipy.fftpack as fft

def Fplane(n, scale, size= 1024):
    sym = np.zeros( (size+1 , size+1))         #512+512+1
    antisym = np.zeros( (size+1 , size+1) )    #512+512+1
    # generating the symmetrci+ antisymmetric matrices
    for ki in np.arange(0, size+1):
        for kj in np.arange(0, ki+1):
            # building kx, ky to be in (-512, 512), from the indices ki, kj
            #the variance at each point:
            if kj != 512 or ki != 512:
            # defining random number (a+ib)
                sigma = (np.sqrt(((kj - 512 )/scale)**(2.) + ((ki - 512 )/scale)**(2.)) )**n
                a, b = normal_dist(0., sigma, 1)

            else:
                sigma = (np.sqrt(((5e-6)/scale)**(2.) + ((5e-6)/scale)**(2.)) )**(0.5*n)
                a, b = normal_dist(0., sigma, 1)

            #filling the  symmetric matrix:
            sym[ki, kj] = a[0]
            sym[kj, ki] = a[0]
            #filling the anti-symmetric matrix:
            if ki == kj:
                antisym[ki, kj] = 0.
            else:
                antisym[ki, kj] = b[0]
                antisym[kj, ki] = -b[0]

    return sym + 1j * antisym

#------Inverse fourier transformation

# n = -1
fourier = Fplane(-1., 10, size=1024)
real = fft.ifft2(fourier)
plt.imshow( np.abs(real) )
plt.colorbar()
plt.xlabel('pixel number')
plt.ylabel('pixel number')
plt.savefig('./Plots/fourier_2a_1.png')
#plt.show()
plt.close()

# n = -2
fourier = Fplane(-2., 10, size=1024)
real = fft.ifft2(fourier)
plt.imshow( np.abs(real) )
plt.colorbar()
plt.xlabel('pixel number')
plt.ylabel('pixel number')
plt.savefig('./Plots/fourier_2a_2.png')
#plt.show()
plt.close()

# n = -3
fourier = Fplane(-3., 10, size=1024)
real = fft.ifft2(fourier)
plt.imshow( np.abs(real) )
plt.colorbar()
plt.xlabel('pixel number')
plt.ylabel('pixel number')
plt.savefig('./Plots/fourier_2a_3.png')
#plt.show()
plt.close()
