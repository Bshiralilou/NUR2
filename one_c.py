#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from one_b import normal_dist

#----- defining CDF of normal distribution
def normalCDF(x, mean, sigma):
    return (1.0 + math.erf( (x - mean) / (np.sqrt(2.0) * sigma) )) * 0.5


#----- defining the sort function ( H 2.g from handin 1)
def partition( array, low, high):
    i = ( low - 1 )           # index of smaller element
    pivot = array[high]       # pivot

    for j in range(low , high):
        # If current element is smaller than or
        # equal to pivot
        if array[j] <= pivot:

            # increment index of smaller element
            i = i+1
            array[i], array[j] = array[j], array[i]

    array[i+1], array[high] = array[high], array[i+1]
    return i+1


def quickSort(arr, low, high):
    if low < high:
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr,low,high)

        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi-1)
        quickSort(arr, pi+1, high)
    return

#----- the KS cumulative distribution function (from numerical recepies book):
def ksCDF(x):
    A = np.exp(-(1./8.)*(np.pi/x)**2.)
    B = np.exp(-2.*x**2.)

    if x < 1.18:
        pks = (np.sqrt(2.*np.pi)/x) * ( A + A**9. + A**25. )
    else:
        pks = 1. - 2.*( B - B**4. + B**9. )

    return pks

#----- KS test
def KS_test(data, N, D = 1e-3):
    # we assume here that the data is a sorted array
    fo= 0.              # initial value for dataCDF

    for i in range(N):
        fn = (i+1.)/N   #CDF of datapoints
        ff = normalCDF(data[i], mean=0., sigma=1.)
        dt = max(abs(fo-ff) , abs(fn-ff))  #Maximum distance between each step and its previous one
        if (dt > D):
            D = dt
        fo = fn
    #computing the p-value
    prob = ksCDF( (np.sqrt(N) + 0.12 + (0.11/np.sqrt(N)) ) *D )

    return prob

#-----------------------
num = np.tile(np.nan,(41))      #number of randoms with 0.1 dex spacing
prob = np.zeros(num.size)
prob_scipy = np.zeros(num.size)

for i in range(41):
    num[i] = 10**(1+0.1*i)
    # generating N = floot(num[i]) data points from the distribution:
    N = int(math.floor(num[i]))             #num[i] is not generaly an int value
    data1, data2 = normal_dist(0., 1., N)   # we will only use data1 sample
    quickSort (data1, 0, N-1)               #sorting the datapoints

    # ks test using the written function
    prob[i] = KS_test(data1, i)

    # ks test using scipy
    D, p_value = stats.kstest(data1, 'norm')
    prob_scipy[i] = p_value

#the p-value plot
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(num , prob)
plt.xlabel('random number')
plt.ylabel('p-value')
#comparison with scipy KS test
plt.subplot(1,2,2)
plt.plot(num , prob, label='coded ks test')
plt.plot(num , prob_scipy, label='scipy ks test')
plt.xlabel('random number')
plt.ylabel('p-value')

plt.savefig('./Plots/KS_1c.png')
#plt.show()
plt.close()

