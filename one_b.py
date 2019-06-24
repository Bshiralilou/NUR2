#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from one_a import rand
import scipy.stats as stats

def normal_dist(mean, sigma, num):
    z1= []
    z2= []
    i= 0
    while i<num:
        # uniformly distributed values between [0,1)
        u1= rand()
        u2= rand()
        # generating random variables with standard normal distribution
        y1= np.sqrt(-2. * np.log(u1)) * np.cos(2.*np.pi*u2)
        y2= np.sqrt(-2. * np.log(u1)) * np.sin(2.*np.pi*u2)
        #(this is to check, because sometimes the computation gives positive values for log which it should not!)
        if np.log(u1) <= 0 and np.log(u2) <= 0:
            z1.append( y1 * sigma + mean)
            z2.append( y2 * sigma + mean)
            i= i+1

    return z1, z2

#The histogram
sigma= 2.4
mean= 3.
z1, z2 = normal_dist(mean, sigma, 1000)
bins1 = np.linspace(min(z1), max(z1), 30)
bins2 = np.linspace(min(z2), max(z2), 30)
#--------------- z1
#histogram
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.xlim(mean - 5.*sigma, mean + 5.*sigma)  #5 sigma interval
plt.hist(z1, bins= bins1, density= 'normed')
plt.xlabel('random number z1')
plt.ylabel('distribution')
#the distribution from scipy package
x1 = np.linspace(mean - 5.*sigma, mean + 5.*sigma, 100)
plt.plot(x1, stats.norm.pdf(x1, mean, sigma))
#the sigma lines
plt.axvline(x=mean +1*sigma, color= 'r', linestyle= 'dashed')
plt.axvline(x=mean +2*sigma, color= 'g', linestyle= 'dashed')
plt.axvline(x=mean +3*sigma, color= 'c', linestyle= 'dashed')
plt.axvline(x=mean +4*sigma, color= 'y', linestyle= 'dashed')

plt.axvline(x=mean -1*sigma, color= 'r', linestyle= 'dashed')
plt.axvline(x=mean -2*sigma, color= 'g', linestyle= 'dashed')
plt.axvline(x=mean -3*sigma, color= 'c', linestyle= 'dashed')
plt.axvline(x=mean -4*sigma, color= 'y', linestyle= 'dashed')

#---------------- z2
#histogram
plt.subplot(1,2,2)
plt.xlim(mean - 5.*sigma, mean + 5.*sigma)  #5 sigma interval
plt.hist(z2, bins= bins2, density= 'normed')
plt.xlabel('random number z2')
plt.ylabel('distribution')
#the distribution from scipy package
x2 = np.linspace(mean - 5.*sigma, mean + 5.*sigma, 100)
plt.plot(x2, stats.norm.pdf(x2, mean, sigma))
#the sigma lines
plt.axvline(x=mean +1*sigma, color= 'r', linestyle= 'dashed')
plt.axvline(x=mean +2*sigma, color= 'g', linestyle= 'dashed')
plt.axvline(x=mean +3*sigma, color= 'c', linestyle= 'dashed')
plt.axvline(x=mean +4*sigma, color= 'y', linestyle= 'dashed')

plt.axvline(x=mean -1*sigma, color= 'r', linestyle= 'dashed')
plt.axvline(x=mean -2*sigma, color= 'g', linestyle= 'dashed')
plt.axvline(x=mean -3*sigma, color= 'c', linestyle= 'dashed')
plt.axvline(x=mean -4*sigma, color= 'y', linestyle= 'dashed')

plt.savefig('./Plots/distribution_1b.png')
#plt.show()
plt.close()
