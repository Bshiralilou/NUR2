#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def rand( a= 2811536238 , b= 2**32):
    # XOR shift
    rand.current ^= rand.current << 21
    rand.current ^= rand.current >> 35
    rand.current ^= rand.current << 4
    # MWC
    rand.current = a * (rand.current & (b - 1)) + (rand.current >> 32)

    return rand.current/((a*b - 2))


rand.current = 987654321   # Seed
print('the seed value of the random generator is:', rand.current)

#----- generating the scatter plots

rands_scatt=[]
for i in range(1000):
    rands_scatt.append( rand() )

plt.scatter( rands_scatt[0:998], rands_scatt[1:999], marker= '.' )
plt.xlabel(r'$x_i$')
plt.ylabel(r'$x_{i+1}$')
plt.savefig('./Plots/scatt_1a.png')
#plt.show()
plt.close()

plt.bar(np.arange(0,1000), np.array(rands_scatt))
plt.ylabel('random value')
plt.xlabel('index of the number')
plt.savefig('./Plots/bar_1a.png')
#plt.show()
plt.close()
#----- generating the histogram plot

rands_hist=[]
for i in range(1000000):
    rands_hist.append (rand() )

bins = np.arange(0,1.05,0.05)
plt.hist( rands_hist, bins )
plt.xlabel(r'$x_{random}$')
plt.ylabel("frequency")
plt.savefig('./Plots/hist_1a.png')
#plt.show()
plt.close()

