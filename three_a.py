#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

#----- the parameter values for adaptive step size, Runge Kutta method
c = np.array([1, 0.2, 0.3, 0.8, 8/9, 1, 1])
b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

a = np.tile(np.nan, (7,6))
a[1,0] = 0.2
a[2,0] = 3/40
a[2,1] = 9/40
a[3,0] = 44/45
a[3,1] = -56/15
a[3,2] = 32/9
a[4,0] = 19372/6561
a[4,1] = -25360/2187
a[4,2] = 64448/6561
a[4,3] = -212/729
a[5,0] = 9017/3168
a[5,1] = -355/33
a[5,2] = 46732/5247
a[5,3] = 49/176
a[5,4] = -5103/18656
a[6,0] = 35/384
a[6,1] = 0
a[6,2] = 500/1113
a[6,3] = 125/192
a[6,4] = -2187/6784
a[6,5] = 11/84

#----- the seccond order ODE
def f(t,e,d):
    return (2/3)*d/t**2 - (4/3)*e/t
#----- Numerical integration
def runge_kutta(D0, E0, t_i, t_f, h0, N_max=5000 ):

    D = np.array([D0])   # growth factor
    E = np.array([E0])   # derivative of the growth factor
    T = np.array([t_i])  # time steps

	#values of the parameter k for D and E equations.
    kE = np.tile(np.nan , (6))
    kD = np.tile(np.nan , (6))

    h = h0
    N = 0   #number of steps without change in step size
    M = 0   #number of steps with change in step size

    while (T[-1] < t_f) and ((N+M) < N_max) :

        kE[0] = h* f(T[-1], E[-1], D[-1])
        kD[0] = h* E[-1]

	# 12 lines of updating the values of kD and kE in just two lines! :D
        for i in range(1,6):
            kE[i] = h* f(T[-1] + h*c[i] , E[-1]+ np.sum(kE[:i] *a[i,:i]) , D[-1]+ np.sum(kD[:i] *a[i,:i]) )
            kD[i] = h* (E[-1]+ np.sum(kE[:i] *a[i,:i]))

	# calculating the solution at each time step
        D_new = D[-1] + np.sum( b[:6] * kD )
        E_new = E[-1] + np.sum( b[:6] * kE )

	#error estimation
        deltaE = np.absolute( np.sum( (b[:6]-b_star[:6]) * kE ) )
        deltaD = np.absolute( np.sum( (b[:6]-b_star[:6]) * kD ) )

        atol = 0.
        rtol = 5e-3  #chosen such that the computation time be reasonable
        scaleD = atol + rtol * np.amax( np.array([D[-1], D_new]) )
        scaleE = atol + rtol * np.amax( np.array([E[-1], E_new]) )

        err = np.sqrt( 0.5*( (deltaE/scaleE)**2 + (deltaD/scaleD)**2 ) )

	#checking for change in step size
        if err <= 1.:
            N += 1
            t = T[-1] + h
            D = np.pad(D , ((0,1)), 'constant', constant_values= D_new )
            E = np.pad(E , ((0,1)), 'constant', constant_values= E_new )
            T = np.pad(T , ((0,1)), 'constant', constant_values= t )

        else:
            M += 1
            s = 0.9  #as noted in lecture note, the value of s is few percent less than one
            h *= s* err**(-0.2)

    if T[-1] < t_f:
        print('Failed to reach t_f')
        print('Last time step after',N_max ,'iterations =',T[-1] )


    print('N_max =',int(N_max))
    print('N =',N)
    print('M =',M)

    return T,D

#----- plots of the three cases
D0 = 3.
E0 = 2.
t_i = 1.
t_f = 1000.
h0 = 1
N_max = 3*(t_f - t_i)//h0

T,D = runge_kutta(D0, E0, t_i, t_f, h0, N_max )

plt.loglog(T,D)
plt.xlabel('log Time(year)')
plt.ylabel('log D')
plt.savefig('./Plots/ODE_3a_1.png')
#plt.show()
plt.close()

D0 = 10.
E0 = -10.
t_i = 1.
t_f = 1000.
h0 = 0.1
N_max = 3*(t_f - t_i)//h0

T,D = runge_kutta(D0, E0, t_i, t_f, h0, N_max )

plt.loglog(T,D)
plt.xlabel('log Time(year)')
plt.ylabel('log D')
plt.savefig('./Plots/ODE_3a_2.png')
#plt.show()
plt.close()

D0 = 5.
E0 = 0.
t_i = 1.
t_f = 1000.
h0 = 1
N_max = 3*(t_f - t_i)//h0

T,D = runge_kutta(D0, E0, t_i, t_f, h0, N_max )

plt.loglog(T,D)
plt.xlabel('log Time(year)')
plt.ylabel('log D')
plt.savefig('./Plots/ODE_3a_3.png')
#plt.show()
plt.close()

