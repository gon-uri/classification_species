# -*- coding: utf-8 -*-
"""
Created on march 2017

@author: Gabo Mindlin

Integrator with rk4, and tube with delays

it creates wav


"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io.wavfile import write
import random


global alpha
global feedback 
global estimulo
global destimulodt
    
# --------------------
# Parameters setting
# --------------------
gamma = 24000
uoch  = (350/5.)*100000000
uolb  = 0.0001
uolg  = 1/20.
rb    = 0.5*10000000
rdis  = 24*10000

samp_rate = 882000
beta      = -0.15
dt        = 1/float(samp_rate)
t0        = 0
tf        = 0.5
L         = 0.03
t         = 0
fsamp     = 1/dt

v = np.zeros(5)
v[0] = 0.01
v[1] = 0.001
v[2] = 0.001
v[3] = 0.0001
v[4] = 0.0001

# --------------------
# Function definitions
# --------------------
def ecuaciones(v, dv):
    x,y,i1,i2,i3 = v
    dv[0] = y
    dv[1] = gamma*gamma*alpha+beta*gamma*gamma*x-gamma*gamma*x*x*x-gamma*x*x*y+gamma*gamma*x*x-gamma*x*y
    dv[2] = i2
    dv[3] = -uolg*uoch*i1-(rdis*uolb+rdis*uolg)*i2+(uolg*uoch-rdis*rb*uolg*uolb)*i3+uolg*destimulodt+rdis*uolg*uolb*estimulo
    dv[4] = -(uolb/uolg)*i2-rb*uolb*i3+uolb*estimulo
    return dv
    
def s(t):
    sig = 1/(1+np.exp(-(t-0.05)/0.005))-1/(1+np.exp(-(t-0.15)/0.005))+1/(1+np.exp(-(t-0.2)/0.005))-1/(1+np.exp(-(t-0.3)/0.005))+1/(1+np.exp(-(t-0.35)/0.005))-1/(1+np.exp(-(t-0.45)/0.005))
    return sig
    
def rk4(dv,v,n,t,dt):
    v1  = np.zeros(n)
    k1  = np.zeros(n)
    k2  = np.zeros(n)
    k3  = np.zeros(n)
    k4  = np.zeros(n)
    dt2 = dt/2.0
    dt6 = dt/6.0
    v1  = v
    k1  = dv(v1, k1)
    v1  = v+dt2*k1
    k2  = dv(v1, k2) 
    v1  = v+dt2*k2
    k3  = dv(v1, k3)
    v1  = v+dt*k3
    k4  = dv(v1, k4)
    v   = v+dt6*(2.0*(k2+k3)+k1+k4)
    return v

    # v1 = []
    # k1 = []
    # k2 = []
    # k3 = []
    # k4 = []
    # for x in range(0, n):
    #     v1.append(x)
    #     k1.append(x)
    #     k2.append(x)
    #     k3.append(x)
    #     k4.append(x)

    # dt2 = dt/2.0
    # dt6 = dt/6.0
    # for x in range(0, n):
    #     v1[x] = v[x]
    # dv(v1, k1)
    # for x in range(0, n):
    #     v1[x] = v[x]+dt2*k1[x]
    # dv(v1, k2)     
    # for x in range(0, n):
    #     v1[x] = v[x]+dt2*k2[x]
    # dv(v1, k3)
    # for x in range(0, n):
    #     v1[x] = v[x]+dt*k3[x]
    # dv(v1, k4)
    # for x in range(0, n):
    #     v1[x] = v[x]+dt*k4[x] --> Creo que esta de mas     
    # for x in range(0, n):
    #     v[x] = v[x]+dt6*(2.0*(k2[x]+k3[x]+k1[x]+k4[x])) --> Creo que aca hay que multiplicar solo k2 y k3
    # return v


n = 5 #Cantidad de variables   

# --------------------
# Main Program
# --------------------
vec_size   = int(tf//dt)
x          = np.zeros(vec_size)
y          = np.zeros(vec_size)
tiempo     = np.zeros(vec_size)
sonido     = np.zeros(vec_size)
amplitud   = np.zeros(vec_size)
forzado    = np.zeros(vec_size)
dforzadodt = np.zeros(vec_size)

# x          = []
# y          = []
# tiempo     = []
# sonido     = []
# amplitud   = []
# forzado    = []
# dforzadodt = []

cont     = 0
N        = int((L/(350*dt))//1)
fil      = np.zeros(N)
back     = np.zeros(N)
feedback = 0

while t<tf:
    alpha       = 0.04-0.16*s(t)*(1+random.normalvariate(0,0.05))
    estimulo    = fil[N-1]
    destimulodt = (fil[N-1]-fil[N-2])/dt
    v           = rk4(ecuaciones,v,n,t,dt)
    fil[0]      = v[1]+back[N-1]
    back[0]     = -0.97*fil[N-1]
    fil[1:]     = fil[:-1]
    back[1:]    = back[:-1]
    feedback    = back[N-1]
    t           += dt
    # x.append(cont)  #ACÁ ARMO LOS ARREGLOS DE X Y Z CON LOS RESULTADOS QUE VA LARGANDO "V"
    # y.append(cont)
    # tiempo.append(cont)
    # sonido.append(cont)
    # amplitud.append(cont)
    # forzado.append(cont)
    # dforzadodt.append(cont)
    x[cont]          = v[0]
    y[cont]          = v[1]
    tiempo[cont]     = t
    # sonido[cont]   =back[0]
    sonido[cont]     = v[4]
    amplitud[cont]   = s(t)*(1+random.normalvariate(0,0.05))
    forzado[cont]    = estimulo
    dforzadodt[cont] = destimulodt
    cont             = cont+1


# --------------------
# Figure Generation
# ----------------
sns.set()

plt.figure(1)    
plt.plot(tiempo,amplitud)
plt.xlim([0,0.5])
plt.xlabel(r'$t\ (\mathrm{sec})$')
plt.ylabel(r'$x\ (\mathrm{arb. units})$')

plt.figure(2)
plt.plot(tiempo,sonido)
plt.xlim([0,0.5])
plt.xlabel(r'$t\ (\mathrm{sec})$')
plt.ylabel(r'$x\ (\mathrm{arb. units})$')

plt.figure(3)
F    = np.fft.fft(sonido)
freq = np.fft.fftfreq(len(x),1/fsamp)
spec = 2/float(len(x))*np.abs(F[1:5000])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot(freq[1:5000],spec,'k')
plt.xlabel('Frequency in Hz')
plt.ylabel('Power')


plt.show()

    
#fu,tu,Sxx = signal.spectrogram(sonido,fsamp,nperseg=4*1024,noverlap=4*512, scaling='spectrum')
#fu,tu,Sxx = signal.spectrogram(sonido,fsamp)
#plt.pcolormesh(tu, fu, Sxx)
#plt.ylim(0,10000)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

scaled = np.int16(32767*(sonido/(np.max(np.abs(sonido)))))
write('test.wav', samp_rate, scaled)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    