# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:24:13 2017

@author: gabrielmindlin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:48:18 2017

@author: gabrielmindlin
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import scipy.fftpack

#from scipy.signal import hilbert

samplerate_s, data_s = wavfile.read('BOS06_zfAB009-ba_v08_denoised.wav')
times                = np.arange(len(data_s))/float(samplerate_s)
indexes              = np.arange(len(data_s))


NU=len(data_s)
frecuencia=np.zeros(NU)

def getEnvelope(inputSignal):
# Taking the absolute value

    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append (abs (sample))

    # Peak detection

    intervalLength = 405 # change this number depending on your Signal frequency content and time scale
    outputSignal   = []

    for baseIndex in range (0, len (absoluteSignal)):
        maximum = 0
        for lookbackIndex in range (intervalLength):
            maximum = max (absoluteSignal [baseIndex - lookbackIndex], maximum)
        outputSignal.append (maximum)

    return outputSignal

amplitude_envelope=getEnvelope(data_s)


#
# Calculo de los SCI en funcion del tiempo

fu,tu,Sxx = signal.spectrogram(data_s,samplerate_s,nperseg=2*256,noverlap=256, scaling='spectrum')
#plt.xlim(0,1)
plt.ylim(0,10000)
Sxx=np.clip(Sxx,a_min=np.amax(Sxx)*0.00001,a_max=np.amax(Sxx))
plt.pcolormesh(tu,fu,np.log(Sxx),cmap=plt.get_cmap('Greys'),rasterized=True)
plt.ylabel('Frecuencia (Hz) y posicion del labio')
plt.xlabel('Time [sec]')
plt.show()

spectral_index       = []
spectral_complexity  = []
spectral_information = []

for i in range(Sxx.shape[1]):
    spec=[x[i] for x in Sxx]
    
    norma         = 0
    integral      = 0
    sci           = 0
    complexity    = 0
    information   = 0
    desequilibrio = 0
    #calculo del espectro por ventanitas normalizado
    for i in range(len(spec)):
        norma=norma+spec[i]
    spec=spec/norma
    # spec ahora tiene el espectro normalizado

    # Con el espectro normalizado en ventanitas, calculo de la complejidad
    for i in range(len(spec)):
        information=information-(1/(np.log(len(spec))))*spec[i]*np.log(spec[i])
        desequilibrio=desequilibrio+(spec[i]-1/(len(spec)))*(spec[i]-1/(len(spec)))
    complexity=information*desequilibrio
    spectral_complexity.append(complexity)
    spectral_information.append(information)
   
    # Con el espectro normalizado en ventanitas, calculo de los SCI
    for i in range(len(spec)):
        sci=sci+spec[i]*fu[i]
    spectral_index.append(sci)

plt.ylabel('Envolvente')
#plt.xlim(0,1)
maximo=max(amplitude_envelope)
plt.plot(times,amplitude_envelope)
plt.axhline(y=maximo/30.0,xmin=0,xmax=times[len(times)-1], hold=None,color="red")
plt.show()

filter=(amplitude_envelope>maximo/30.0)

#plt.xlim(0,1)
plt.ylabel('Spectral Content Index')
plt.plot(tu,spectral_index)
plt.show()

#plt.xlim(0,1)
plt.ylabel('Spectral Complexity')
plt.plot(tu,spectral_complexity)
plt.show()

#plt.xlim(0,1)
plt.ylabel('Spectral Information')
plt.plot(tu,spectral_information)
plt.show()

f = open('caracterizacion.txt', 'w')

spectral_index_pos = []
spectral_information_pos = []
spectral_complexity_pos = []

for i in range(len(spec)):
    if (amplitude_envelope[i]>maximo/30.0):
        f.write("%f %f %f \n "% (spectral_information[i],spectral_complexity[i],spectral_index[i]))
        spectral_complexity_pos.append(spectral_complexity[i])
        spectral_information_pos.append(spectral_information[i])
        spectral_index_pos.append(spectral_index[i])

f.close()


# Calculo de Fourier
# Number of samplepoints
N = NU
# sample spacing
T = 1.0 / samplerate_s

yf = scipy.fftpack.fft(data_s)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()

norma2=0
fourier_normalizado =[]
for i in range(N//2):
    norma2=norma2+np.abs(yf[i])
    fourier_normalizado.append(np.abs(yf[i]))
fourier_normalizado=fourier_normalizado/norma2

sci_global=0
complexity_global=0
information_global=0
disorder_global=0

for i in range(N//2):
    sci_global=sci_global+xf[i]*fourier_normalizado[i]
    information_global=information_global-(1/(N//2))*(fourier_normalizado[i])*np.log(fourier_normalizado[i])
    disorder_global=disorder_global+(fourier_normalizado[i]-1/(N//2))*(fourier_normalizado[i]-1/(N//2))
complexity_global=information_global*disorder_global

fig, ax = plt.subplots()
ax.plot(xf,fourier_normalizado)
plt.show()
    
print(complexity_global)
print(information_global)
print(sci_global)

fig=plt.figure()
ax=fig.add_subplot(111, projection= '3d')

ax.scatter(spectral_information_pos,spectral_complexity_pos,spectral_index_pos,color='r',marker='.')

plt.show()


n    =0
k    =8
l    =7
lolo =9
