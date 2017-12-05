# -*- coding: utf-8 -*-

"""
Created on November 2017

@author: Gonzlao Uribarri

It extracts signal envelope and plots it over the original signal


"""
import numpy as np
import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import scipy.fftpack
import seaborn as sns

# from scipy.signal import hilbert

#samplerate_s, data_s = wavfile.read('BOS06_zfAB009-ba_v08_denoised.wav')
#samplerate_s, data_s = wavfile.read('/home/usuario/Desktop/Canto_Cortado/57-751/Thamnophilus_caerulescens_07_01.wav')
samplerate_s, data_s = wavfile.read('/home/usuario/Desktop/Canto_Cortado/48-574/Leucochloris_albicollis_04_01.wav')
#samplerate_s, data_s = wavfile.read('/home/usuario/Desktop/Canto_Cortado/02-5/Crypturellus_undulatus_01_01.wav')

times                = np.arange(len(data_s))/float(samplerate_s)

interval_time = 2 # Length of the amplitude window in miliseconds
close_ret_window = 2000 # Height of the Close Returns graph in miliseconds
close_ret_epsilon = 0.025 # Close Returns proximity Criteria 
amp_cut = 0.25 # Percentage of amplitude for cut-off

def get_features(input_signal,samp_rate,interval_time):

    # Number of samples in one step
    step_length = int(np.floor(samp_rate*float(interval_time)/1000))
    # Number of steps in the input signal 
    steps = int(np.floor(len(input_signal)/float(step_length)))
    # full vector of times
    full_times = np.arange(len(input_signal))/float(samp_rate)

    # Defining vector length
    times     = np.zeros(steps)
    amplitude = np.zeros(steps)
    entropy = np.zeros(steps)

    # Getting the amplitude of each window

    for i in range(steps):
        maximum = np.amax(np.absolute(input_signal[(i*step_length):((i+1)*step_length-1)]))
        fourier = np.abs(np.fft.rfft(input_signal[(i*step_length):((i+1)*step_length-1)]))
        tot = float(np.sum(fourier))
        if tot!=0:
            fourier = np.divide(fourier,tot)
            ent = 0
            for prob in fourier:
                if prob>0:
                    ent = ent + -1.0* prob * np.log(prob)
        else:
            ent = 0
        amplitude[i] = maximum
        entropy[i] = ent
        times[i] = full_times[i*step_length]

  # filt = (amplitude>np.amax(amplitude)/45.0)
  # filtered_amplitude = filt*amplitude
  # filtered_entropy = filt*entropy

    return times, amplitude, entropy, input_signal[(i*step_length):((i+1)*step_length-1)]

def get_close_returns(amp_window,temp_window,close_ret_window,close_ret_epsilon,amp_cut):
    y_size = int(np.floor(close_ret_window/float(1000*temp_window[1]))) # temp_window esta en segundos y close_ret_windows en milisegundos
    if y_size>len(amp_window):
        close_returns = np.ones([len(amp_window),len(amp_window)-1])
    else:
        close_returns = np.ones([len(amp_window),y_size])
    for i in range(len(amp_window)):
        if amp_window[i]>amp_cut:
            for j in range(y_size):
                if i+j<=len(amp_window)-1:
                    if np.abs(amp_window[i]-amp_window[i+j])<close_ret_epsilon:
                        close_returns[i,j]=0
    return close_returns


def getEnvelope(inputSignal):
# Taking the absolute value

    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append (abs (sample))

    # Peak detection

    intervalLength = 220 # change this number depending on your Signal frequency content and time scale
    outputSignal   = []

    for baseIndex in range (0, len (absoluteSignal)):
        maximum = 0
        for lookbackIndex in range (intervalLength):
            maximum = max (absoluteSignal [baseIndex - lookbackIndex], maximum)
        outputSignal.append (maximum)

    return outputSignal

# amplitude_envelope=getEnvelope(data_s)

temp_window,amp_window,entropy,last_signal = get_features(data_s,samplerate_s,interval_time)
amp_window    = np.divide(amp_window,float(max(amp_window)))
fourier = np.abs(np.fft.rfft(data_s))
freqs = np.fft.fftfreq(data_s.size,d=times[1]) # times[1] es equivalente a 1/float(samplerate_s)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def tri_norm(x, *args):
    m1, m2, m3, s1, s2, s3, k1, k2, k3 = args
    ret = k1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
    ret += k2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
    ret += k3*scipy.stats.norm.pdf(x, loc=m3 ,scale=s3)
    return ret

sns.set()

# plt.figure(1)    
# plt.plot(times,amplitude_envelope)
# #np.asarray(amplitude_envelope)
# neg_envelope = np.multiply(-1,amplitude_envelope)
# plt.plot(times,neg_envelope)
# plt.plot(times,data_s)
# plt.show()

# plt.figure(2)
# plt.plot(times,data_s)
# plt.plot(temp_window,amp_window)
# #np.asarray(amplitude_envelope)
# neg_amp = np.multiply(-1,amp_window)
# plt.plot(temp_window,neg_amp)
# plt.plot(temp_window,entropy*10000-35000)
# plt.show()



# from sklearn import mixture
# from scipy.optimize import curve_fit
# def func(x, *params):
#     y = np.zeros_like(x)
#     for i in range(0, len(params), 3):
#         ctr = params[i]
#         amp = params[i+1]
#         wid = params[i+2]
#         y = y + amp * np.exp(-0.5*((x - ctr)/wid)**2)
#     return y

smooth_fourier = smooth(fourier,400)
smooth_fourier_norm = np.divide(np.asarray(smooth_fourier),np.max(smooth_fourier))

# #guess = [1000, 500000, 500, 9000, 1500000, 200, 11000, 1000000, 200]
# #guess = [1000, 5000000, 1000, 5000, 5000000, 1000, 15000, 5000000, 1000]
# guess = [10000, 1, 1000]
# popt, pcov = curve_fit(func, range(len(smooth_fourier_norm)),smooth_fourier_norm, p0=guess)
# fit = func(range(len(smooth_fourier_norm)), *popt)

# print popt


# #plt.plot(np.divide(fourier)
fig1, (ax1, ax2) = plt.subplots(2)
ax2.plot(freqs[0:(len(smooth_fourier)-1)], smooth_fourier_norm[0:-1])
ax2.set(xlabel='Freqs (hz)',ylabel='Power')
ax2.set(xlim=[0, 10000])
data_norm = np.divide(data_s,float(max(data_s)))
ax1.plot(times,data_norm)
ax1.plot(temp_window,amp_window)
neg_amp = np.multiply(-1,amp_window)
ax1.plot(temp_window,neg_amp)
ax1.set(xlabel='Time (s)',ylabel='Preassure (a.u.)')
ax1.set(ylim=[-1, 1])
#axarr[1].xlabel('Times (s)')
#axarr[1].ylabel('Power')
plt.draw()

smooth_amp_window = smooth(amp_window,26)
smooth_amp_window = np.divide(smooth_amp_window,float(max(smooth_amp_window)))
# close_returns = get_close_returns(smooth_amp_window,temp_window,close_ret_window,close_ret_epsilon,amp_cut)
# colapsed_returns = smooth(np.sum(close_returns*-1+1,0),20)
# Computes the mean and std of syllables duration

syllab_indexs = np.nonzero(smooth_amp_window>amp_cut)[0]

limits = [syllab_indexs[0]]
for n in range(len(syllab_indexs)-1):
    if (syllab_indexs[n+1]-syllab_indexs[n])>1:
        limits.append(syllab_indexs[n])
        limits.append(syllab_indexs[n+1])
limits.append(syllab_indexs[-1])

syllab_width_list = []

init_index_list = []
finish_index_list = []

for n in range(0,len(limits),2):
    # This was tricky, relating width with max position
    # inter_max_index = limits[n]+int((limits[n+1]-limits[n])/2)
    if (limits[n]!=limits[n+1]):
        ini = temp_window[limits[n]]
        finish = temp_window[limits[n+1]]
        init_index_list.append(ini)
        finish_index_list.append(finish)
        syllab_width_list.append(finish-ini)

syllab_width_array = np.asarray(syllab_width_list)


# fig2 = plt.figure(2)
# gs = grd.GridSpec(2, 2, height_ratios=[1,3], width_ratios=[6,1], wspace=0.1)

# ax = plt.subplot(gs[2])
# p = ax.imshow(close_returns.T,aspect='auto',origin='lower',extent=[temp_window[0],temp_window[-1],0,temp_window[close_returns.shape[1]]])
# plt.xlabel('Time (seconds)')
# plt.ylabel('Time Delay (seconds)')

# ax2 = plt.subplot(gs[0])
# x2 = plt.plot(temp_window,smooth_amp_window)
# x22 = plt.axhline(y=amp_cut,xmin=0,xmax=temp_window[-1], hold=None,color="red")
# x222 = plt.scatter(init_index_list,amp_cut*np.ones(len(init_index_list)),color="green")
# x2222 = plt.scatter(finish_index_list,amp_cut*np.ones(len(finish_index_list)),color="orange")
# axes = plt.gca()
# axes.set_xlim(temp_window[0],temp_window[-1])
# plt.ylabel('Amplitude')

# distrib = plt.subplot(gs[3])
# distrib.spines['right'].set_visible(False)
# distrib.spines['top'].set_visible(False)
# dis = plt.plot(colapsed_returns[20:], temp_window[20:close_returns.shape[1]])
# #axes.set_xlim(0,temp_window[close_returns.shape[1]])
# axes2 = plt.gca()
# axes2.set_ylim(0,temp_window[close_returns.shape[1]])

# #distrib.xaxis.tick_top()
# plt.xlabel('Smooth Counts')
# plt.draw()



ax2 = plt.figure(1)
x2 = plt.plot(temp_window,smooth_amp_window)
x22 = plt.axhline(y=amp_cut,xmin=0,xmax=temp_window[-1], hold=None,color="red")
x222 = plt.scatter(init_index_list,amp_cut*np.ones(len(init_index_list)),color="green")
x2222 = plt.scatter(finish_index_list,amp_cut*np.ones(len(finish_index_list)),color="orange")
axes = plt.gca()
axes.set_xlim(temp_window[0],temp_window[-1])
plt.ylabel('Amplitude')
plt.draw()


plt.show()

# plt.plot(temp_window,smooth(np.gradient(smooth_amp_window),20))
# plt.show()