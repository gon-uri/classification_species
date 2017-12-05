# -*- coding: utf-8 -*-

"""
Created on November 2017

@author: Gonzlao Uribarri

It walks over the directory "bird_number", and decodes 
ALL MP3 files it contains. It calculates some features of the signal 
in intervals over time, preparing a set of feature vectors for each audio.

Calculated Features:
- Max, width and entropy of Fourier Transform
- Max, width and entropy of Close Returns

Grafica la transformada de Fourier y las distribuciones de Close Returns
para cada archivo. Pensado para graficar los disitintos individuos de una
misma especie y ver variedad.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import os
from pydub import AudioSegment
import scipy.io.wavfile


# -----------------------------------------------------------------------------
# Parameters and files
# -----------------------------------------------------------------------------

interval_time     = 2       # Length of the amplitude window (miliseconds)
close_ret_window  = 1500     # Height of the Close Returns graph (miliseconds)
close_ret_epsilon = 0.025   # Close Returns proximity Criteria 
close_ret_start   = 100     # Take cr points from this time (miliseconds)
amp_cut           = 0.15    # Percentage of amplitude for cut-off

smooth_time_amp = 0.25     # Smooth window for amplitude (miliseconds)
smooth_time_cr  = 75     # Smooth window for colapsed returns (miliseconds)
smooth_freq_fou = 100      # Smooth window for fourier transform (hertz) -300


# Bird numbers: "02-5" , "48-574" , "57-751"
bird_number = "57-751"
bird_folder = "/home/usuario/Desktop/Canto_Cortado/"+bird_number
files = [bird_folder+"/"+name for name in os.listdir(bird_folder)]
files.sort()


# -----------------------------------------------------------------------------
# Functions definition
# -----------------------------------------------------------------------------

def get_features(input_signal,samp_rate,interval_time):
    # Number of samples in one step
    step_length = int(np.floor(samp_rate*float(interval_time)/1000))
    # Number of steps in the input signal
    steps       = int(np.floor(len(input_signal)/float(step_length)))
    # Full vector of times
    full_times  = np.arange(len(input_signal))/float(samp_rate)

    # Defining vector lengths
    times     = np.zeros(steps)
    amplitude = np.zeros(steps)
    #entropy = np.zeros(steps)

    # Getting the amplitude of each window
    for i in range(steps): 
        maximum = np.amax(np.absolute(
                input_signal[(i*step_length):((i+1)*step_length-1)]
                ))
        # fourier = np.abs(np.fft.rfft(
        #         input_signal[(i*step_length):((i+1)*step_length-1)]
        #         ))
        # tot     = float(np.sum(fourier))
        # ENTROPY CALCULATION
        # if tot!=0:
        #   fourier = np.divide(fourier,tot)
        #   ent = 0
        #   for prob in fourier:
        #       if prob>0:
        #           ent = ent + -1.0* prob * np.log(prob)
        # else:
        #   ent = 0
        # entropy[i] = ent  
        amplitude[i] = maximum
        times[i]     = full_times[i*step_length]
#   AMPLITUDE FILTERING
#   filt = (amplitude>np.amax(amplitude)/45.0)
#   filtered_amplitude = filt*amplitude
#   filtered_entropy = filt*entropy
    return times, amplitude

def get_close_returns(amp_window,temp_window,close_ret_window,
                      close_ret_epsilon,amp_cut,close_ret_start):
    # temp_window in segundos and close_ret_windows in miliseconds
    y_size = int(np.floor(
            close_ret_window/float(1000*temp_window[1])
            ))
    start = int(np.floor(
            close_ret_start/float(1000*temp_window[1])
            ))
    if y_size>len(amp_window):
        close_returns = np.ones([len(amp_window),len(amp_window)-1])
    else:
        close_returns = np.ones([len(amp_window),y_size])
    for i in range(len(amp_window)):
        # Only fill if amplitud is grater than amp_cut
        if (amp_window[i]>amp_cut):
            for j in range(start,y_size):
                if i+j<=len(amp_window)-1:
                    if (np.abs(amp_window[i]-amp_window[i+j])
                            <close_ret_epsilon):
                        close_returns[i,j]=0
    return close_returns

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_entropy(vector):
    tot = float(np.sum(vector))
    if tot!=0:
        vector_norm = np.divide(vector,tot)
        ent = 0
        for prob in vector_norm:
            if prob>0:
                ent = ent+(-1.0)*prob*np.log(prob)
    return ent

# -----------------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------------
bird_id_list = []
times_list = []
envelope_list = []

freq_list = []
fourier_list=[]
fou_max_list=[]
fou_wid_list=[]
fou_entropy_list=[]

cr_distribution_list = []
cr_max_list=[]
cr_wid_list=[]
cr_entropy_list=[]

for i in range(len(files)):

    # Toakes bird id from file name
    name_list = files[i].split('_')
    bird_id_list.append(int(name_list[3]))

    samp_rate, signal = scipy.io.wavfile.read(files[i])
    times = np.arange(len(signal))/float(samp_rate)

    # Computes the smoothed and normalized Envelope of the sound
    temp_window, amp_window = get_features(signal,samp_rate,interval_time)
    amp_window = np.divide(amp_window,float(max(amp_window)))
    # divides by 1000 because smooth_time_amp is in miliseconds
    smooth_length_amp = int(np.floor(samp_rate*float(smooth_time_amp)/1000))
    amp_window_smooth = smooth(amp_window,smooth_length_amp)
    amp_window_smooth_norm = np.divide(amp_window_smooth,
                                       float(max(amp_window_smooth)))
    times_list.append(temp_window)
    envelope_list.append(amp_window_smooth_norm)

    # Computes the smoothed and normalized Fourier transform
    fourier = np.abs(np.fft.rfft(signal))
    # times[1] es equivalente a 1/float(samp_rate)
    freqs = np.fft.fftfreq(signal.size,d=times[1])
    # smooth_length_fou is in hertz
    smooth_length_fou = int(np.floor(smooth_freq_fou/float(freqs[1])))
    fourier_smooth = smooth(fourier,smooth_length_fou)
    fourier_smooth_norm = np.divide(np.asarray(fourier_smooth),
                                    np.max(fourier_smooth))
    freq_list.append(freqs[0:(len(fourier_smooth_norm)-1)])
    fourier_list.append(fourier_smooth_norm[0:-1])

    # Gets max position
    max_index = np.argmax(fourier_smooth_norm)
    # Gets indexs where fou is greater than 50% its max value
    high_fou = fourier_smooth_norm>0.5
    # Gets the superior limit of width
    # Starts at the max and move rigth until fou less than 50%
    flag = 0
    current_ind = max_index
    while (flag==0):
        if (high_fou[current_ind]==1):
            current_ind = current_ind+1
        else:
            sup_lim_wid = current_ind
            flag = 1
    # Gets the inferior limit of width
    flag = 0
    current_ind = max_index
    while (flag==0):
        if (high_fou[current_ind]==1):
            current_ind = current_ind-1
        else:
            inf_lim_wid = current_ind
            flag = 1

    # Get fourier signal entropy
    fou_entropy = get_entropy(fourier_smooth_norm)

    fou_wid_list.append(freqs[sup_lim_wid]-freqs[inf_lim_wid])
    # This was tricky, relating fou width with max position
    # fou_max_list.append(freqs[int(non_zero_indexs[0]+(non_zero_indexs[-1]-non_zero_indexs[0])/float(2))])
    fou_max_list.append(freqs[np.argmax(fourier_smooth_norm)])
    fou_entropy_list.append(fou_entropy)

    # Computes the smoothed Closed Returns Distribution
    close_returns = get_close_returns(amp_window_smooth,
                                      temp_window, 
                                      close_ret_window,
                                      close_ret_epsilon,
                                      amp_cut,
                                      close_ret_start)
    # Changes 0->1 and 1->0 in matriz and colapses it in the x-axis
    returns_dist = np.sum(close_returns*-1+1,0)
    # Divided by 1000 because smooth_time_cr is in miliseconds
    # smooth_length_cr = int(np.floor(samp_rate*float(smooth_time_cr)/1000))
    smooth_length_cr = int(np.floor(
                           smooth_time_cr/float(1000*temp_window[1])
                           ))
    returns_dist_smooth = smooth(returns_dist,
                                 smooth_length_cr)
    # Normalized by the total number of counts
    tot_counts = returns_dist.sum()
    returns_dist_smooth_norm = np.divide(np.asarray(returns_dist_smooth),
                                         float(tot_counts))

    # Gets entropy of the close returns signal 
    cr_entropy = get_entropy(returns_dist_smooth_norm)

    # Calculates the max and width of close returns distribution
    # max_index = np.argmax(returns_dist_smooth_norm)
    # max_value = returns_dist_smooth_norm[max_index]
    max_value = np.amax(returns_dist_smooth_norm)
    # Calculates the minimun for the nonzero region
    non_zero_init_time = close_ret_start+(smooth_time_cr/float(2))
    non_zero_end_time = temp_window[-1]-(smooth_time_cr/float(2))
    non_zero_init = int(np.floor(
                             non_zero_init_time/float(1000*temp_window[1])
                             ))
    non_zero_end = int(np.floor(
                             non_zero_end_time/float(1000*temp_window[1])
                             ))
    # Sum non_zero_init to get index in the full vector
    min_index = np.argmin(returns_dist_smooth_norm[non_zero_init:non_zero_end])+non_zero_init
    min_value = returns_dist_smooth_norm[min_index]
    # Gets indexs greater half the size of the maximun 
    high_indexs = np.nonzero(returns_dist_smooth_norm[0:-1]>(min_value+(max_value-min_value)/2))[0]
    # Gets indexs of intervals limits
    limits = [high_indexs[0]]
    for n in range(len(high_indexs)-1):
        if (high_indexs[n+1]-high_indexs[n])>1:
            limits.append(high_indexs[n])
            limits.append(high_indexs[n+1])
    limits.append(high_indexs[-1])
    # Gets intervals maximun width, amplitude and index
    inter_max_index_list =[]
    inter_width_list     =[]
    inter_max_value_list =[]
    for n in range(0,len(limits),2):
        # This was tricky, relating width with max position
        # inter_max_index = limits[n]+int((limits[n+1]-limits[n])/2)
        if (limits[n]!=limits[n+1]):
            inter_max_index = np.argmax(returns_dist_smooth_norm[limits[n]:limits[n+1]])
            inter_max_index_list.append(inter_max_index)
            inter_width_list.append(temp_window[limits[n+1]]-temp_window[limits[n]])
            inter_max_value_list.append(returns_dist_smooth_norm[inter_max_index])
    max_inter = max(inter_max_value_list)
    max_index = inter_max_value_list.index(max_inter)
    select_index = max_index
    for it in range(select_index-1,0,-1):
        # PARAMETRO PROVISORIO
        if (inter_max_index_list[it]-(inter_max_index_list[select_index]/2))<(0.05*inter_max_index_list[select_index]):
            select_index = it
    
    cr_max_list.append(temp_window[inter_max_index_list[select_index]])
    cr_wid_list.append(inter_width_list[select_index])
    cr_distribution_list.append(returns_dist_smooth_norm)
    cr_entropy_list.append(cr_entropy)
# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

sns.set()
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
labels = ['Subject 0','Subject 1','Subject 2','Subject 3','Subject 4','Subject 5','Subject 6','Subject 7','Subject 8','Subject 9','Subject 10']

fig1 = plt.figure(1)
ax1 = fig1.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax1.plot(freq_list[j], fourier_list[j],colors[bird_id_list[j]])
    else:
        ax1.plot(freq_list[j],fourier_list[j],colors[bird_id_list[j]],label=labels[bird_id_list[j]])
    old_id = bird_id_list[j]
ax1.set(xlim=[0, 10000])
ax1.set(xlabel='Freqs (hz)',ylabel='Power')
plt.legend(prop={'size':20})
plt.draw()

fig2 = plt.figure(2)
ax2 = fig2.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax2.plot(times_list[j][0:len(cr_distribution_list[j])],cr_distribution_list[j],colors[bird_id_list[j]])
    else:
        ax2.plot(times_list[j][0:len(cr_distribution_list[j])],cr_distribution_list[j],colors[bird_id_list[j]],label=labels[bird_id_list[j]])
    old_id = bird_id_list[j]
ax2.set(xlabel='Time (seconds)',ylabel='Relative Counts')
plt.legend(prop={'size':20})
plt.draw()

fig3 = plt.figure(3)
ax3 = fig3.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax3.scatter(fou_max_list[j],fou_wid_list[j],c=colors[bird_id_list[j]],s=200)
    else:
        ax3.scatter(fou_max_list[j],fou_wid_list[j],c=colors[bird_id_list[j]],s=200,label=labels[bird_id_list[j]])
    ax3.annotate(str(bird_id_list[j]), (fou_max_list[j],fou_wid_list[j]))
    old_id = bird_id_list[j]
ax3.set(xlabel='Max Position (Hz)',ylabel='Width (Hz)',title='Fourier Maximum')
plt.legend(prop={'size':15})
plt.draw()

fig4 = plt.figure(4)
ax4 = fig4.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax4.scatter(cr_max_list[j],cr_wid_list[j],c=colors[bird_id_list[j]],s=200)
    else:
        ax4.scatter(cr_max_list[j],cr_wid_list[j],c=colors[bird_id_list[j]],s=200,label=labels[bird_id_list[j]])
    ax4.annotate(str(bird_id_list[j]), (cr_max_list[j],cr_wid_list[j]))
    old_id = bird_id_list[j]
ax4.set(xlabel='Max Position (Seconds)',ylabel='Width (Seconds)',title='Close Returns Maximum')
plt.legend(prop={'size':15})
plt.draw()

fig5 = plt.figure(5)
ax5 = fig5.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax5.scatter(fou_entropy_list[j],cr_entropy_list[j],c=colors[bird_id_list[j]],s=200)
    else:
        ax5.scatter(fou_entropy_list[j],cr_entropy_list[j],c=colors[bird_id_list[j]],s=200,label=labels[bird_id_list[j]])
    ax5.annotate(str(bird_id_list[j]), (fou_entropy_list[j],cr_entropy_list[j]))
    old_id = bird_id_list[j]
ax5.set(xlabel='Fourier Entropy',ylabel='CR entropy',title='Entropys')
plt.legend(prop={'size':15})
plt.draw()

plt.show()