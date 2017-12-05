# -*- coding: utf-8 -*-

"""
Created on November 2017

@author: Gonzlao Uribarri

It walks over all the directories present in "bird_species_list", and decodes 
ALL MP3 files they contain. It calculates some features of the signal 
in intervals over time, preparing a set of feature vectors for each audio.

Calculated Features:
- Max, width and entropy of Fourier Transform
- Mean, width and entropy of syllabe time duration

Esta version grafica en el espacio de features, discriminando por especies.
NO grafica las transformadas y NO calcula Close Returns

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
amp_cut           = 0.25    # Percentage of amplitude for cut-off

smooth_time_amp = 0.6     # Smooth window for amplitude (miliseconds)
smooth_freq_fou = 100      # Smooth window for fourier transform (hertz) -300

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

fou_max_list     = []
fou_wid_list     = []
fou_entropy_list = []

syl_mean_list    = []
syl_std_list     = []
syl_entropy_list = []
syl_num_list = []

bird_species_list = ["02-5","02-9","48-574","57-751"]
i = 0

for m in range(len(bird_species_list)):

    bird_species = bird_species_list[m]
    bird_folder = "/home/usuario/Desktop/Canto_Cortado/"+bird_species
    files = [bird_folder+"/"+name for name in os.listdir(bird_folder)]
    files.sort()
    

    for z in range(len(files)):

        samp_rate, signal = scipy.io.wavfile.read(files[z])
        times = np.arange(len(signal))/float(samp_rate)

        # Computes the smoothed and normalized Envelope of the sound
        temp_window, amp_window = get_features(signal,samp_rate,interval_time)
        amp_window = np.divide(amp_window,float(max(amp_window)))
        # divides by 1000 because smooth_time_amp is in miliseconds
        smooth_length_amp = int(np.floor(samp_rate*float(smooth_time_amp)/1000))
        amp_window_smooth = smooth(amp_window,smooth_length_amp)
        amp_window_smooth_norm = np.divide(amp_window_smooth,
                                           float(max(amp_window_smooth)))

        # Computes the mean and std of syllables duration

        syllab_indexs = np.nonzero(amp_window_smooth_norm>amp_cut)[0]

        limits = [syllab_indexs[0]]
        for n in range(len(syllab_indexs)-1):
            if (syllab_indexs[n+1]-syllab_indexs[n])>1:
                limits.append(syllab_indexs[n])
                limits.append(syllab_indexs[n+1])
        limits.append(syllab_indexs[-1])

        syllab_width_list = []

        for n in range(0,len(limits),2):
            # This was tricky, relating width with max position
            # inter_max_index = limits[n]+int((limits[n+1]-limits[n])/2)
            if (limits[n]!=limits[n+1]):
                syllab_width_list.append(temp_window[limits[n+1]]-temp_window[limits[n]])

        syllab_width_array = np.asarray(syllab_width_list)

        syl_mean_list.append(np.mean(syllab_width_array))
        syl_std_list.append(np.std(syllab_width_array))
        syl_entropy_list.append(get_entropy(syllab_width_array))

        # Computes the smoothed and normalized Fourier transform
        fourier = np.abs(np.fft.rfft(signal))
        # times[1] es equivalente a 1/float(samp_rate)
        freqs = np.fft.fftfreq(signal.size,d=times[1])
        # smooth_length_fou is in hertz
        smooth_length_fou = int(np.floor(smooth_freq_fou/float(freqs[1])))
        fourier_smooth      = smooth(fourier,smooth_length_fou)
        fourier_smooth_norm = np.divide(np.asarray(fourier_smooth),
                                        np.max(fourier_smooth))

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

        
        bird_id_list.append(int(m+1))
        i = i+1

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

sns.set()
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
labels = ['Species 0','Species 1','Species 2','Species 3','Species 4','Species 5','Species 6','Species 7','Species 8','Species 9','Species 10']

fig3 = plt.figure(1)
ax3 = fig3.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax3.scatter(fou_max_list[j],fou_wid_list[j],c=colors[bird_id_list[j]],s=200)
    else:
        ax3.scatter(fou_max_list[j],fou_wid_list[j],c=colors[bird_id_list[j]],s=200,label=labels[bird_id_list[j]])
    ax3.annotate(str(bird_id_list[j]), (fou_max_list[j],fou_wid_list[j]))
    old_id = bird_id_list[j]
ax3.set(xlabel='Max Position (Hz)',ylabel='Width (Hz)',title='Fourier Analisis')
plt.legend(prop={'size':15})
plt.draw()

fig4 = plt.figure(2)
ax4 = fig4.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax4.scatter(syl_mean_list[j],syl_std_list[j],c=colors[bird_id_list[j]],s=200)
    else:
        ax4.scatter(syl_mean_list[j],syl_std_list[j],c=colors[bird_id_list[j]],s=200,label=labels[bird_id_list[j]])
    ax4.annotate(str(bird_id_list[j]), (syl_mean_list[j],syl_std_list[j]))
    old_id = bird_id_list[j]
ax4.set(xlabel='Mean Syllabe Duration (Seconds)',ylabel='STD Syllabe Duration (Seconds)',title='Syllabe Statistics')
plt.legend(prop={'size':15})
plt.draw()

fig5 = plt.figure(3)
ax5 = fig5.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax5.scatter(fou_entropy_list[j],syl_entropy_list[j],c=colors[bird_id_list[j]],s=200)
    else:
        ax5.scatter(fou_entropy_list[j],syl_entropy_list[j],c=colors[bird_id_list[j]],s=200,label=labels[bird_id_list[j]])
    ax5.annotate(str(bird_id_list[j]), (fou_entropy_list[j],syl_entropy_list[j]))
    old_id = bird_id_list[j]
ax5.set(xlabel='Fourier Entropy',ylabel='Syllabe Duration entropy',title='Entropys')
plt.legend(prop={'size':15})
plt.draw()

plt.show()