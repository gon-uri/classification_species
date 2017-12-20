# -*- coding: utf-8 -*-

"""
Created on November 2017

@author: Gonzlao Uribarri

It walks over all the directories present in "folders" list, and decodes ALL MP3
files they contain. The it calculates some features of the signal in intervals
over time, preparing a set of fature vectors for each audio.

It then writes on a txt file the results, properly prepared to be
used as inpun in the feature selection algorithm tsfresh: 
https://github.com/blue-yonder/tsfresh.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from os import walk
from pydub import AudioSegment

# --------------------
# Parameters and files
# --------------------

interval_time = 5; #Length of the feature window in miliseconds

folders = [
'/home/usuario/Desktop/data_grupo1/800 Elaenia albiceps',
# # '/home/usuario/Desktop/data_grupo1/674 Sylviorthorhynchus desmursii',
# # '/home/usuario/Desktop/data_grupo1/1057 Zonotrichia capensis',
# # '/home/usuario/Desktop/data_grupo1/1177 Molothrus rufoaxillaris',
'/home/usuario/Desktop/data_grupo1/574 Leucochloris albicollis'
]

#folders = [
#'/home/usuario/Desktop/data_grupo1/ejem']

f = open("/home/usuario/Desktop/data_grupo1/amp_ent_2.txt","w")

# --------------------
# Function definition
# --------------------

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

#	filt = (amplitude>np.amax(amplitude)/45.0)
#	filtered_amplitude = filt*amplitude
#	filtered_entropy = filt*entropy

	return times, amplitude, entropy, input_signal[(i*step_length):((i+1)*step_length-1)]



# --------------------
# Main Program
# --------------------

label      = 1
sound_id   = 1
label_list = []
sound_list = []

f.write("id\ttime\tamplitude\tentropy\n")

for folder_path in folders:
	files = []
	print ("Folder %d of %d" % (label,len(folders)))
	for (dirpath, dirnames, filenames) in walk(folder_path):
		files.extend(filenames)
	for file_path in files:
		path      = folder_path+'/'+file_path
		bird_song = AudioSegment.from_mp3(path)
		signal    = bird_song.get_array_of_samples()
		signal    = np.divide(signal,float(max(signal)))
		samp_rate = bird_song.frame_rate
		times, amplitude, entropy, fourier = get_features(signal,samp_rate,interval_time)
		for i in range(len(times)):		
			f.write("%d\t%.3f\t%.3f\t%.3f\n" % (sound_id,times[i],amplitude[i],entropy[i]))
		label_list.append(label)
		sound_list.append(sound_id)
		sound_id = sound_id+1
	label = label + 1

f.close()

f2 = open("/home/usuario/Desktop/data_grupo1/amp_ent_2_labels.txt","w")
for i in range(len(label_list)):
	f2.write("%d\t%d\n"% (sound_list[i],label_list[i]))
f2.close()