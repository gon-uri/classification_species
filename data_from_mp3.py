# -*- coding: utf-8 -*-

"""
Created on November 2017

@author: Gonzlao Uribarri

It walks over all the directories present in "folders" list, and decodes ALL MP3
files they contain. It writes on a txt file the results, properly prepared to be
used as inpun in the feature selection algorithm tsfresh: 
https://github.com/blue-yonder/tsfresh.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import scipy.fftpack
import seaborn as sns
import tsfresh as ts
from os import walk
from pydub import AudioSegment



folders = ['/home/usuario/Desktop/data_grupo1/800 Elaenia albiceps',
# '/home/usuario/Desktop/data_grupo1/674 Sylviorthorhynchus desmursii',
# '/home/usuario/Desktop/data_grupo1/1057 Zonotrichia capensis',
# '/home/usuario/Desktop/data_grupo1/1177 Molothrus rufoaxillaris',
 '/home/usuario/Desktop/data_grupo1/574 Leucochloris albicollis']

# folders = ['/home/usuario/Desktop/data_grupo1/test_2']
f = open("/home/usuario/Desktop/data_grupo1/data_2.txt","w")
f.write("id\ttime\tvalues\n")

label = 1
sound_id = 1
label_list = []
sound_list = []

for folder_path in folders:
	files = []
	for (dirpath, dirnames, filenames) in walk(folder_path):
		files.extend(filenames)
	for file_path in files:
		path = folder_path+'/'+file_path
		bird_song = AudioSegment.from_mp3(path)
#		bird_song = AudioSegment.from_wav(path)
		samples = bird_song.get_array_of_samples()
		samples = np.divide(samples,float(max(samples)))
#		samples = samples/float(max(samples))
		rate = bird_song.frame_rate
		times = np.arange(len(samples))/float(rate)
		for i in range(len(times)/8):		
			f.write("%d\t%.4f\t%.4f\n" % (sound_id,times[i],samples[i]))
		label_list.append(label)
		sound_list.append(sound_id)
		sound_id = sound_id+1
	label = label + 1

f.close()

f2 = open("/home/usuario/Desktop/data_grupo1/data_2_labels.txt","w")
for i in range(len(label_list)):
	f2.write("%d\t%d\n"% (sound_list[i],label_list[i]))
f2.close()