import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import scipy.fftpack
import seaborn as sns
import tsfresh as ts
from os import walk

folders = ['/home/usuario/Desktop/data_grupo1/test_2']

for folder_path in folders:
	files = []
	for (dirpath, dirnames, filenames) in walk(folder_path):
		files.extend(filenames)
	for file_path in files:
		path                 = folder_path+'/'+file_path
		samplerate_s, data_s = wavfile.read(path)
		times                = np.arange(len(data_s))/float(samplerate_s)
