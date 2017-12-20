# -*- coding: utf-8 -*-

"""
Created on December 2017

@author: Gonzlao Uribarri

It walks over all the directories present in "bird_species_list", and decodes
ALL MP3 files they contain. It calculates features of the signal
preparing a set of feature vectors for each audio. Then it calculates the
mean (trimmed-mean) for each species.

It also shows the vector for each bird and the species mean.

Then it calculates the DENDOGRAM based on the distance matrix between the
different species.

Calculated Features:
- Max and width of Fourier Transform
- Max and width of Syllabes Times

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import os
import scipy.io.wavfile
# from pydub import AudioSegment

# For dendogram
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

# -----------------------------------------------------------------------------
# Parameters and files
# -----------------------------------------------------------------------------

interval_time = 2       # Length of the amplitude window (miliseconds)
amp_cut = 0.25    # Percentage of amplitude for cut-off

smooth_time_amp = 0.6     # Smooth window for amplitude (miliseconds)
smooth_freq_fou = 100      # Smooth window for fourier transform (hertz) -300

bird_species_list = ["02-5", "02-9", "05-64", "48-574", "57-751",
                     "63-911", "75-995", "76-1002", "85-1188"]

# -----------------------------------------------------------------------------
# Functions definition
# -----------------------------------------------------------------------------


def get_amplitude(input_signal, samp_rate, interval_time):
    # Number of samples in one step
    step_length = int(np.floor(samp_rate * float(interval_time) / 1000))
    # Number of steps in the input signal
    steps = int(np.floor(len(input_signal) / float(step_length)))
    # Full vector of times
    full_times = np.arange(len(input_signal)) / float(samp_rate)

    # Defining vector lengths
    times = np.zeros(steps)
    amplitude = np.zeros(steps)

    # Getting the amplitude of each window
    for i in range(steps):
        maximum = np.amax(np.absolute(
            input_signal[(i * step_length):((i + 1) * step_length - 1)]
        ))
        amplitude[i] = maximum
        times[i] = full_times[i * step_length]

    return times, amplitude


def get_syllabes_width(temp, sign, cut):

    syllab_indexs = np.nonzero(sign > cut)[0]

    limits = [syllab_indexs[0]]
    for n in range(len(syllab_indexs) - 1):
        if (syllab_indexs[n + 1] - syllab_indexs[n]) > 1:
            limits.append(syllab_indexs[n])
            limits.append(syllab_indexs[n + 1])
    limits.append(syllab_indexs[-1])

    syllab_width_list = []

    for n in range(0, len(limits), 2):
        # This was tricky, relating width with max position
        # inter_max_index = limits[n]+int((limits[n+1]-limits[n])/2)
        if (limits[n] != limits[n + 1]):
            syllab_width_list.append(
                temp[limits[n + 1]] - temp[limits[n]])

    syllab_width_array = np.asarray(syllab_width_list)
    return syllab_width_array


def get_max_wid(xs, signal, cut):

    # Gets max position
    max_index = np.argmax(signal)
    # Gets indexs where fou is greater than (cut*100)% its max value
    high_sig = signal > cut
    # Gets the superior limit of width
    # Starts at the max and move rigth until fou less than (cut*100)%
    flag = 0
    current_ind = max_index
    while (flag == 0):
        if (high_sig[current_ind] == 1):
            current_ind = current_ind + 1
        else:
            sup_lim_wid = current_ind
            flag = 1
    # Gets the inferior limit of width
    flag = 0
    current_ind = max_index
    while (flag == 0):
        if (high_sig[current_ind] == 1):
            current_ind = current_ind - 1
        else:
            inf_lim_wid = current_ind
            flag = 1
    sig_wid = xs[sup_lim_wid] - xs[inf_lim_wid]
    sig_max = xs[max_index]
    return sig_max, sig_wid


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_entropy(vector):
    tot = float(np.sum(vector))
    if tot != 0:
        vector_norm = np.divide(vector, tot)
        ent = 0
        for prob in vector_norm:
            if prob > 0:
                ent = ent + (-1.0) * prob * np.log(prob)
    return ent


def dist(x, y):
    return np.sqrt(np.sum(np.divide((x - y),
                   np.asfarray(np.minimum(x, y), dtype='float'))**2))


def trimmed_data(data, percentile):
    data = np.array(data)
    data.sort()
    percentile = percentile / 2.
    low = int(percentile * len(data))
    high = int((1. - percentile) * len(data))
    return data[low:high]

# -----------------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------------


bird_id_list = []

fou_max_list = []
fou_wid_list = []
fou_entropy_list = []

syl_mean_list = []
syl_std_list = []
syl_entropy_list = []
syl_num_list = []

i = 0

group_fou_max_list = []
group_fou_wid_list = []
group_fou_ent_list = []
group_syl_mean_list = []
group_syl_std_list = []
group_syl_ent_list = []

for m in range(len(bird_species_list)):

    bird_species = bird_species_list[m]
    bird_folder = "/home/usuario/Desktop/Canto_Cortado/" + bird_species
    files = [bird_folder + "/" + name for name in os.listdir(bird_folder)]
    files.sort()

    specie_fou_max_mean = []
    specie_fou_wid_mean = []
    specie_fou_ent_mean = []
    specie_syl_mean_mean = []
    specie_syl_wid_mean = []
    specie_syl_ent_mean = []

    for z in range(len(files)):

        samp_rate, signal = scipy.io.wavfile.read(files[z])
        times = np.arange(len(signal)) / float(samp_rate)

        # Computes the smoothed and normalized Envelope of the sound
        temp, amp = get_amplitude(
            signal, samp_rate, interval_time)

        # amp = np.divide(amp, float(max(amp)))
        # divides by 1000 because smooth_time_amp is in miliseconds
        smooth_length_amp = int(
            np.floor(samp_rate * float(smooth_time_amp) / 1000))
        amp_smooth = smooth(amp, smooth_length_amp)
        amp_smooth_norm = np.divide(amp_smooth,
                                    float(max(amp_smooth)))

        # Computes the mean and std of syllables duration
        syllab_width_array = get_syllabes_width(temp, amp_smooth_norm, amp_cut)
        # Get syl duration distribution entropy
        syl_entropy = get_entropy(syllab_width_array)

        syl_mean_list.append(np.mean(syllab_width_array))
        specie_syl_mean_mean.append(np.mean(syllab_width_array))

        syl_std_list.append(np.std(syllab_width_array))
        specie_syl_wid_mean.append(np.std(syllab_width_array))

        syl_entropy_list.append(syl_entropy)
        specie_syl_ent_mean.append(syl_entropy)

        # Computes the smoothed and normalized Fourier transform
        fourier = np.abs(np.fft.rfft(signal))
        # times[1] es equivalente a 1/float(samp_rate)
        freqs = np.fft.fftfreq(signal.size, d=times[1])
        # smooth_length_fou is in hertz
        smooth_length_fou = int(np.floor(smooth_freq_fou / float(freqs[1])))
        fourier_smooth = smooth(fourier, smooth_length_fou)
        fourier_smooth_norm = np.divide(np.asarray(fourier_smooth),
                                        np.max(fourier_smooth))

        # Gets max position position and FWHM (width at half amplitude)
        cut = 0.5
        fou_max, fou_wid = get_max_wid(freqs, fourier_smooth_norm, cut)

        # Get fourier signal entropy
        fou_entropy = get_entropy(fourier_smooth_norm)

        fou_wid_list.append(fou_wid)
        specie_fou_wid_mean.append(fou_wid)

        fou_max_list.append(fou_max)
        specie_fou_max_mean.append(fou_max)

        fou_entropy_list.append(fou_entropy)
        specie_fou_ent_mean.append(fou_entropy)

        bird_id_list.append(int(m + 1))

        i = i + 1

    group_fou_max_list.append(sp.stats.trim_mean(
        np.asarray(specie_fou_max_mean), 0.1))
    group_fou_wid_list.append(sp.stats.trim_mean(
        np.asarray(specie_fou_wid_mean), 0.1))
    group_fou_ent_list.append(sp.stats.trim_mean(
        np.asarray(specie_fou_ent_mean), 0.1))
    group_syl_mean_list.append(sp.stats.trim_mean(
        np.asarray(specie_syl_mean_mean), 0.1))
    group_syl_std_list.append(sp.stats.trim_mean(
        np.asarray(specie_syl_wid_mean), 0.1))
    group_syl_ent_list.append(sp.stats.trim_mean(
        np.asarray(specie_syl_ent_mean), 0.1))

    # group_fou_max_list.append(np.mean(np.asarray(specie_fou_max_mean)))
    # group_syl_mean_list.append(np.mean(np.asarray(specie_syl_mean_mean)))
    # group_syl_std_list.append(np.mean(np.asarray(specie_syl_wid_mean)))
    # group_fou_wid_list.append(np.mean(np.asarray(specie_fou_wid_mean)))

matrix = []
# matrix.append(group_fou_max_list)
# matrix.append(group_fou_wid_list)
# matrix.append(group_fou_ent_list)
# matrix.append(group_syl_mean_list)
# matrix.append(group_syl_std_list)
matrix.append(group_syl_ent_list)
matrix = np.asarray(matrix)
matrix = np.transpose(matrix)

# # # dist_mat = np.zeros((np.shape(matrix)[0], np.shape(matrix)[0]))
# # dist_mat = []
# # # dist_mat2 = sp.spatial.distance.pdist(matrix)
# # for i in range(np.shape(matrix)[0]):
# #     for j in range(0, i, 1):
# #         dist_mat.append(dist(matrix[i], matrix[j]))
# #         # dist_mat[i][j] = dist(matrix[i], matrix[j])

# # dist_mat = np.asarray(dist_mat)

Z = linkage(matrix, 'weighted', 'seuclidean')

labels = [ii for ii in range(1, len(bird_species_list) + 1)]

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
H = dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    labels=labels,
    get_leaves=True
)
plt.show()


dm = scipy.spatial.distance.squareform(pdist(matrix,'seuclidean'))
dm1 = dm[:, H['leaves']][H['leaves']]
# Podriamos pasarle directamente "matrix" y calcular el clustering por fila
# Pero hacemos explicito el clustering y le pasamos la pmatriz de distancia
# Y ademas como se linkean estos valores
ax = sns.clustermap(dm, row_linkage=Z, col_linkage=Z)
plt.show()

# import plotly as py
# import plotly.figure_factory as ff

# fig = ff.create_dendrogram(matrix, orientation='left', labels=bird_species_list)
# fig['layout'].update({'width':800, 'height':800})
# py.offline.iplot(fig, filename='dendrogram_with_labels')




# # Initialize figure by creating upper dendrogram
# figure = FF.create_dendrogram(matrix, orientation='bottom', labels=bird_species_list)
# for i in range(len(figure['data'])):
#     figure['data'][i]['yaxis'] = 'y2'

# # Create Side Dendrogram
# dendro_side = FF.create_dendrogram(matrix, orientation='right')
# for i in range(len(dendro_side['data'])):
#     dendro_side['data'][i]['xaxis'] = 'x2'

# # Add Side Dendrogram Data to Figure
# figure['data'].extend(dendro_side['data'])

# # Create Heatmap
# dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
# dendro_leaves = list(map(int, dendro_leaves))
# data_dist = pdist(matrix)
# heat_data = squareform(data_dist)
# heat_data = heat_data[dendro_leaves,:]
# heat_data = heat_data[:,dendro_leaves]

# heatmap = Data([
#     Heatmap(
#         x = dendro_leaves,
#         y = dendro_leaves,
#         z = heat_data,
#         colorscale = 'YIGnBu'
#     )
# ])

# heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
# heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

# # Add Heatmap Data to Figure
# figure['data'].extend(Data(heatmap))

# # Edit Layout
# figure['layout'].update({'width':800, 'height':800,
#                          'showlegend':False, 'hovermode': 'closest',
#                          })
# # Edit xaxis
# figure['layout']['xaxis'].update({'domain': [.15, 1],
#                                   'mirror': False,
#                                   'showgrid': False,
#                                   'showline': False,
#                                   'zeroline': False,
#                                   'ticks':""})
# # Edit xaxis2
# figure['layout'].update({'xaxis2': {'domain': [0, .15],
#                                    'mirror': False,
#                                    'showgrid': False,
#                                    'showline': False,
#                                    'zeroline': False,
#                                    'showticklabels': False,
#                                    'ticks':""}})

# # Edit yaxis
# figure['layout']['yaxis'].update({'domain': [0, .85],
#                                   'mirror': False,
#                                   'showgrid': False,
#                                   'showline': False,
#                                   'zeroline': False,
#                                   'showticklabels': False,
#                                   'ticks': ""})
# # Edit yaxis2
# figure['layout'].update({'yaxis2':{'domain':[.825, .975],
#                                    'mirror': False,
#                                    'showgrid': False,
#                                    'showline': False,
#                                    'zeroline': False,
#                                    'showticklabels': False,
#                                    'ticks':""}})

# # Plot!
# py.iplot(figure, filename='dendrogram_with_heatmap')


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

sns.set()
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
labels = ['Species 0', 'Species 1', 'Species 2', 'Species 3', 'Species 4',
          'Species 5', 'Species 6', 'Species 7', 'Species 8', 'Species 9',
          'Species 10', 'Species 11', 'Species 12', 'Species 13']


# import matplotlib.mlab.bivariate_normal as b_n

# old_id = 0
# for j in range(len(bird_id_list)):


# b_n(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0)

fig3 = plt.figure(3)
ax3 = fig3.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax3.scatter(fou_max_list[j], fou_wid_list[j],
                    c=colors[bird_id_list[j]], s=50, alpha=0.5)
    else:
        ax3.scatter(fou_max_list[j], fou_wid_list[j],
                    c=colors[bird_id_list[j]], s=50,
                    label=labels[bird_id_list[j]], alpha=0.5)
    old_id = bird_id_list[j]
old_id = 0
# Repito para que se vea arriba de lo anterior
for j in range(len(bird_id_list)):
    if old_id != bird_id_list[j]:
        ax3.scatter(group_fou_max_list[bird_id_list[j] - 1],
                    group_fou_wid_list[bird_id_list[j] - 1],
                    c=colors[bird_id_list[j]], s=600,
                    edgecolors='black', marker='+')
        ax3.annotate(str(bird_id_list[j]), (
            group_fou_max_list[bird_id_list[j] - 1],
            group_fou_wid_list[bird_id_list[j] - 1]))
    old_id = bird_id_list[j]
ax3.set(xlabel='Max Position (Hz)',
        ylabel='Width (Hz)', title='Fourier Analisis')
plt.legend(prop={'size': 15})
plt.draw()

# from statsmodels.robust.scale import huber
fig4 = plt.figure(4)
ax4 = fig4.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax4.scatter(syl_mean_list[j], syl_std_list[j],
                    c=colors[bird_id_list[j]], s=50, alpha=0.5)
    else:
        ax4.scatter(syl_mean_list[j], syl_std_list[j],
                    c=colors[bird_id_list[j]], s=50,
                    label=labels[bird_id_list[j]], alpha=0.5)
    old_id = bird_id_list[j]
old_id = 0
# Repito para que se vea arriba de lo anterior
for j in range(len(bird_id_list)):
    if old_id != bird_id_list[j]:
        ax4.scatter(group_syl_mean_list[bird_id_list[j] - 1],
                    group_syl_std_list[bird_id_list[j] - 1],
                    c=colors[bird_id_list[j]], s=600,
                    edgecolors='black', marker='+')
        # ax4.scatter(huber(np.asarray(specie_syl_mean_mean))[0], huber(
        # np.asarray(specie_syl_wid_mean))[0], c='black',
        # s=600, marker='+')
        ax4.annotate(str(bird_id_list[j]), (
            group_syl_mean_list[bird_id_list[j] - 1],
            group_syl_std_list[bird_id_list[j] - 1]))
    old_id = bird_id_list[j]
ax4.set(xlabel='Mean Syllabe Duration (Seconds)',
        ylabel='STD Syllabe Duration (Seconds)', title='Syllabe Statistics')
plt.legend(prop={'size': 15})
plt.draw()


fig5 = plt.figure(5)
ax5 = fig5.gca()
old_id = 0
for j in range(len(bird_id_list)):
    if old_id == bird_id_list[j]:
        ax5.scatter(fou_entropy_list[j], syl_entropy_list[j],
                    c=colors[bird_id_list[j]], s=50, alpha=0.5)
    else:
        ax5.scatter(fou_entropy_list[j], syl_entropy_list[j],
                    c=colors[bird_id_list[j]], s=50,
                    label=labels[bird_id_list[j]], alpha=0.5)
    old_id = bird_id_list[j]
old_id = 0
# Repito para que se vea arriba de lo anterior
for j in range(len(bird_id_list)):
    if old_id != bird_id_list[j]:
        ax5.scatter(group_fou_ent_list[bird_id_list[j] - 1],
                    group_syl_ent_list[bird_id_list[j] - 1],
                    c=colors[bird_id_list[j]], s=600,
                    edgecolors='black', marker='+')
        # ax5.scatter(huber(np.asarray(specie_syl_mean_mean))[0], huber(
        # np.asarray(specie_syl_wid_mean))[0], c='black',
        # s=600, marker='+')
        ax5.annotate(str(bird_id_list[j]), (
            group_fou_ent_list[bird_id_list[j] - 1],
            group_syl_ent_list[bird_id_list[j] - 1]))
    old_id = bird_id_list[j]
ax5.set(xlabel='Fourier Entropy',
        ylabel='Syllabe Duration entropy', title='Entropys')
plt.legend(prop={'size': 15})
plt.draw()

plt.show()


fig6 = plt.figure(6)
ax6 = fig6.gca()
frec = np.asarray(group_fou_max_list)
min_size = [28, 40, 50, 10, 14, 21, 24.5, 23.5, 10]
max_size = [30, 41, 55, 11.5, 16, 27, 25.5, 26, 14]
ax6.set(ylabel='Size(cm)',
        xlabel='Frequency (Hz)', title='Freq vs Size')
plt.scatter(frec, min_size)
plt.scatter(frec, max_size)
plt.show()
