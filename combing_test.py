from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip

import matplotlib.pylab as plt
import numpy as np
from matplotlib.pylab import cm, mpl
import os
import glob
import sys
from matplotlib import gridspec

path = os.path.dirname(__file__)

target_folder = '\\figs\\'

# print(str(sys.argv[1]))
# target_folder = str(sys.argv[1])
# if os.path.isdir(target_folder) == False:
#     print("Error: folder " + str(sys.argv[1]) + " does not exist.")
#     exit()

# input_probability_vector = np.random.uniform(0, 1, (8, 16))
# angle_vector = np.random.uniform(-np.pi, np.pi, [8, 16])
# fig_title = np.arange(1, 9)
# main_title = ("Shor's Algorithm")

target_folder = '\\target_folder\\'


def plot_quantum_state(input_probability_vector, angle_vector, main_title=None, fig_title=None, save=None):
    num_figs = 8
    input_length = 16

    num_qubits = len(bin(input_length - 1)) - 2
    labels = []
    for x in range(input_length):
        label_tem = bin(x).split('0b')[1]
        label_tem_2 = label_tem
        for y in range(num_qubits - len(label_tem)):
            label_tem_2 = '0' + label_tem_2
        labels.append(label_tem_2)

    cmap = cm.get_cmap('hsv')  # Get desired colormap - you can change this!
    max_height = 2 * np.pi
    min_height = -np.pi
    X = np.linspace(1, input_length, input_length)

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, :])

    num_column = int(num_figs / 2)
    fig, ax = plt.subplots(2, num_column, figsize=(24, 12))
    gs = fig.add_gridspec(2, num_column)
    # if str(main_title):
    #     fig.suptitle(str(main_title),fontsize=24)
    ax_main = fig.add_subplot(gs[0, 1:3])
    ax_main.axes.xaxis.set_visible(False)
    ax_main.axes.yaxis.set_visible(False)

    index = 1
    for i in range(2):
        for j in range(num_column):
            if i == 0 and j == 1:
                ax[i, j].axes.yaxis.set_visible(False)
                ax[i, j].axes.xaxis.set_visible(False)
            elif i == 0 and j == 2:
                ax[i, j].axes.yaxis.set_visible(False)
                ax[i, j].axes.xaxis.set_visible(False)
            else:
                plt.sca(ax[i, j])
                rgba = [cmap((k - min_height) / max_height) for k in angle_vector[index, :]]
                bar_list = plt.bar(X, input_probability_vector[index, :], width=0.5)
                ax[i, j].set_ylim([0, np.max(input_probability_vector)])
                # if str(fig_title):
                #     ax[i, j].set_title(str(fig_title[index]), fontsize=20)
                for x in range(input_length):
                    bar_list[x].set_color(rgba[x])
                if j != 0:
                    ax[i, j].axes.yaxis.set_visible(False)
                ax[i, j].axes.xaxis.set_visible(False)
                if j == 0:
                    plt.yticks(fontsize=20)
                index = index + 1

    fig.text(0.5, 0.08, 'Quantum states', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Probability', va='center', rotation='vertical', fontsize=20)

    rgba = [cmap((k - min_height) / max_height) for k in angle_vector[0, :]]
    bar_list = ax_main.bar(X, input_probability_vector[0, :], width=0.5)
    ax_main.set_ylim([0, np.max(input_probability_vector)])
    for x in range(input_length):
        bar_list[x].set_color(rgba[x])
    # ax_main.set_title(str(fig_title[0]), fontsize=20)
    if save:
        files = glob.glob(f'{path}\\' + target_folder + '\\*.png')
        filename = path + '\\' + target_folder + '\\frame_' + str(len(files)) + '.png'
        plt.savefig(filename)
        plt.close('all')
    return 0


import json

with open(path + '\\' + target_folder + '\\grover_Fmin7_noisy_sounds_list.json') as json_file:
    sound_list = json.load(json_file)

# with open(path + '\\' + target_folder + '\\rhythm.json') as json_file:
#     rhythm = json.load(json_file)
# print("rhythm: ", rhythm)

files = glob.glob(f'{path}\\' + target_folder + '\\frame_*')
for file in files:
    os.remove(file)
#
# init_2 = 0
# for sound_data in sound_list:
#     largest_iter = 0
#     largest_prob = 0
#     iter = 0
#
#     for prob, state_data, prob_vec, angle_vec in sound_data:
#         if prob > largest_prob:
#             largest_prob = prob
#             largest_iter = iter
#             most_likely_prob_vec = prob_vec
#             most_likely_angle_vec = angle_vec
#         iter += 1
#     plot_quantum_state(most_likely_prob_vec, most_likely_angle_vec, save=True)
#     init_2 += 1
#     if (init_2 == 6):
#         print("most_likely_angle_vec:", most_likely_angle_vec)


num_frames = len(sound_list)
input_probability_vector = np.zeros((8, 16))
angle_vector = np.zeros((8, 16))
for i in range(num_frames):
    for j in range(8):
        input_probability_vector[j, :] = np.array(sound_list[i][j][2]) * sound_list[i][j][0]
        angle_vector[j, :] = sound_list[i][j][3]
    plot_quantum_state(input_probability_vector, angle_vector, save=1)

audioclip = AudioFileClip(path + "\\test-music.mp3")
audio_clip = AudioFileClip(path + "\\test-music.mp3", fps=44100)
arr = audio_clip.to_soundarray()

files = glob.glob(f'{path+target_folder}\\frame_*.png')
clips = []

total_time = 0
for file in files:
    clips.append(ImageClip(file).set_duration(0.1))
    total_time += 0.1

video = concatenate(clips, method="compose")

audio_clip_new = AudioArrayClip(arr[0:int(44100 * total_time)], fps=44100)

audio_clip_new.write_audiofile(path + '\\audio.mp3')

circuit_video = ImageClip(path + '\\circuit.png').set_duration(video.duration)

clip_arr = clips_array([[circuit_video.resize(0.8)], [video]])

# video_final = video.set_audio(audio_clip_new)
video_final = clip_arr.set_audio(audio_clip_new)

video_final.write_videofile(path + '\\test.avi', fps=24, codec='mpeg4')
