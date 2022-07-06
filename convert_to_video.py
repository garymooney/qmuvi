from jinja2 import FileSystemLoader
import matplotlib.pylab as plt
import numpy as np
from matplotlib.pylab import cm, mpl
import os
import glob
import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
import sys
import math

path = os.path.dirname(__file__)

print(str(sys.argv[1]))
target_folder = str(sys.argv[1])
if os.path.isdir(target_folder) == False:
    print("Error: folder " + str(sys.argv[1]) + " does not exist.")
    exit()

def plot_quantum_state(input_probability_vector, angle_vector, save=None):
    input_length = len(input_probability_vector)
    num_qubits = len(bin(input_length - 1)) - 2
    labels = []
    for x in range(input_length):
        label_tem = bin(x).split('0b')[1]
        label_tem_2 = label_tem
        for y in range(num_qubits - len(label_tem)):
            label_tem_2 = '0' + label_tem_2
        labels.append(label_tem_2)

    cmap = cm.get_cmap('hsv')  # Get desired colormap - you can change this!
    max_height = 2*np.pi
    min_height = -np.pi
    rgba = [cmap((k - min_height) / max_height) for k in angle_vector]

    
    X = np.linspace(1, input_length, input_length)
    fig = plt.figure('State Visualisation', figsize=(15, 6))
    bar_list = plt.bar(X, input_probability_vector, width=0.5)
    for x in range(input_length):
        bar_list[x].set_color(rgba[x])
    #cb1 = fig.colorbar(cm.ScalarMappable(cmap=cmap), ticks=[0, 0.25, 0.5, 0.75, 1], aspect=10)
    #cb1.ax.tick_params(labelsize=20)
    #cb1.set_label('Phase (degree)', rotation=270, fontsize=20, labelpad=20)
    #cb1.ax.set_yticklabels([-180, -90, 0, 90, 180])

    if num_qubits > 4:
        plt.xticks(X, labels, fontsize=20 / (2 ** (num_qubits - 4)) - 2)
    else:
        plt.xticks(X, labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0, max(input_probability_vector)])
    plt.xlabel('State', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.tight_layout()

    if save:
        files = glob.glob(f'{path}\\' + target_folder + '\\*.png')
        filename = path + '\\' + target_folder + '\\frame_' + str(len(files)) + '.png'
        plt.savefig(filename)
        plt.close('all')
    return 0


import json

with open(path + '\\' + target_folder + '\\sounds_list.json') as json_file:
    sound_list = json.load(json_file)

with open(path + '\\' + target_folder + '\\rhythm.json') as json_file:
    rhythm = json.load(json_file)
print("rhythm: ", rhythm)

files = glob.glob(f'{path}\\' + target_folder + '\\frame_*')
for file in files:
    os.remove(file)

init_2 = 0
for sound_data in sound_list:
    largest_iter = 0
    largest_prob = 0
    iter = 0

    for prob, state_data, prob_vec, angle_vec in sound_data:
        if prob > largest_prob:
            largest_prob = prob
            largest_iter = iter
            most_likely_prob_vec = prob_vec
            most_likely_angle_vec = angle_vec
        iter += 1
    plot_quantum_state(most_likely_prob_vec, most_likely_angle_vec, save=True)
    init_2 += 1
    if (init_2 == 6):
        print("most_likely_angle_vec:", most_likely_angle_vec)
    
# video_clip = mpy.ImageSequenceClip(path + '\\figs\\', fps=1)


# clip.write_videofile('moive.mp4',fps=1,audio=False,codec='mpeg4')

def make_frame(t):
    # print(t)
    value = 2 * [np.sin(440 * 2 * np.pi * t)]
    #print(value)
    return value

# audio_clip = mpy.AudioClip(make_frame, duration=5,fps=44100)
#
# video_final = video_clip.set_audio(audio_clip)

from moviepy.editor import *

files = glob.glob(f'{path}\\' + target_folder + '\\*.mp3')
#audioclip = AudioFileClip(path + "\\test-music.mp3")
audio_clip = mpy.AudioFileClip(files[0], fps=44100)
arr = audio_clip.to_soundarray()



# Load myHolidays.mp4 and select the subclip 00:00:50 - 00:00:60
# clip = VideoFileClip("cat-0.mp4").subclip(50,60)
# clip1 = ImageClip(path+"\\figs\\0.png")
# clip1 = clip1.set_duration(1)
#
# # Reduce the audio volume (volume x 0.8)
# # clip = clip.volumex(0.8)
#
# clip2 = ImageClip(path+"\\figs\\1.png")
# clip2 = clip2.set_duration(3)

clips = []

total_time = 0
files = glob.glob(f'{path}\\' + target_folder + '\\frame_*')
iter = 0
for file in files:
    time = (rhythm[iter][0] + rhythm[iter][1]) / 480.0
    clips.append(ImageClip(file).set_duration(time))
    total_time += time
    iter += 1

video = concatenate(clips, method="compose")

audio_clip_new = AudioArrayClip(arr[0:int(44100 * total_time)], fps=44100)

#audio_clip_new.write_audiofile(path + '\\audio.mp3')

circuit_video = ImageClip(path + '\\' + target_folder + '\\circuit.png').set_duration(video.duration)

clip_arr = clips_array([[circuit_video.resize(0.8)], [video]])

# video_final = video.set_audio(audio_clip_new)
video_final = clip_arr.set_audio(audio_clip_new)

video_final.write_videofile(path + '\\' + target_folder + '\\' + target_folder + '.avi', fps=24, codec='mpeg4')

files = glob.glob(f'{path}\\' + target_folder + '\\*.mp4')
for file in files:
    os.remove(file)