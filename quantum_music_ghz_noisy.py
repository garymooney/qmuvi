
import matplotlib
import qiskit
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator
qiskit.__qiskit_version__
import numpy as np
import time

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

# Import Qiskit
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.providers.aer import AerSimulator
from qiskit.tools.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error


NAME = 'ghz_noisy'
target_folder = NAME
if os.path.isdir(target_folder) == False:
    os.mkdir("./" + NAME)

noise_model = NoiseModel()
single_qubit_dep_error = depolarizing_error(0.02, 1)
noise_model.add_all_qubit_quantum_error(single_qubit_dep_error, ['u1', 'u2', 'u3'])
two_qubit_dep_error = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(two_qubit_dep_error, ['cx'])
simulator = AerSimulator(noise_model = noise_model)

circ = QuantumCircuit(4)
circ.h(0)
circ.save_density_matrix(label=f'rho0')
circ.cx(0, 1)
circ.save_density_matrix(label=f'rho1')
circ.cx(1, 2)
circ.save_density_matrix(label=f'rho2')
circ.cx(2, 3)
circ.save_density_matrix(label=f'rho3')
circ.h(3)
circ.save_density_matrix(label=f'rho4')
circ.cx(3,2)
circ.save_density_matrix(label=f'rho5')
circ.cx(2,1)
circ.save_density_matrix(label=f'rho6')
circ.cx(1,0)
circ.save_density_matrix(label=f'rho7')
circ.measure_all()

circ = transpile(circ, simulator)
circ.draw(filename=f'./{NAME}\\circuit.png',output="mpl")
result = simulator.run(circ).result()
counts = result.get_counts(circ)
plot_histogram(counts, title='Bell-State counts')

rho0 = result.data(0)['rho0']
rho1 = result.data(0)['rho1']
rho2 = result.data(0)['rho2']
rho3 = result.data(0)['rho3']
rho4 = result.data(0)['rho4']
rho5 = result.data(0)['rho5']
rho6 = result.data(0)['rho6']
rho7 = result.data(0)['rho7']

rhos = [rho0,rho1,rho2,rho3,rho4,rho5,rho6,rho7]



sounds_list = []

for rho in rhos:
    sound_data = []
    eps = 1E-8
    w,v = np.linalg.eig(rho)
    for i,p in enumerate(w):
        if p.real > eps:
            prob0 = p.real
            note = {}
            vec = v[:,i]
            for j,a in enumerate(vec):
                if np.abs(a)**2 > eps:
                    prob1 = np.abs(a)**2
                    angle = np.angle(a)
                    note[j] = (prob1,angle)
            sound_data.append((prob0,note,[abs(complex(x))*abs(complex(x)) for x in vec],[np.angle(x) for x in vec]))
    sounds_list.append(sound_data)

import json
with open(f'{NAME}\\sounds_list.json', 'w') as f:
    json.dump(sounds_list, f)

E_MIX = [4, 6, 8, 9, 11, 13, 14, 16, 18, 20, 21, 23, 25, 26, 28, 30, 32, 33, 35, 37, 38, 40, 42, 44, 45, 47, 49, 50, 52, 54, 56, 57, 59, 61, 62, 64, 66, 68, 69, 71, 73, 74, 76, 78, 80, 81, 83, 85, 86, 88, 90, 92, 93, 95, 97, 98, 100, 102, 104, 105, 107, 109, 110, 112, 114, 116, 117, 119, 121, 122, 124, 126]

F_MIN = [5, 7, 8, 10, 12, 13, 15, 17, 19, 20, 22, 24, 25, 27, 29, 31, 32, 34, 36, 37, 39, 41, 43, 44, 46, 48,  49, 51, 53, 55, 56, 58, 60, 61, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87, 89, 91, 92, 94, 96, 97, 99, 101, 103, 104, 106, 108, 109, 111, 113, 115, 116, 118, 120, 121, 123, 125, 127]

C_MAJ = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24, 26, 28, 29, 31, 33, 35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96, 98, 100, 101, 103, 105, 107, 108, 110, 112, 113, 115, 117, 119, 120, 122, 124, 125, 127]


def round_to_scale(n, scale = C_MAJ):
    CURR_MODE = scale
    note = 60+n
    
    # Shifting
    if CURR_MODE and note < CURR_MODE[0]:
        note = CURR_MODE[0]
    else:
        while (CURR_MODE and note not in CURR_MODE):
            note -= 1
    return note

instrument_dict = {'piano': list(range(1,9)),
                   'tuned_perc': list(range(9,17)),
                   'organ': list(range(17,25)),
                   'guitar': list(range(25,33)),
                   'bass': list(range(33,41)),
                   'strings': list(range(41,49)),
                   'ensemble': list(range(49,57)),
                   'brass': list(range(57,65)),
                   'reed': list(range(65,73)),
                   'pipe': list(range(73,81))}

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

mid = MidiFile()
numtracks = 8
tracks = [MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack()]
#track_instruments = ['piano','bass','brass','ensemble','organ','pipe','reed','strings']
track_instruments = ['ensemble']*8

tracks[0].append(MetaMessage('set_tempo', tempo=bpm2tempo(60)))

#time_list = [[120,0]]*25
time_list = [[240,0]] * len(sounds_list)
#time_list = [[80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
#             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
#             [80,0],[40,0],[120,0],[120,0],[120,0],[120,0],[120,0],
#             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],]

with open(f'{NAME}\\rhythm.json', 'w') as f:
    json.dump(time_list, f)

for t_i, sound in enumerate(sounds_list):    
    
    active_notes = []
    sorted_chords = sorted(sound, key = lambda a: a[0], reverse=True)
    for trackno in range(numtracks):
        track = tracks[trackno]

        if trackno >= len(sorted_chords):
            track.append(Message('note_on', note=0, velocity=0, time=0))
            active_notes.append((trackno,note))
            continue
            
        chord_prob, chord, _, _ = sorted_chords[trackno]
        
        max_note_prob = max([chord[n][0] for n in chord])
        
        noteiter = 0
        for n in chord:
            noteiter += 1
            
            note_prob, angle = chord[n]
            
            #prob = note_prob*chord_prob
            prob = (note_prob/max_note_prob)*chord_prob
            #prob = (note_prob/max_note_prob)
            
            note = 60 + n
            #note = round_to_scale(n, scale=C_MAJ)
            #notelist = []
            #note = notelist[n%len(notelist)]
            volume = round(127*(prob))
            #instrument = round(127*(angle + np.pi)/(2*np.pi))
            instruments = instrument_dict[track_instruments[trackno]]
            instrument = instruments[round(7*(angle + np.pi)/(2*np.pi))]
            #instrument = 1

            
            track.append(Message('program_change', program=instrument, time=0))
            track.append(Message('note_on', note=note, velocity=volume, time=0))
            active_notes.append((trackno,note))
    
     
    for track in tracks:   
        track.append(Message('note_on', note=0, velocity=0, time=time_list[t_i][0]))
        track.append(Message('note_off', note=0, velocity=0, time=0))
    
    for trackno, note in active_notes:
        track = tracks[trackno]
        track.append(Message('note_off', note=note, velocity=0, time=0))
        
    for track in tracks:
        track.append(Message('note_on', note=0, velocity=0, time=time_list[t_i][1]))
        track.append(Message('note_off', note=0, velocity=0, time=0))
                        
                        
for track in tracks:
    mid.tracks.append(track)
    
midi_filename = f'{NAME}\\{NAME}'
mid.save(midi_filename + ".mid")
string = 'vlc.exe ' + midi_filename + '.mid -I dummy --no-sout-video --sout-audio --no-sout-rtp-sap --no-sout-standard-sap --ttl=1 --sout-keep --sout "#transcode{acodec=mp3,ab=128}:std{access=file,mux=dummy,dst=./' + midi_filename + '.mp3}"'
command_string = f"{string}"

def run_vlc():
    import os
    #print(string)
    directories = os.system(command_string)

import threading
t = threading.Thread(target=run_vlc,name="vlc",args=())
t.daemon = True
t.start()


print("converting midi to mp3...")
time.sleep(3)










#print(string)
#directories = os.system(command_string)


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
        files = glob.glob(f'{target_folder}\\' + '*.png')
        filename = target_folder + '\\frame_' + str(len(files)) + '.png'
        plt.savefig(filename)
        plt.close('all')
    return 0


import json

with open(target_folder + '\\sounds_list.json') as json_file:
    sound_list = json.load(json_file)

with open(target_folder + '\\rhythm.json') as json_file:
    rhythm = json.load(json_file)
print("rhythm: ", rhythm)

files = glob.glob(target_folder + '\\frame_*')
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

files = glob.glob(target_folder + '\\*.mp3')
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
files = glob.glob(target_folder + '\\frame_*')
iter = 0
for file in files:
    time = (rhythm[iter][0] + rhythm[iter][1]) / 480.0
    clips.append(ImageClip(file).set_duration(time))
    total_time += time
    iter += 1

video = concatenate(clips, method="compose")

audio_clip_new = AudioArrayClip(arr[0:int(44100 * total_time)], fps=44100)

#audio_clip_new.write_audiofile(path + '\\audio.mp3')

circuit_video = ImageClip(target_folder + '\\circuit.png').set_duration(video.duration)

clip_arr = clips_array([[circuit_video.resize(0.8)], [video]])

# video_final = video.set_audio(audio_clip_new)
video_final = clip_arr.set_audio(audio_clip_new)

video_final.write_videofile(target_folder + '\\' + target_folder + '.avi', fps=24, codec='mpeg4')

files = glob.glob(target_folder + '\\*.mp4')
for file in files:
    os.remove(file)