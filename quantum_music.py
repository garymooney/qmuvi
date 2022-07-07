
import matplotlib
import qiskit
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator
qiskit.__qiskit_version__
import numpy as np
import time
from moviepy.editor import *

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

from qiskit.converters import circuit_to_dag

def chromatic_middle_c(n):
    return n + 60

def c_major(n):
    C_MAJ = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24, 26, 28, 29, 31, 33, 35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96, 98, 100, 101, 103, 105, 107, 108, 110, 112, 113, 115, 117, 119, 120, 122, 124, 125, 127]

    CURR_MODE = C_MAJ
    note = 60+n
    
    # Shifting
    if CURR_MODE and note < CURR_MODE[0]:
        note = CURR_MODE[0]
    else:
        while (CURR_MODE and note not in CURR_MODE):
            note -= 1
    return note

def c_major(n):
    C_MAJ = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24, 26, 28, 29, 31, 33, 35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96, 98, 100, 101, 103, 105, 107, 108, 110, 112, 113, 115, 117, 119, 120, 122, 124, 125, 127]

    CURR_MODE = C_MAJ
    note = 60+n
    
    # Shifting
    if CURR_MODE and note < CURR_MODE[0]:
        note = CURR_MODE[0]
    else:
        while (CURR_MODE and note not in CURR_MODE):
            note -= 1
    return note

def f_minor(n):
    F_MIN = [5, 7, 8, 10, 12, 13, 15, 17, 19, 20, 22, 24, 25, 27, 29, 31, 32, 34, 36, 37, 39, 41, 43, 44, 46, 48,  49, 51, 53, 55, 56, 58, 60, 61, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87, 89, 91, 92, 94, 96, 97, 99, 101, 103, 104, 106, 108, 109, 111, 113, 115, 116, 118, 120, 121, 123, 125, 127]

    CURR_MODE = F_MIN
    note = 60+n
    
    # Shifting
    if CURR_MODE and note < CURR_MODE[0]:
        note = CURR_MODE[0]
    else:
        while (CURR_MODE and note not in CURR_MODE):
            note -= 1
    return note

def make_music_video(qc, name, rhythm, single_qubit_error, two_qubit_error, input_instruments, note_map = chromatic_middle_c):
    NAME = name
    target_folder = NAME
    if os.path.isdir(target_folder) == False:
        os.mkdir("./" + NAME)

    noise_model = NoiseModel()
    single_qubit_dep_error = depolarizing_error(single_qubit_error, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_dep_error, ['u1', 'u2', 'u3'])
    two_qubit_dep_error = depolarizing_error(two_qubit_error, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_dep_error, ['cx'])
    simulator = AerSimulator(noise_model = noise_model)

    dag = circuit_to_dag(qc)
    new_qc = QuantumCircuit(len(dag.qubits))

    barrier_iter = 0
    for node in dag.topological_op_nodes():
        
        if node.name == "barrier":
            new_qc.save_density_matrix(label=f'rho{barrier_iter}')
            barrier_iter += 1
        if node.name != "measure":
            new_qc.append(node.op, node.qargs, node.cargs)
    barrier_count = barrier_iter
    circ = new_qc

    circ = transpile(circ, simulator)
    
    qc.draw(filename=f'./{NAME}\\circuit.png',output="mpl")
    result = simulator.run(circ).result()
    # = result.get_counts(circ)
    #plot_histogram(counts, title='Bell-State counts')

    rhos = []
    for i in range(barrier_count):
        rhos.append(result.data(0)[f'rho{i}'])

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

    for i, sound in enumerate(sounds_list):
        sounds_list[i] = sorted(sound, key = lambda a: a[0], reverse=True)
    
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

    

    from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

    mid = MidiFile()
    numtracks = 8
    tracks = [MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack(),MidiTrack()]
    #track_instruments = ['piano','bass','brass','ensemble','organ','pipe','reed','strings']
    track_instruments = []
    #input_instruments = ['ensemble']*8
    for intr in range(8):
        if intr < len(input_instruments):
            track_instruments.append(input_instruments[intr])
        else:
            track_instruments.append(input_instruments[len(input_instruments)-1])

    #track_instruments = ['ensemble']*8

    tracks[0].append(MetaMessage('set_tempo', tempo=bpm2tempo(60)))

    #time_list = [[120,0]]*25
    time_list = rhythm
    print("time_list:", time_list)
    #time_list = [[80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
    #             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
    #             [80,0],[40,0],[120,0],[120,0],[120,0],[120,0],[120,0],
    #             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],]

    with open(f'{NAME}\\rhythm.json', 'w') as f:
        json.dump(time_list, f)

    print("len(sounds_list):", len(sounds_list))
    for t_i, sound in enumerate(sounds_list):    
        
        active_notes = []
        
        sorted_chords = sorted(sound, key = lambda a: a[0], reverse=True)
        max_chord_prob = sorted_chords[0][0]
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
                

                vol = (note_prob/max_note_prob)*(chord_prob/max_chord_prob)
                
                note = note_map(n)
                #note = round_to_scale(n, scale=C_MAJ)
                #notelist = []
                #note = notelist[n%len(notelist)]
                vol_128 = round(127*(vol))
                #instrument = round(127*(angle + np.pi)/(2*np.pi))
                instruments = track_instruments[trackno]
                instrument = instruments[round((len(track_instruments[trackno])-1)*((angle/(2*np.pi)) % 1))]



                #instrument = 1

                
                track.append(Message('program_change', program=instrument, time=0))
                track.append(Message('note_on', note=note, velocity=vol_128, time=0))
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

    import time
    print("converting midi to mp3...")
    time.sleep(3)










    #print(string)
    #directories = os.system(command_string)


    def plot_quantum_state(input_probability_vector, angle_vector, main_title=None, fig_title=None, save=None):
        num_figs = 8
        input_length = input_probability_vector.shape[1]

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
            files = glob.glob(target_folder + '\\*.png')
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

    num_frames = len(sound_list)

    input_probability_vector = np.zeros((8, len(sound_list[0][0][2])))
    angle_vector = np.zeros((8, len(sound_list[0][0][2])))

    for i, sound_data in enumerate(sound_list):
        for j in range(8):
            if j < len(sound_data):
                input_probability_vector[j, :] = np.array(sound_list[i][j][2]) * sound_list[i][j][0]
                angle_vector[j, :] = sound_list[i][j][3]


        plot_quantum_state(input_probability_vector, angle_vector, save=True)
        
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
    file_tuples = []
    for file in files:
        num = int(os.path.splitext(file)[0].lstrip(target_folder + '\\frame_'))
        file_tuples.append((num, file))
    file_tuples = sorted(file_tuples)
    files = [x[1] for x in file_tuples]
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

def get_instruments(instruments_name):
    instrument_dict = {'piano': list(range(1,9)),
                    'tuned_perc': list(range(9,17)),
                    'organ': list(range(17,25)),
                    'guitar': list(range(25,33)),
                    'bass': list(range(33,41)),
                    'strings': list(range(41,49)),
                    'ensemble': list(range(49,57)),
                    'brass': list(range(57,65)),
                    'reed': list(range(65,73)),
                    'pipe': list(range(73,81)),
                    'windband': [74,69,72,67,57,58,71,59]}
    return instrument_dict[instruments_name]