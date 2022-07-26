

import qiskit
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator
qiskit.__qiskit_version__
import numpy as np
import time


import numpy as np
import os
import glob

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
import qiskit.visualization.qcstyle as qiskit_viz_style

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

def make_music_midi(qc, name, rhythm, single_qubit_error, two_qubit_error, input_instruments, note_map = chromatic_middle_c):
    """ Simulates the quantum circuit (with provided errors) and samples the state at every inserted barrier time step and converts the state info to a midi file. 
    Args:
        qc: The qiskit QuantumCircuit.
        name: The name of the midi file to output to a folder in the working directory with the same name. The folder is created if it does not exist.
        rhythm: The sound length and post-rest times in units of ticks (480 ticks is 1 second) List[Tuple[int soundLength, int soundRest]]
        single_qubit_error: Amount of depolarisation error to add to each single-qubit gate.
        two_qubit_error: Amount of depolarisation error to add to each CNOT gate.
        input_instruments: The collections of instruments for each pure state in the mixed state (up to 8 collections). 
            Computational basis state phase determines which instrument from the collection is used. List[List[int intrument_index]]
        note_map: Converts state number to a note number where 60 is middle C. Map[int -> int]
    """

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
    simulator_zeronoise = AerSimulator()

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
    
    result = simulator.run(circ).result()
    result_zeronoise = simulator_zeronoise.run(circ).result()

    rhos = []
    for i in range(barrier_count):
        rhos.append(result.data(0)[f'rho{i}'])
        
    rhos_zeronoise = []
    for i in range(barrier_count):
        rhos_zeronoise.append(result_zeronoise.data(0)[f'rho{i}'])
    
    fidelity_list = [qi.state_fidelity(rho,rhos_zeronoise[i]) for i,rho in enumerate(rhos)]

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
    with open(f'{NAME}/sounds_list.json', 'w') as f:
        json.dump(sounds_list, f)
     
    with open(f'{NAME}/fidelity_list.json', 'w') as f:
        json.dump(fidelity_list, f)

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

    #for track in tracks:
    #    track.append(Message('control_change', program=81, control=68, value=0))
    #    track.append(Message('control_change', program=80, control=68, value=0))

    #time_list = [[120,0]]*25
    time_list = rhythm
    print("time_list:", time_list)
    #time_list = [[80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
    #             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
    #             [80,0],[40,0],[120,0],[120,0],[120,0],[120,0],[120,0],
    #             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],]

    with open(f'{NAME}/rhythm.json', 'w') as f:
        json.dump(time_list, f)

    print("len(sounds_list):", len(sounds_list))

    for t_i, sound in enumerate(sounds_list):    
        
        active_notes = []
        
        sorted_chords = sorted(sound, key = lambda a: a[0], reverse=True)
        max_chord_prob = sorted_chords[0][0]
        for trackno in range(numtracks):
            track = tracks[trackno]

            if trackno >= len(sorted_chords):
                track.append(Message('note_on', channel=trackno, note=0, velocity=0, time=0))
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
                #instrument = instruments[0]
                angle = angle % 2*np.pi
                instrument = instruments[int((len(instruments)-1) * (angle)/(2*np.pi))]

                
                phase = int(127*(angle)/(2*np.pi))
                #print("angle:", angle)
                #print("phase", phase)
                #instrument = 1

                
                track.append(Message('program_change', channel=trackno, program=instrument, time=0))
                #track.append(Message('control_change', channel=trackno, control=8, value=phase, time=0))
                track.append(Message('note_on', channel=trackno, note=note, velocity=vol_128, time=0))
                active_notes.append((trackno,note))
        
        
        for trackno, track in enumerate(tracks):   
            track.append(Message('note_on', channel=trackno, note=0, velocity=0, time=time_list[t_i][0]))
            track.append(Message('note_off', channel=trackno, note=0, velocity=0, time=0))
        
        for trackno, note in active_notes:
            track = tracks[trackno]
            track.append(Message('note_off', channel=trackno, note=note, velocity=0, time=0))
            
        for trackno, track in enumerate(tracks): 
            track.append(Message('note_on', channel=trackno, note=0, velocity=0, time=time_list[t_i][1]))
            track.append(Message('note_off', channel=trackno, note=0, velocity=0, time=0))
                            
                            
    for track in tracks:
        mid.tracks.append(track)
        
    midi_filename = f'{NAME}/{NAME}'
    mid.save(midi_filename + ".mid")

    return mid

def convert_midi_to_mp3_vlc(midi_filename_no_ext, wait_time = 3):
    """ Uses headless VLC to convert a midi file to mp3 in the working directory.
    Args:
        midi_filename_no_ext: the name of the midi file in the working dir.
        wait_time: The amount of time to wait after the VLC process has started. Used to make sure the process is finished before continuing execution.
    """

    string = 'vlc ' + midi_filename_no_ext + '.mid -I dummy --no-sout-video --sout-audio --no-sout-rtp-sap --no-sout-standard-sap --ttl=1 --sout-keep --sout "#transcode{acodec=mp3,ab=256}:std{access=file,mux=dummy,dst=./' + midi_filename_no_ext + '.mp3}"'
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
    print("Converting " + midi_filename_no_ext + ".mid midi to " + midi_filename_no_ext + ".mp3...")
    time.sleep(wait_time)

def convert_midi_to_wav_vlc(midi_filename_no_ext, wait_time = 3):
    """ Uses headless VLC to convert a midi file to wav in the working directory.
    Args:
        midi_filename_no_ext: the name of the midi file in the working dir.
        wait_time: The amount of time to wait after the VLC process has started. Used to make sure the process is finished before continuing execution.
    """

    string = 'vlc ' + midi_filename_no_ext + '.mid -I dummy --no-sout-video --sout-audio --no-sout-rtp-sap --no-sout-standard-sap --ttl=1 --synth-polyphony="65535 " --sout-keep --sout "#transcode{acodec=s16l,channels=2}:std{access=file,mux=wav,dst=./' + midi_filename_no_ext + '.wav}"'
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
    print("Converting " + midi_filename_no_ext + ".mid midi to " + midi_filename_no_ext + ".wav...")
    time.sleep(wait_time)

def convert_midi_to_wav_timidity(midi_filename_no_ext, wait_time = 3):
    """ Uses timidity++ to convert a midi file to wav in the working directory.
    Args:
        midi_filename_no_ext: the name of the midi file in the working dir.
        wait_time: The amount of time to wait after the VLC process has started. Used to make sure the process is finished before continuing execution.
    """

    #string = 'vlc ' + midi_filename_no_ext + '.mid -I dummy --no-sout-video --sout-audio --no-sout-rtp-sap --no-sout-standard-sap --ttl=1 --synth-polyphony="65535 " --sout-keep --sout "#transcode{acodec=s16l,channels=2}:std{access=file,mux=wav,dst=./' + midi_filename_no_ext + '.wav}"'
    
    # documentation found here: https://www.mankier.com/1/timidity#Input_File
    options = []
    options.append("-Ow")
    options.append("-A,120")
    options.append("--no-anti-alias") # anti-aliasing seems to cause some crackling
    options.append("--mod-wheel")
    options.append("--portamento")
    options.append("--vibrato")
    options.append("--no-ch-pressure")
    options.append("--mod-envelope")
    options.append("--trace-text-meta")
    options.append("--overlap-voice")
    #options.append("--temper-control")
    options.append("--default-bank=0")
    options.append("--default-program=0")
    options.append("--delay=d,0") #d: disabled, l: left, r: right, b: swap l&r
    #d: disabled, 
    # n: enable MIDI chorus effect control, 
    # s: surround sound, chorus detuned to a lesser degree. 
    # last number is chorus level 0-127
    options.append("--chorus=n,64")
    #d: disabled, 
    # n: enable MIDI reverb effect control, 
    # g: global reverb effect, 
    # f: Freeverb MIDI reverb effect control, 
    # G: global, Freeverb effect. 
    # num 1: reverb level 0-127
    # num 2: reverb scaleroom [0,1], roomsize = C * scaleroom + offsetroom, where C is reverberation character
    # num 3: reverb offsetroom [0,1]
    # num 4: reverb factor for pre-delay time of reverberation in percent
    #options.append("--reverb=f,60,0.28,0.7,100")
    options.append("--reverb=f,40,0.28,0.7,100")
    options.append("--voice-lpf=c") # d: disable, c: Chamberlin resonant LPF (12dB/oct), m: Moog resonant low-pass VCF (24dB/oct)
    options.append("--noise-shaping=4") # 0: no shaping, 1: trad, 2: Overdrive-like soft-clipping + new noise shaping, 3: Tube-amplifier-like soft-clipping + new noise shaping, 4: New noise shaping
    options.append("--resample=5") # 0-5, 5 is highest quality
    options.append("--voice-queue=0")
    #options.append("--fast-decay") # 0 means no voices will be killed even when there's a delay due to so many in the queue
    options.append("--decay-time=0")
    #options.append("-R 100")
    options.append("--interpolation=gauss")
    options.append("-EFresamp=34") # for interpolation gauss: 0-34
    options.append("--output-stereo")
    options.append("--output-24bit")
    options.append("--polyphony=16000")
    options.append("--sampling-freq=44100")
    options.append("--audio-buffer=5/100")
    options.append("--volume-curve=1.661") # (regular: 0, linear: 1, ideal: ~1.661, GS: ~2)
    #options.append("--module=4")
    options_string = ""
    for option in options:
        options_string += option + " "

    if os.name == 'nt':
        string = '"TiMidity_Win\\timidity.exe" ' + options_string + '-o ' + midi_filename_no_ext + '.wav ' + midi_filename_no_ext + '.mid'
    else:
        string = 'timidity ' + options_string + '-o ' + midi_filename_no_ext + '.wav ' + midi_filename_no_ext + '.mid'
    
    command_string = f"{string}"

    def run_timidity():
        import os
        #print(string)
        directories = os.system(command_string)

    import threading
    t = threading.Thread(target=run_timidity,name="timidity++",args=())
    t.daemon = True
    t.start()

    import time
    print("Converting " + midi_filename_no_ext + ".mid midi to " + midi_filename_no_ext + ".wav using TiMidity++...")
    time.sleep(wait_time)


def make_music_video(qc, name, rhythm, single_qubit_error, two_qubit_error, input_instruments, note_map = chromatic_middle_c, invert_colours = False, fps=60):
    """ Simulates the quantum circuit (with provided errors) and samples the state at every inserted barrier time step and converts the state info to a music video file (.avi). 
    Args:
        qc: The qiskit QuantumCircuit.
        name: The name of the midi file to output to a folder in the working directory with the same name. The folder is created if it does not exist.
        rhythm: The sound length and post-rest times in units of ticks (480 ticks is 1 second) List[Tuple[int soundLength, int soundRest]]
        single_qubit_error: Amount of depolarisation error to add to each single-qubit gate.
        two_qubit_error: Amount of depolarisation error to add to each CNOT gate.
        input_instruments: The collections of instruments for each pure state in the mixed state (up to 8 collections). 
            Computational basis state phase determines which instrument from the collection is used. List[List[int intrument_index]]
        note_map: Converts state number to a note number where 60 is middle C. Map[int -> int]
    """
    
    make_music_midi(qc, name, rhythm, single_qubit_error, two_qubit_error, input_instruments, note_map=note_map)
    convert_midi_to_wav_timidity(f'{name}/{name}', wait_time = 3)
    make_video(qc, name, rhythm, single_qubit_error, two_qubit_error, input_instruments, note_map = note_map, invert_colours = invert_colours, fps = fps)

def make_video(qc, name, rhythm, single_qubit_error, two_qubit_error, input_instruments, note_map = chromatic_middle_c, invert_colours = False, fps=60):
    
    import matplotlib
    import matplotlib.pylab as plt
    from matplotlib.pylab import cm, mpl
    import moviepy.editor as mpy
    from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
    from moviepy.editor import ImageClip, concatenate, clips_array
    from moviepy.video.fx import invert_colors, crop, fadeout
    from moviepy.editor import CompositeVideoClip
    import copy
    
    NAME = name
    target_folder = NAME
    circuit_layers_per_line = 50
    
    dag = circuit_to_dag(qc)

    barrier_iter = 0
    for node in dag.topological_op_nodes():
        if node.name == "barrier":
            barrier_iter += 1
    barrier_count = barrier_iter

    circuit_list = []
    current_circuit = QuantumCircuit(len(dag.qubits))
    for node in dag.topological_op_nodes():
        if node.name == "barrier":
            circuit_list.append(current_circuit)
            current_circuit = QuantumCircuit(len(dag.qubits))
        if node.name != "measure" and node.name != "barrier":
            current_circuit.append(node.op, node.qargs, node.cargs)
    circuit_list.append(current_circuit)

    empty_circuit = QuantumCircuit(len(dag.qubits))
    empty_circuit.draw(filename=f'./{NAME}/partial_circ_empty.png',output="mpl", fold=-1)

    barrier_circuit = QuantumCircuit(len(dag.qubits))
    barrier_circuit.barrier()
    barrier_circuit.draw(filename=f'./{NAME}/partial_circ_barrier.png',output="mpl", fold=-1)

    for i, partial_circ in enumerate(circuit_list):
        partial_circ.draw(filename=f'./{NAME}/partial_circ_{i}.png',output="mpl", fold=-1)

    qc.draw(filename=f'./{NAME}/circuit.png',output="mpl", fold=circuit_layers_per_line)

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
            files = glob.glob(target_folder + '/*.png')
            filename = target_folder + '/frame_' + str(len(files)) + '.png'
            plt.savefig(filename)
            plt.close('all')
        return 0

    import json

    with open(target_folder + '/sounds_list.json') as json_file:
        sound_list = json.load(json_file)

    with open(target_folder + '/rhythm.json') as json_file:
        rhythm = json.load(json_file)

    with open(target_folder + '/fidelity_list.json') as json_file:
        fidelity_list = json.load(json_file)

    print("rhythm: ", rhythm)

    files = glob.glob(target_folder + '/frame_*')
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

    clips = []

    total_time = 0
    files = glob.glob(target_folder + '/frame_*')
    iter = 0
    file_tuples = []
    for file in files:
        file = file.replace("\\", "/")
        num = int(os.path.splitext(file)[0].lstrip(target_folder + '/frame_'))
        file_tuples.append((num, file))
    file_tuples = sorted(file_tuples)
    files = [x[1] for x in file_tuples]
    for file in files:
        time = (rhythm[iter][0] + rhythm[iter][1]) / 480.0
        clips.append(ImageClip(file).set_duration(time))
        total_time += time
        iter += 1

    video = concatenate(clips, method="compose")

    bg_color = [0xFF, 0xFF, 0xFF]
    bg_color_inverse = [0x00, 0x00, 0x00]
    if invert_colours == True:
        video = invert_colors.invert_colors(video)
        bg_color = [0x00, 0x00, 0x00]
        bg_color_inverse = [0xFF, 0xFF, 0xFF]

    partial_circ_image_clips = []
    positions_x = []
    accumulated_width = 0
    barrier_image_width = 0
    image_barrier_clip = ImageClip(target_folder + f'/partial_circ_barrier.png').set_duration(video.duration)
    if image_barrier_clip.size[0] > 156:
        image_barrier_clip = crop.crop(image_barrier_clip, x1=133, x2=image_barrier_clip.size[0]-23)
    barrier_image_width = image_barrier_clip.size[0]

    # create image clip same size as barrier.
    image_empty_clip = ImageClip(target_folder + f'/partial_circ_empty.png').set_duration(video.duration)
    if image_empty_clip.size[0] > 156:
        image_empty_clip = crop.crop(image_empty_clip, x1=133, x2=image_empty_clip.size[0]-23)
    image_empty_clip_array = clips_array([[image_empty_clip, image_empty_clip, image_empty_clip]], bg_color=[0xFF, 0xFF, 0xFF])
    image_empty_clip_array = crop.crop(image_empty_clip_array, x1=0, x2=barrier_image_width)
    

    image_empty_clip = ImageClip(target_folder + f'/partial_circ_barrier.png').set_duration(video.duration)
    if image_empty_clip.size[0] > 156:
        image_empty_clip = crop.crop(image_empty_clip, x1=133, x2=image_empty_clip.size[0]-23)
    
    barrier_only_clip = crop.crop(image_empty_clip, x1=35, x2=72)
    barrier_only_clip = barrier_only_clip.margin(left=35, right=72, color=[0xFF, 0xFF, 0xFF])

    barrier_only_clip_mask = barrier_only_clip.to_mask()
    barrier_only_clip_mask = barrier_only_clip_mask.fl_image( lambda pic: filter_colour_round_with_threshold(pic, threshold=0.9, colour_dark=0.0, colour_light=1.0))
    barrier_only_clip_mask = invert_colors.invert_colors(barrier_only_clip_mask)
    #barrier_only_clip.save_frame("./barrier_only_clip.png", t=0)


    #barrier_only_clip = barrier_only_clip.fl_image( lambda pic: filter_color_multiply(pic, [0, 0, 255]))
    barrier_only_clip = barrier_only_clip.fl_image( lambda pic: filter_color_blend(pic, [255, 255, 255], 0.3))
    barrier_only_clip = barrier_only_clip.add_mask() # might be able to remove this line
    barrier_only_clip = barrier_only_clip.set_mask(barrier_only_clip_mask)
    vertical_shrink = 0.05
    clip_height = barrier_only_clip.size[1]
    barrier_start_y = 43
    barrier_end_y = 25
    height = clip_height - barrier_start_y - barrier_end_y
    barrier_only_clip = crop.crop(barrier_only_clip, y1 = int((vertical_shrink/2.0) * height) + barrier_start_y, y2 = int((1.0 - vertical_shrink/2.0) * height) + barrier_start_y)
    barrier_only_clip = barrier_only_clip.resize(newsize=(int(0.5 * barrier_only_clip.size[0]), int(barrier_only_clip.size[1])))
    
    #image_empty_clip_array = CompositeVideoClip([image_empty_clip_array, barrier_only_clip.set_position(("center", int((vertical_shrink/2.0) * height) + barrier_start_y))], use_bgclip=True)

    for i, partial_circ in enumerate(circuit_list):
        
        new_image_clip = ImageClip(target_folder + f'/partial_circ_{i}.png').set_duration(video.duration)
        if new_image_clip.size[0] > 156:
            new_image_clip = crop.crop(new_image_clip, x1=133, x2=new_image_clip.size[0]-23)
        
        if i != len(circuit_list)-1:
            accumulated_width += new_image_clip.size[0] + barrier_image_width
        else:
            accumulated_width += new_image_clip.size[0]

        positions_x.append(accumulated_width)
        partial_circ_image_clips.append(new_image_clip)
        
    
    all_clips = []
    for i in range(len(partial_circ_image_clips)):
        all_clips.append(partial_circ_image_clips[i])
        if i != len(partial_circ_image_clips)-1:
            all_clips.append(image_empty_clip_array)

    circ_clip_arr = clips_array([[x for x in all_clips]], bg_color=[0xFF, 0xFF, 0xFF])
    
    
    circ_clip_arr.fps = fps

    
    composited_with_barrier_clips = []
    composited_with_barrier_clips.append(circ_clip_arr)
    note_accumulated_info = [] # accumulated time, note length, note rest
    accumulated_time = 0
    
    for iter in range(len(rhythm)):
        #new_barrier_clip = image_barrier_clip.set_start(accumulated_time).set_end(min(video.duration, accumulated_time + 1 / 4.0)).set_position((positions_x[iter]-barrier_image_width, 0))
        note_length = rhythm[iter][0] / 480.0
        new_barrier_clip = image_barrier_clip.set_start(0).set_end(min(accumulated_time, video.duration)).set_position((positions_x[iter]-barrier_image_width, 0))
        note_accumulated_info.append((accumulated_time, rhythm[iter][0] / 480.0, rhythm[iter][1] / 480.0))
        accumulated_time += (rhythm[iter][0] + rhythm[iter][1]) / 480.0
        #new_barrier_clip.add_mask()
        #new_barrier_clip = new_barrier_clip.crossfadeout(note_length)
        composited_with_barrier_clips.append(new_barrier_clip)

    video_duration = video.duration
    video_size = video.size



    circ_clip_arr = CompositeVideoClip(composited_with_barrier_clips)
    
    vertical_scale = 1080 / float(video.size[1] + circ_clip_arr.size[1])
    circ_clip_target_width = int(1920 / vertical_scale)
    
    clip_orig_x = circ_clip_arr.size[0]
    circ_clip_arr = circ_clip_arr.margin(left=circ_clip_target_width, right=circ_clip_target_width, color=[0xFF, 0xFF, 0xFF])
    #circ_clip_arr = crop.crop(circ_clip_arr, x1 = -clip_orig_x, x2 = 2 * clip_orig_x)
    
    #circ_clip_arr.save_frame("contatenated_circ.png", t = 0.1)
    h = circ_clip_arr.h
    w = circ_clip_target_width

    

    def f(gf,t):
        x_start = w/2
        time_fraction = t / video_duration
        accumulated_time = 0
        prev_accumulated_time = 0
        for iter in range(len(rhythm)):
            accumulated_time += (rhythm[iter][0] + rhythm[iter][1]) / 480.0
            if t < accumulated_time:
                break
            prev_accumulated_time = accumulated_time
        target_iter = iter
        prev_iter = iter - 1
        prev_time = prev_accumulated_time
        target_time = accumulated_time
        if iter != len(rhythm)-1:
            target_pos = positions_x[iter+1] - barrier_image_width / 2
        else:
            target_pos = positions_x[iter+1]
        prev_pos = positions_x[iter] - barrier_image_width / 2

        x = int(x_start + (prev_pos + (target_pos - prev_pos)*(t - prev_time)/(target_time - prev_time)))
        y = 0
        return gf(t)[y:y+h, x:x+w]

    circ_clip_arr = circ_clip_arr.fl(f, apply_to = "mask")
    
    
    if invert_colours == True:
        circ_clip_arr = invert_colors.invert_colors(circ_clip_arr)


    #clip_arr = clips_array([[circuit_video.resize(circuit_rescale)], [video]], bg_color=bg_color)
    clip_arr = clips_array([[circ_clip_arr], [video]], bg_color=bg_color)
    
    files = glob.glob(target_folder + '/*.wav')
    video_final = clip_arr
    if len(files) > 0:
        audio_clip = mpy.AudioFileClip(files[0], nbytes=4, fps=44100)
        print("audio_clip:", audio_clip)
        arr = audio_clip.to_soundarray(nbytes=4)
        print("arr:", arr)
        audio_clip_new = AudioArrayClip(arr[0:int(44100 * total_time)], fps=44100)
        video_final = clip_arr.set_audio(audio_clip_new)

    video_final = crop.crop(video_final, x1=int(clip_arr.size[0]/2-circ_clip_target_width/2), x2=int(clip_arr.size[0]/2+circ_clip_target_width/2))
    
    vertical_scale = 1080 / video_final.size[1]
    if video_final.size[1] / 1080 > video_final.size[0] / 1920:
        video_final = video_final.resize(height=1080)
        left_margin = int((1920 - video_final.size[0])/2)
        video_final = video_final.margin(left=left_margin, right=(1920 - video_final.size[0] - left_margin), color=bg_color)
    else:
        video_final = video_final.resize(width=1920)
        video_final = video_final.margin(top=(1080 - video_final.size[1])/2, bottom=(1080 - video_final.size[1])/2, color=bg_color)
    
    highlight_time = 1.0 / 8.0
    highlight_fade_time = 1.0 / 16.0

    cmap_jet = cm.get_cmap('jet')
    cmap_coolwarm = cm.get_cmap('coolwarm')
    cmap_rainbow = cm.get_cmap('rainbow')

    def draw_needle(get_frame, t):
        """Draw a rectangle in the frame"""
        # change (top, bottom, left, right) to your coordinates
        top = 1
        bottom = int(vertical_scale*circ_clip_arr.size[1] - 1 + vertical_scale*(barrier_start_y - barrier_end_y))
        left = int(video_final.size[0] / 2 - 9)
        right = int(video_final.size[0] / 2 + 9)

        current_note_info = note_accumulated_info[len(note_accumulated_info)-1]
        fidelity = fidelity_list[len(note_accumulated_info)-1]
        for i, note_info in enumerate(note_accumulated_info):
            if note_info[0] > t:
                current_note_info = note_accumulated_info[i-1]
                fidelity = fidelity_list[i-1]
                break
            
        
        frame = get_frame(t)

        time_since_note_played = t - current_note_info[0]
        note_remaining_play_time = max(0.0, current_note_info[1] - time_since_note_played)
        time_to_start_fade = 0.0 #highlight_time - highlight_fade_time
        time_to_stop_fade = highlight_time
        highlight_fade_time = current_note_info[1]
        
        idle_colour = [127, 127, 127] #blend_colour(bg_color_inverse, [127, 127, 127], 0.8)
        lerp_time = (time_since_note_played - time_to_start_fade) / highlight_fade_time
        scale = (2.0 * (1 - fidelity) - 1.0) # -1 to 1
        scale = np.tanh(scale) / np.tanh(1)
        #scale = np.arcsinh(scale) / np.arcsinh(1)
        scale = (scale + 1.0) / 2.0

        highlight_colour = [int(255 * cmap_jet(scale)[i]) for i in range(3)]

        lerp_colour = [int(x) for x in ease_out(highlight_colour, idle_colour, lerp_time)]
        frame[top: top+3, left: right] = lerp_colour
        frame[bottom-3: bottom, left: right] = lerp_colour
        frame[top+3: bottom, left: left+3] = lerp_colour
        frame[top+3: bottom, right-3: right] = lerp_colour
        
        return frame

    video_final = video_final.fl(draw_needle)

    #video_final.save_frame(target_folder + '/' +"save_frame_0.png", t=0.0)
    #video_final.save_frame(target_folder + '/' +"save_frame_1.png", t=1.0)
    #video_final.save_frame(target_folder + '/' +"save_frame_fading.png", t=1.0 - highlight_fade_time / 2.0)

    #def supersample(clip, d, nframes):
    #    """ Replaces each frame at time t by the mean of `nframes` equally spaced frames
    #    taken in the interval [t-d, t+d]. This results in motion blur. 
    #    credit: Zulko https://gist.github.com/Zulko/f90674b2e64c5600370e"""
    #    def fl(gf, t):
    #        tt = np.linspace(t-d, t+d, nframes) 
    #        avg = np.mean(1.0*np.array([gf(t_) for t_ in tt]),axis=0)
    #        return avg.astype("uint8")
    #    return clip.fl(fl)
#
    #video_final = supersample(video_final, d=0.008, nframes=3)
    # preset options (speed vs filesize): ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
    video_final.write_videofile(target_folder + '/' + target_folder + '.mp4', preset='ultrafast', fps=fps, codec='mpeg4', audio_fps=44100, audio_bitrate="512K", audio_nbytes=4, ffmpeg_params=["-b:v", "12000K", "-b:a", "512K"])

    files = glob.glob(target_folder + '/*.mp4')
#     for file in files:
#         os.remove(file)

def get_instruments(instruments_name):
    '''
    Instrument collections:
        'piano': list(range(1,9)),
        'tuned_perc': list(range(9,17)),
        'organ': list(range(17,25)),
        'guitar': list(range(25,33)),
        'bass': list(range(33,41)),
        'strings': list(range(41,49)),
        'ensemble': list(range(49,57)),
        'brass': list(range(57,65)),
        'reed': list(range(65,73)),
        'pipe': list(range(73,81)),
        'synth_lead': list(range(81,89)),
        'synth_pad': list(range(89,97)),
        'synth_effects': list(range(97,105)),
        'ethnic': list(range(105,113)),
        'percussive': list(range(113,121)),
        'sound_effects': list(range(121,128)),
        'windband': [74,69,72,67,57,58,71,59]
    '''
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
                    'synth_lead': list(range(81,89)),
                    'synth_pad': list(range(89,97)),
                    'synth_effects': list(range(97,105)),
                    'ethnic': list(range(105,113)),
                    'percussive': list(range(113,121)),
                    'sound_effects': list(range(121,128)),
                    'windband': [74,69,72,67,57,58,71,59]}
    return instrument_dict[instruments_name]

def filter_color_blend(pic, colour, alpha):
    """ assumes pic is 2d array of uint8. Returns weighted average of image pizel colours and given colour.
    """
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            pic[i][j] = blend_colour(pic[i][j], colour, alpha)
    
    return pic

def blend_colour(colour1, colour2, alpha):
    return [((1 - alpha) * colour1[i] + (alpha) * colour2[i]) for i in range(min(len(colour2), len(colour1)))]

def filter_color_multiply(pic, colour):
    """ assumes pic is 2d array of uint8. Returns weighted average of image pizel colours and given colour.
    """
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            pic[i][j] = [int(pic[i][j][k] * (colour[k]/255.0)) for k in range(min(len(colour), len(pic[i][j])))]
    
    return pic

def filter_colour_round_with_threshold(pic, threshold = 0.5, colour_dark = [0, 0, 0], colour_light = [255, 255, 255]):
        """ assumes pic is 2d array of uint8. Rounds each colour to black or white based on average RGB and threshold.
        """
        colour_array_dims = 0
        try:
            colour_array_dims = len(pic[0][0])
        except:
            colour_array_dims = 0

        if (colour_array_dims > 0): # pic is an RGB image (0-255)
            for i in range(len(pic)):
                for j in range(len(pic[i])):
                    average_rgb = np.mean([pic[i][j][k] for k in range(min(3, colour_array_dims))])
                    if average_rgb > threshold:
                        pic[i][j] = colour_light
                    else:
                        pic[i][j] = colour_dark
        else: # pic is a mask (0-1)
            for i in range(len(pic)):
                for j in range(len(pic[i])):
                    if pic[i][j] > threshold:
                        pic[i][j] = colour_light
                    else:
                        pic[i][j] = colour_dark
        return pic

def lerp(lerp_from, lerp_to, t):
    t = max(min(t, 1.0), 0.0)
    return [((1-t) * a) + (t * b) for a, b in zip(lerp_from, lerp_to)]

def ease_in(ease_from, ease_to, t):
    ''' slow at the beginning, fast at the end.'''
    t = max(min(t, 1.0), 0.0)
    scaled_t = 1 - math.cos(t * math.pi / 2.0)
    return [((1-scaled_t) * a) + (scaled_t * b) for a, b in zip(ease_from, ease_to)]

def ease_out(ease_from, ease_to, t):
    ''' slow at the beginning, fast at the end.'''
    t = max(min(t, 1.0), 0.0)
    scaled_t = math.sin(t * math.pi / 2.0)
    return [((1-scaled_t) * a) + (scaled_t * b) for a, b in zip(ease_from, ease_to)]
