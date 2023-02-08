# Methods relating to the generation of video from sampled density matrices
import qmuvi.musical_processing as musical_processing
import qmuvi.data_manager as data_manager

import os
import math
import collections.abc
import numpy as np
from typing import Any, AnyStr, List, Tuple, Union, Optional, Mapping
import qiskit
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

def filter_blend_colour(bitmap, colour, alpha):
    """ Blends the bitmap's pixels with the given colour using the alpha value (weighted average). 
        Assumes that bitmap is 2-dimensional. 
    Args:
        bitmap: 2d array of colour channel values
        colour: sequence of colour channel values
        alpha: float between 0 and 1. 0 = bitmap, 1 = colour
    """
    for i in range(len(bitmap)):
        for j in range(len(bitmap[i])):
            bitmap[i][j] = blend_colour(bitmap[i][j], colour, alpha)

    return bitmap

def blend_colour(colour1, colour2, alpha) -> collections.abc.Sequence:
    """ blends the two colours together using the alpha value (weighted average). 
        The returned colour contains as many channels as the colour with the least channels. E.g. RG + RGBA = RG
    Args: 
        colour1: sequence of colour channel values
        colour2: sequence of colour channel values
        alpha: float between 0 and 1. 0 = colour1, 1 = colour2
    """
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
    ''' fast at the beginning, slow at the end.'''
    t = max(min(t, 1.0), 0.0)
    scaled_t = math.sin(t * math.pi / 2.0)

    return [((1-scaled_t) * a) + (scaled_t * b) for a, b in zip(ease_from, ease_to)]

def invert_cmap(cmap):
    from matplotlib.colors import ListedColormap
    newcolors = cmap(np.linspace(0, 1, 256))
    for i in range(256):
        newcolors[i, :] = np.array([1-cmap(i/256)[0],1-cmap(i/256)[1],1-cmap(i/256)[2],1])
        
    return ListedColormap(newcolors)

def generate_video_from_data(quantum_circuit: qiskit.QuantumCircuit, 
                             output_manager: data_manager.DataManager, 
                             rhythm: List[Tuple[int, int]], 
                             phase_instruments: List[List[int]] = [list(range(81, 89))], 
                             invert_colours: bool = False, 
                             fps: int = 60, 
                             vpr: Optional[float] = None, 
                             smooth_transitions: bool = True, 
                             phase_marker: bool = True, 
                             probability_distribution_only: bool = False
                             ):
    """ Samples the quantum circuit at every barrier and uses the state properties to create a silent video (.mp4). No music is generated using this method.
    Args:
        quantum_circuit: The qiskit QuantumCircuit.
        output_manager: The data manager to use for saving the video and all of its pieces.
        rhythm: The sound length and post-rest times in units of ticks (480 ticks is 1 second) List[Tuple[int soundLength, int soundRest]]
        phase_instruments: The collections of instruments for each pure state in the mixed state (up to 8 collections).
            Computational basis state phase determines which instrument from the collection is used. List[List[int intrument_index]]
        invert_colours: Whether to render the video in dark mode. (default: False) Bool
        fps: The frames per second of the output video. (default: 60) Int
        vpr: Propotion of vertical space that the circuit will occupy. Float (default: 1/3)
        smooth_transitions: Whether to smoothly animate between histogram frames. Significantly increased render time. (default: False) Bool
        phase_marker: Whether to draw lines on the phase wheel indicating phases of the primary pure state.
        probability_distribution_only: Whether to only plot the basis state probabilities. (default: False) Bool
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import moviepy.editor as mpy
    from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
    from moviepy.editor import ImageClip, concatenate, clips_array
    from moviepy.video.fx import invert_colors, crop, freeze
    from moviepy.editor import CompositeVideoClip, VideoClip
    from matplotlib.lines import Line2D
    from moviepy.video.io.bindings import mplfig_to_npimage
    import re

    if vpr == None:
        def vpr(n): return 1/3
    else:
        def vpr(n): return vpr

    sounds_list = output_manager.load_json('sounds_list.json')
    #rhythm = output_manager.load_json('rhythm.json')
    fidelity_list = output_manager.load_json('fidelity_list.json')
    meas_probs_list = output_manager.load_json('meas_probs_list.json')

    # format loaded data
    for sound_index in range(len(sounds_list)):
        for pure_state_info_index in range(len(sounds_list[sound_index])):
            # pure_state_prob: float
            sounds_list[sound_index][pure_state_info_index][0] = float(sounds_list[sound_index][pure_state_info_index][0])
            # pure_state_info: Dict[int basis_state_number, Tuple[float basis_state_prob, float basis_state_angle]], # where (basis_state_prob > eps)
            pure_state_info = {}
            for basis_state_number in sounds_list[sound_index][pure_state_info_index][1].keys():
                pure_state_info[int(basis_state_number)] = (float(sounds_list[sound_index][pure_state_info_index][1][basis_state_number][0]), float(sounds_list[sound_index][pure_state_info_index][1][basis_state_number][1]))
            sounds_list[sound_index][pure_state_info_index][1] = pure_state_info
            # all_basis_state_probs: List[float basis_state_prob]
            for basis_state_index in range(len(sounds_list[sound_index][pure_state_info_index][2])):
                sounds_list[sound_index][pure_state_info_index][2][basis_state_index] = float(sounds_list[sound_index][pure_state_info_index][2][basis_state_index])
            # all_basis_state_angles: List[float basis_state_angle]
            for basis_state_index in range(len(sounds_list[sound_index][pure_state_info_index][3])):
                sounds_list[sound_index][pure_state_info_index][3][basis_state_index] = float(sounds_list[sound_index][pure_state_info_index][3][basis_state_index])

    #for rhythm_index in range(len(rhythm)):
    #    rhythm[rhythm_index] = (int(rhythm[rhythm_index][0]), int(rhythm[rhythm_index][1]))

    zero_noise = True
    for i in range(len(sounds_list)):
        if len(sounds_list[i]) > 1:
            zero_noise = False
            break

    dag = circuit_to_dag(quantum_circuit)
    qubit_count = len(dag.qubits)

    circuit_layers_per_line = 50
    quantum_circuit.draw(filename=output_manager.get_path(f'circuit.png'), output="mpl", fold=circuit_layers_per_line)

    # create partial circuits separated by barrier gates
    partial_circuit_list = []
    current_circuit = QuantumCircuit(qubit_count)
    for node in dag.topological_op_nodes():
        if node.name == "barrier":
            partial_circuit_list.append(current_circuit)
            current_circuit = QuantumCircuit(qubit_count)
        if node.name != "measure" and node.name != "barrier":
            current_circuit.append(node.op, node.qargs, node.cargs)
    partial_circuit_list.append(current_circuit)
    for i, partial_circ in enumerate(partial_circuit_list):
        partial_circ.draw(filename=output_manager.get_path(f'partial_circ_{i}.png'), output="mpl", fold=-1)
    
    # an empty circuit is used to create space in the circuit
    empty_circuit = QuantumCircuit(qubit_count)
    empty_circuit.draw(filename=output_manager.get_path('partial_circ_empty.png'), output="mpl", fold=-1)

    # a barrier circuit is inserted between partial circuits
    barrier_circuit = QuantumCircuit(qubit_count)
    barrier_circuit.barrier()
    barrier_circuit.draw(filename=output_manager.get_path('partial_circ_barrier.png'), output="mpl", fold=-1)
    
    
    # define some colours
    cmap_jet = cm.get_cmap('jet')
    cmap_coolwarm = cm.get_cmap('coolwarm')
    cmap_rainbow = cm.get_cmap('rainbow')
    cmap_rvb = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "violet", "blue"])
    cmap_bvr = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "violet", "red"])
    tick_colour = [0.4, 0.4, 0.4, 1.0]


    def plot_quantum_state(input_probability_vector_list: List[np.ndarray], 
                           angle_vector_list: List[np.ndarray], 
                           meas_prob_vector_list: List[np.ndarray], 
                           plot_number: Union[int, float], 
                           interpolate: bool = False, 
                           save: bool = True, 
                           fig: Optional[matplotlib.figure.Figure] = None, 
                           zero_noise: bool = False, 
                           probability_distribution_only: bool = probability_distribution_only
                           ):
        '''
        Args:
            input_probability_vector_list: a list containing a 2d probability vector (pure_state_numbers x basis_state_numbers) for each sampled state
            angle_vector_list: a list containing a 2d phase angle vector (pure_state_numbers x basis_state_numbers) for each sampled state
            meas_prob_vector_list: a list containing a 1d basis-state measurement probability vector (basis_state_numbers) for each sampled state
            plot_number: the index of the sampled state to be the plotted. If interpolate is True, then this can be a decimal fraction
            interpolate: whether to interpolate plot numbers
            save: whether to save the plotted frame in the animation (.png)
            fig: an already created figure
            zero_noise: whether to only plot the data for the single pure state (no noise)
            probability_distribution_only: whether to only plot the basis-state probability distribution
        '''
        if interpolate == True:
            input_length = input_probability_vector_list[0].shape[1]
            
            # create new vectors that lerp between the sampled states with respect to plot_number
            input_probability_vector_start = input_probability_vector_list[math.floor(plot_number)]
            input_probability_vector_end = input_probability_vector_list[math.ceil(plot_number)]
            input_probability_vector = np.zeros((8, input_length))
            for i in range(input_probability_vector_start.shape[0]):
                input_probability_vector[i, :] = lerp(
                    input_probability_vector_start[i, :], input_probability_vector_end[i, :], plot_number - math.floor(plot_number))

            angle_vector_start = angle_vector_list[math.floor(plot_number)]
            angle_vector_end = angle_vector_list[math.ceil(plot_number)]
            angle_vector = np.zeros((8, input_length))
            for i in range(angle_vector.shape[0]):
                angle_vector[i, :] = lerp(angle_vector_start[i, :], angle_vector_end[i, :], plot_number - math.floor(plot_number))

            meas_prob_vector_start = meas_prob_vector_list[math.floor(plot_number)]
            meas_prob_vector_end = meas_prob_vector_list[math.ceil(plot_number)]
            meas_prob_vector = lerp(meas_prob_vector_start[:], meas_prob_vector_end[:], plot_number - math.floor(plot_number))

            # avoid animating the phases when the probability's start or end point is zero
            for i in range(angle_vector.shape[0]):
                for j in range(angle_vector.shape[1]):
                    if input_probability_vector_start[i, j] <= 0.0001 and input_probability_vector_end[i, j] > 0:
                        angle_vector[i, j] = angle_vector_end[i, j]
                    if input_probability_vector_start[i, j] > 0 and input_probability_vector_end[i, j] <= 0.0001:
                        angle_vector[i, j] = angle_vector_start[i, j]
        else:
            input_probability_vector = input_probability_vector_list[plot_number]
            angle_vector = angle_vector_list[plot_number]
            meas_prob_vector = meas_prob_vector_list[plot_number]
            input_length = input_probability_vector.shape[1]

        num_qubits = len(bin(input_length - 1)) - 2
        #num_qubits = (input_length - 1).bit_length()
        labels = []
        for x in range(input_length):
            label_term = bin(x).split('0b')[1]
            #label_term_2 = '0'*(num_qubits - len(label_term)) + label_term
            label_term_2 = label_term
            for y in range(num_qubits - len(label_term)):
                label_term_2 = '0' + label_term_2
            labels.append(label_term_2)
        cmap = cm.get_cmap('hsv')
        if invert_colours == True:
            cmap = invert_cmap(cmap)
        max_height = 2 * np.pi
        min_height = -np.pi
        x_values = [x for x in range(input_length)]
        if fig == None:
            fig = plt.figure(figsize=(20, (1 - vpr(qubit_count)) * 13.5))
        if probability_distribution_only:
            grid_spec = {
                "bottom": 0.1,
                "top": 0.95,
                "left": 0.07,
                "right": 0.99,
                "wspace": 0.05,
                "hspace": 0.09,
            }
            ax_dict = fig.subplot_mosaic(
                [
                    ["meas_probs", "meas_probs", "meas_probs", "meas_probs"],
                    ["meas_probs", "meas_probs", "meas_probs", "meas_probs"],
                ],
                gridspec_kw=grid_spec,
            )
            plots_order = ["meas_probs"]
        else:
            if zero_noise:
                grid_spec = {
                    "bottom": 0.1,
                    "top": 0.95,
                    "left": 0.07,
                    "right": 0.99,
                    "wspace": 0.05,
                    "hspace": 0.09,
                }
                ax_dict = fig.subplot_mosaic(
                    [
                        ["main", "main", "main", "main"],
                        ["main", "main", "main", "main"],
                    ],
                    gridspec_kw=grid_spec,
                )
                plots_order = ["main"]
            else:
                grid_spec = {
                    "bottom": 0.08,
                    "top": 0.95,
                    "left": 0.07,
                    "right": 0.99,
                    "wspace": 0.05,
                    "hspace": 0.09,
                }
                ax_dict = fig.subplot_mosaic(
                    [
                        ["pure_state_2", "main", "main", "pure_state_3"],
                        ["pure_state_4", "pure_state_5",
                            "pure_state_6", "pure_state_7"],
                    ],
                    gridspec_kw=grid_spec,
                )
                plots_order = ["main", "pure_state_2", "pure_state_3",
                               "pure_state_4", "pure_state_5", "pure_state_6", "pure_state_7"]
        for i, ax_name in enumerate(plots_order):
            ax_dict[ax_name].tick_params(axis='y', labelsize=20)
            if ax_name == "meas_probs":
                in_prob_vec = meas_prob_vector
            else:
                in_prob_vec = input_probability_vector[i, :]
            bar_list = ax_dict[ax_name].bar(x_values, in_prob_vec, width=0.5)
            ax_dict[ax_name].set_ylim([0, np.max(input_probability_vector)])
            if ax_name != "meas_probs":
                rgba = [cmap((k - min_height) / max_height) for k in angle_vector[i, :]]
                for x in range(input_length):
                    bar_list[x].set_color(rgba[x])
            ax_dict[ax_name].tick_params(axis='x', colors=tick_colour)
            ax_dict[ax_name].tick_params(axis='y', colors=tick_colour)
            ax_dict[ax_name].axes.xaxis.set_visible(False)
            ax_dict[ax_name].axes.yaxis.set_visible(False)
            if (zero_noise and ax_name == "main") or ax_name == "meas_probs":
                ax_dict[ax_name].axes.yaxis.set_visible(True)
            if ax_name == "pure_state_2" or ax_name == "pure_state_4":
                ax_dict[ax_name].axes.yaxis.set_visible(True)
            if ax_name == "main" or ax_name == "meas_probs":
                ax_dict[ax_name].set_xlim((-0.5, math.pow(2, num_qubits)-1+0.5))
                ax_dict[ax_name].axes.xaxis.set_visible(True)
                number_of_states = math.pow(2, num_qubits)
                if (zero_noise and num_qubits > 4) \
                        or ((not zero_noise) and num_qubits > 3) \
                        or (ax_name == "meas_probs" and num_qubits > 4):
                    x_ticks = [0]
                    x_ticks.append(int(number_of_states / 4))
                    x_ticks.append(int(2 * number_of_states / 4))
                    x_ticks.append(int(3 * number_of_states / 4))
                    x_ticks.append(int(number_of_states-1))
                else:
                    x_ticks = list(range(2**num_qubits))
                ax_dict[ax_name].set_xticks(x_ticks)
                if num_qubits < 7:
                    x_tick_labels = [bin(x)[2:].zfill(num_qubits)[::-1] for x in x_ticks]
                ax_dict[ax_name].set_xticklabels(x_tick_labels)
                # plt.xticks(fontsize=14)
                ax_dict[ax_name].tick_params(axis='x', labelsize=14)
        fig.text(0.01, (grid_spec["top"] - grid_spec["bottom"])/2 + grid_spec["bottom"],
                 'Probability', va='center', rotation='vertical', fontsize=20)
        fig.text((grid_spec["right"] - grid_spec["left"])/2 +
                 grid_spec["left"], 0.035, 'Quantum states', ha='center', fontsize=20)
        if save:
            plt.savefig(output_manager.get_path('frame_' + str(plot_number) + '.png'))
            plt.close('all')
        return fig

    def plot_info_panel(plot_number, fidelity, prob_vec, angles_vec, probability_distribution_only=probability_distribution_only):

        probs = list(prob_vec[0, :])
        angles = list(angles_vec[0, :])

        fig = plt.figure(figsize = (4, (1 - vpr(qubit_count)) * 13.5))
        grid_spec_bars = {
                "bottom": (0.95 - 0.08)/2 + 0.08 + 0.02,
                "top": 0.95,
                "left": 0.01,
                "right": 0.93,
                "wspace": 0.0,
                "hspace": 0.0,
                "height_ratios": [1]
            }
        ax_dict_bars = fig.subplot_mosaic(
            [
                ["fidelity"]
            ],
            gridspec_kw = grid_spec_bars,
        )
        
        c_gray = [0.6, 0.6, 0.6, 0.1]

        if probability_distribution_only == False:
            grid_spec_phase_wheel = {
                    "bottom": 0.12,
                    "top": 0.44,
                    "left": 0.01,
                    "right": 0.93,
                    "wspace": 0.0,
                    "hspace": 0.0,
                    "height_ratios": [1]
                }
            ax_dict_phase_wheel = fig.subplot_mosaic(
                [
                    ["phase_wheel"],
                ],
                gridspec_kw = grid_spec_phase_wheel,
                subplot_kw={"projection": "polar"}
            )

            plt.sca(ax_dict_phase_wheel["phase_wheel"])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            #plt.tight_layout()

            # phase wheel
            cmap = cm.get_cmap('hsv', 256)
            if invert_colours == True:
                cmap = invert_cmap(cmap)
            
            

            azimuths = np.arange(0, 361, 1)
            zeniths = np.arange(40, 70, 1)
            values = azimuths * np.ones((30, 361))
            ax_dict_phase_wheel["phase_wheel"].pcolormesh(azimuths*np.pi/180.0, zeniths, np.roll(values,180), cmap = cmap)
            ax_dict_phase_wheel["phase_wheel"].fill_between(azimuths*np.pi/180.0, 40, color = '#FFFFFF')

            if invert_colours == True:
                ax_dict_phase_wheel["phase_wheel"].plot(azimuths*np.pi/180.0, [40]*361, color=tick_colour, lw=1)
                if phase_marker == True:
                    for angle_iter, angle in enumerate(angles):
                        if probs[angle_iter] > 0.0001:
                            ax_dict_phase_wheel["phase_wheel"].plot([angle] * 40, np.arange(0, 40, 1), color=tick_colour, lw=2)
                ax_dict_phase_wheel["phase_wheel"].spines['polar'].set_color(tick_colour)
            else:
                ax_dict_phase_wheel["phase_wheel"].plot(azimuths*np.pi/180.0, [40]*361, color=tick_colour, lw=1)
                if phase_marker == True:
                    for angle_iter, angle in enumerate(angles):
                        if probs[angle_iter] > 0.0001:
                            ax_dict_phase_wheel["phase_wheel"].plot([angle] * 40, np.arange(0, 40, 1), color=tick_colour, lw=2)
            ax_dict_phase_wheel["phase_wheel"].set_yticks([])


            ax_dict_phase_wheel["phase_wheel"].tick_params(axis='x', colors=tick_colour)
            ax_dict_phase_wheel["phase_wheel"].tick_params(axis='y', colors=tick_colour)
            fig.text(0.82, 0.465, 'Phase', ha='right', va='bottom', fontsize=20)

            label_positions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
            labels = ['0',r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{2}$']
            ax_dict_phase_wheel["phase_wheel"].set_xticks(label_positions, labels)
            ax_dict_phase_wheel["phase_wheel"].xaxis.set_tick_params(pad = 8)
        
        plt.sca(ax_dict_bars["fidelity"])
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)

        # fidelity colorbar
        cmap = cmap_rvb
        
        if invert_colours == True:
            cmap = invert_cmap(cmap_rvb)    
        
        ax_dict_bars["fidelity"].imshow(np.array(list(reversed([[val] * 6 for val in reversed(np.linspace(0,1,100))]))), cmap = cmap, interpolation='bicubic')
        
        line_y = grid_spec_bars["bottom"] + (grid_spec_bars["top"] - grid_spec_bars["bottom"]) * fidelity
        line_middle_x = grid_spec_bars["left"] + (grid_spec_bars["right"] - grid_spec_bars["left"]) / 2
        line = Line2D([line_middle_x - 0.035, line_middle_x + 0.035], [line_y, line_y], lw=4, color=c_gray, alpha=1)
        line.set_clip_on(False)
        fig.add_artist(line)
        ax_dict_bars["fidelity"].tick_params(axis='x', colors=tick_colour)
        ax_dict_bars["fidelity"].tick_params(axis='y', colors=tick_colour)
        if invert_colours == True:
            ax_dict_bars["fidelity"].spines['bottom'].set_color(tick_colour)
            ax_dict_bars["fidelity"].spines['top'].set_color(tick_colour)
            ax_dict_bars["fidelity"].spines['left'].set_color(tick_colour)
            ax_dict_bars["fidelity"].spines['right'].set_color(tick_colour)
            for t in ax_dict_bars["fidelity"].xaxis.get_ticklines(): t.set_color(tick_colour)
            for t in ax_dict_bars["fidelity"].yaxis.get_ticklines(): t.set_color(tick_colour)
        

        fig.text(0.82, 0.945, 'Fidelity', ha='right', va='center', fontsize=20)
        fig.text(0.82, 0.905, format(fidelity, '.2f'), ha='right', va='center', fontsize=20)
        
        ax_dict_bars["fidelity"].xaxis.set_visible(False)
        ax_dict_bars["fidelity"].set_ylim((0,99))
        y_tick_positions = [0, 50, 99]
        y_tick_labels = [0.0, 0.5, 1.0]
        ax_dict_bars["fidelity"].set_yticks(y_tick_positions)
        ax_dict_bars["fidelity"].set_yticklabels(y_tick_labels)

        plt.savefig(output_manager.get_path(f'info_panel_{plot_number}.png'))
        plt.close('all')
        return None
    
    print("Generating pieces...")
    files = output_manager.glob('frame_*')
    for file in files:
        os.remove(file)
    files = output_manager.glob('info_panel_*')
    for file in files:
        os.remove(file)
    num_frames = len(sounds_list)
    input_probability_vector_list = []
    angle_vector_list = []
    clips = []
    for sound_iter, sound_data in enumerate(sounds_list):
        input_probability_vector = np.zeros((8, len(sounds_list[0][0][2])))
        angle_vector = np.zeros((8, len(sounds_list[0][0][2])))
        for j in range(8):
            if j < len(sound_data):
                input_probability_vector[j, :] = np.array(
                    sounds_list[sound_iter][j][2]) * sounds_list[sound_iter][j][0]
                angle_vector[j, :] = sounds_list[sound_iter][j][3]
        input_probability_vector_list.append(input_probability_vector)
        angle_vector_list.append(angle_vector)
    meas_prob_vector_list = []
    for i in range(len(meas_probs_list)):
        meas_prob_vector_list.append(np.array(meas_probs_list[i]))
    accumulated_times = []
    accumulated_times.append(0)
    accumulated_time = 0
    for times in rhythm:
        frame_time = (times[0] + times[1]) / 480.0
        accumulated_time += frame_time
        accumulated_times.append(accumulated_time)
    for sound_iter, sound_data in enumerate(sounds_list):
        plot_info_panel(sound_iter, fidelity_list[sound_iter],
                        input_probability_vector_list[sound_iter], angle_vector_list[sound_iter])
        # histograms
        if smooth_transitions == True:
            anim_fig = plt.figure(figsize=(20, (1 - vpr(qubit_count)) * 13.5))

            def make_histogram_frame(t, temp=sound_iter):
                frame_iter = temp
                accumulated_time = accumulated_times[frame_iter+1]
                frame_time = (rhythm[frame_iter][0] +
                              rhythm[frame_iter][1]) / 480.0
                anim_fig.clear()
                plt.cla()
                plt.clf()
                target_transition_time = 0.1
                time_since_frame = t
                time_between_frames = frame_time
                transition_scale = 0
                if time_between_frames < target_transition_time:
                    target_transition_time = time_between_frames * 0.3
                if time_since_frame >= target_transition_time:
                    transition_scale = 1
                else:
                    transition_scale = time_since_frame / target_transition_time
                if frame_iter == 0:
                    interpolated_frame = frame_iter
                    fig = plot_quantum_state(input_probability_vector_list, angle_vector_list, meas_prob_vector_list,
                                             interpolated_frame, interpolate=False, save=False, fig=anim_fig, zero_noise=zero_noise)
                else:
                    interpolated_frame = frame_iter - 1 + transition_scale
                    fig = plot_quantum_state(input_probability_vector_list, angle_vector_list, meas_prob_vector_list,
                                             interpolated_frame, interpolate=True, save=False, fig=anim_fig, zero_noise=zero_noise)
                return mplfig_to_npimage(fig)
            clips.append(VideoClip(make_histogram_frame, duration=(
                rhythm[sound_iter][0] + rhythm[sound_iter][1]) / 480))
        else:
            plot_quantum_state(input_probability_vector_list, angle_vector_list,
                               meas_prob_vector_list, sound_iter, save=True, zero_noise=zero_noise)
    
    if smooth_transitions == False:
        clips = []
        total_time = 0
        files = output_manager.glob('frame_*')
        file_tuples = []
        for file in files:
            file = file.replace("\\", "/")
            num_string = re.search(r'\d+$', os.path.splitext(file)[0]).group()
            num = int(num_string)
            file_tuples.append((num, file))
        file_tuples = sorted(file_tuples)
        files = [x[1] for x in file_tuples]
        iter = 0
        for file in files:
            time = (rhythm[iter][0] + rhythm[iter][1]) / 480.0
            clips.append(ImageClip(file).set_duration(time))
            total_time += time
            iter += 1
    else:
        total_time = 0
        for times in rhythm:
            total_time += (times[0] + times[1]) / 480.0
    files = output_manager.glob('info_panel_*')
    file_tuples = []
    for file in files:
        file = file.replace("\\", "/")
        num_string = re.search(r'\d+$', os.path.splitext(file)[0]).group()
        num = int(num_string)
        file_tuples.append((num, file))
    file_tuples = sorted(file_tuples)
    files = [x[1] for x in file_tuples]
    iter = 0
    for file in files:
        time = (rhythm[iter][0] + rhythm[iter][1]) / 480.0
        clips[iter] = clips_array([[clips[iter], ImageClip(file).set_duration(
            time).resize(height=clips[iter].size[1])]], bg_color=[0xFF, 0xFF, 0xFF])
        total_time += time
        iter += 1
    video = concatenate(clips, method="compose")
    video = video.resize(width=1920)
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
    # TODO: video is not being rendered properly anymore. Barrier and empty are the wrong vertical sizes and the circuit 
    # ends way too early. Maybe rhythm is wrong? or there is double counting somewhere, like file count or something.
    image_barrier_clip = ImageClip(output_manager.get_path('partial_circ_barrier.png')).set_duration(video.duration)
    if image_barrier_clip.size[0] > 156:
        image_barrier_clip = crop.crop(
            image_barrier_clip, x1=133, x2=image_barrier_clip.size[0]-23)
    image_barrier_clip = image_barrier_clip.resize(height=1080 - video.size[1])
    barrier_image_width = image_barrier_clip.size[0]
    # create image clip same size as barrier.
    image_empty_clip = ImageClip(output_manager.get_path('partial_circ_empty.png')).set_duration(video.duration)
    if image_empty_clip.size[0] > 156:
        image_empty_clip = crop.crop(
            image_empty_clip, x1=133, x2=image_empty_clip.size[0]-23)
    image_empty_clip = image_empty_clip.resize(height=1080 - video.size[1])
    image_empty_clip_array = clips_array(
        [[image_empty_clip, image_empty_clip, image_empty_clip]], bg_color=[0xFF, 0xFF, 0xFF])
    image_empty_clip_array = crop.crop(
        image_empty_clip_array, x1=0, x2=barrier_image_width)
    image_empty_clip = ImageClip(output_manager.get_path('partial_circ_barrier.png')).set_duration(video.duration)
    if image_empty_clip.size[0] > 156:
        image_empty_clip = crop.crop(
            image_empty_clip, x1=133, x2=image_empty_clip.size[0]-23)
    barrier_only_clip = crop.crop(image_empty_clip, x1=35, x2=72)
    barrier_only_clip = barrier_only_clip.margin(
        left=35, right=72, color=[0xFF, 0xFF, 0xFF])
    barrier_only_clip = barrier_only_clip.resize(height=1080 - video.size[1])
    barrier_only_clip_mask = barrier_only_clip.to_mask()
    barrier_only_clip_mask = barrier_only_clip_mask.fl_image(
        lambda pic: filter_colour_round_with_threshold(pic, threshold=0.9, colour_dark=0.0, colour_light=1.0))
    barrier_only_clip_mask = invert_colors.invert_colors(
        barrier_only_clip_mask)
    # barrier_only_clip.save_frame("./barrier_only_clip.png", t=0)
    # barrier_only_clip = barrier_only_clip.fl_image( lambda pic: filter_color_multiply(pic, [0, 0, 255]))
    barrier_only_clip = barrier_only_clip.fl_image(
        lambda pic: filter_blend_colour(pic, [255, 255, 255], 0.3))
    # might be able to remove this line
    barrier_only_clip = barrier_only_clip.add_mask()
    barrier_only_clip = barrier_only_clip.set_mask(barrier_only_clip_mask)
    vertical_shrink = 0.05
    clip_height = barrier_only_clip.size[1]
    barrier_start_y = 43
    barrier_end_y = 25
    height = clip_height - barrier_start_y - barrier_end_y
    barrier_only_clip = crop.crop(barrier_only_clip, y1=int(
        (vertical_shrink/2.0) * height) + barrier_start_y, y2=int((1.0 - vertical_shrink/2.0) * height) + barrier_start_y)
    barrier_only_clip = barrier_only_clip.resize(
        newsize=(int(0.5 * barrier_only_clip.size[0]), int(barrier_only_clip.size[1])))
    # image_empty_clip_array = Compospartial_iteVideoClip([image_empty_clip_array, barrier_only_clip.set_position(("center", int((vertical_shrink/2.0) * height) + barrier_start_y))], use_bgclip=True)
    for i, partial_circ in enumerate(partial_circuit_list):
        new_image_clip = ImageClip(output_manager.get_path(f'partial_circ_{i}.png')).set_duration(video.duration)
        if new_image_clip.size[0] > 156:
            new_image_clip = crop.crop(
                new_image_clip, x1=133, x2=new_image_clip.size[0]-23)
        new_image_clip = new_image_clip.resize(height=1080 - video.size[1])
        if i != len(partial_circuit_list)-1:
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
    circ_clip_arr = clips_array(
        [[x for x in all_clips]], bg_color=[0xFF, 0xFF, 0xFF])
    circ_clip_arr.fps = fps
    composited_with_barrier_clips = []
    composited_with_barrier_clips.append(circ_clip_arr)
    note_accumulated_info = []  # accumulated time, note length, note rest
    accumulated_time = 0
    for iter in range(len(rhythm)):
        # new_barrier_clip = image_barrier_clip.set_start(accumulated_time).set_end(min(video.duration, accumulated_time + 1 / 4.0)).set_position((positions_x[iter]-barrier_image_width, 0))
        note_length = rhythm[iter][0] / 480.0
        new_barrier_clip = image_barrier_clip.set_start(0).set_end(min(
            accumulated_time, video.duration)).set_position((int(positions_x[iter]-barrier_image_width), 0))
        note_accumulated_info.append(
            (accumulated_time, rhythm[iter][0] / 480.0, rhythm[iter][1] / 480.0))
        accumulated_time += (rhythm[iter][0] + rhythm[iter][1]) / 480.0
        # new_barrier_clip.add_mask()
        # new_barrier_clip = new_barrier_clip.crossfadeout(note_length)
        composited_with_barrier_clips.append(new_barrier_clip)
    video_duration = video.duration
    video_size = video.size
    circ_clip_arr = CompositeVideoClip(composited_with_barrier_clips)
    circ_clip_arr = circ_clip_arr.resize(height=1080 - video.size[1])
    vertical_scale = 1080 / float(video.size[1] + circ_clip_arr.size[1])
    circ_clip_target_width = int(1920 / vertical_scale)
    clip_orig_x = circ_clip_arr.size[0]
    circ_clip_arr = circ_clip_arr.margin(
        left=circ_clip_target_width, right=circ_clip_target_width, color=[0xFF, 0xFF, 0xFF])
    # circ_clip_arr = crop.crop(circ_clip_arr, x1 = -clip_orig_x, x2 = 2 * clip_orig_x)
    # circ_clip_arr.save_frame("contatenated_circ.png", t = 0.1)
    h = circ_clip_arr.h
    w = circ_clip_target_width

    def f(gf, t):
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
        x = int(x_start + (prev_pos + (target_pos - prev_pos)
                * (t - prev_time)/(target_time - prev_time)))
        y = 0
        return gf(t)[y:y+h, x:x+w]
    circ_clip_arr = circ_clip_arr.fl(f, apply_to="mask")
    if invert_colours == True:
        circ_clip_arr = invert_colors.invert_colors(circ_clip_arr)
    # clip_arr = clips_array([[circuit_video.resize(circuit_rescale)], [video]], bg_color=bg_color)
    clip_arr = clips_array([[circ_clip_arr], [video]], bg_color=bg_color)
    video_final = clip_arr

    files = output_manager.glob(f'{output_manager.default_name}-*.wav')
    audio_clips = []
    audio_file_clips = []
    for file in files:
        filename = file.replace("\\", "/")
        audio_file_clip = mpy.AudioFileClip(filename, nbytes=4, fps=44100)
        audio_file_clips.append(audio_file_clip)
        audio_array = audio_file_clip.to_soundarray(nbytes=4)
        total_time = audio_file_clip.duration
        audio_array_clip = AudioArrayClip(
            audio_array[0:int(44100 * total_time)], fps=44100)
        audio_clips.append(audio_array_clip)
    composed_audio_clip = CompositeAudioClip(audio_file_clips)
    video_duration = video_final.duration
    target_duration = max(video_duration, composed_audio_clip.duration)
    video_final = video_final.set_duration(target_duration)
    video_final = freeze.freeze(
        video_final, t=video_duration, freeze_duration=target_duration-video_duration)
    video_final = video_final.set_duration(target_duration)
    video_final = clip_arr.set_audio(composed_audio_clip)
    

    video_final = crop.crop(video_final, x1=int(
        clip_arr.size[0]/2-circ_clip_target_width/2), x2=int(clip_arr.size[0]/2+circ_clip_target_width/2))
    if video_final.size[1] > 1080:
        difference = video_final.size[1] - 1080
        video_final = crop.crop(video_final, x1=math.floor(
            difference/2), x2=video_final.size[0] - math.ceil(difference/2))
    if video_final.size[1] < 1080:
        difference = 1080 - video_final.size[1]
        video_final = video_final.margin(left=math.floor(
            difference/2), right=math.ceil(difference/2))
    # vertical_scale = 1080 / video_final.size[1]
    # if video_final.size[1] / 1080 > video_final.size[0] / 1920:
    #    video_final = video_final.resize(height=1080)
    #    left_margin = int((1920 - video_final.size[0])/2)
    #    video_final = video_final.margin(left=left_margin, right=(1920 - video_final.size[0] - left_margin), color=bg_color)
    # else:
    #    video_final = video_final.resize(width=1920)
    #    video_final = video_final.margin(top=(1080 - video_final.size[1])/2, bottom=(1080 - video_final.size[1])/2, color=bg_color)
    highlight_time = 1.0 / 8.0
    highlight_fade_time = 1.0 / 16.0

    def draw_needle(get_frame, t):
        """Draw a rectangle in the frame"""
        # change (top, bottom, left, right) to your coordinates
        top = 1
        bottom = int(
            vertical_scale*circ_clip_arr.size[1] - 1 + vertical_scale*(barrier_start_y - barrier_end_y))
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
        note_remaining_play_time = max(
            0.0, current_note_info[1] - time_since_note_played)
        time_to_start_fade = 0.0  # highlight_time - highlight_fade_time
        time_to_stop_fade = highlight_time
        highlight_fade_time = current_note_info[1]
        # blend_colour(bg_color_inverse, [127, 127, 127], 0.8)
        idle_colour = [127, 127, 127]
        lerp_time = (time_since_note_played -
                     time_to_start_fade) / highlight_fade_time
        scale = (2.0 * (1 - fidelity) - 1.0)  # -1 to 1
        scale = np.tanh(scale) / np.tanh(1)
        scale = (scale + 1.0) / 2.0
        highlight_colour = [int(255 * cmap_bvr(scale)[i]) for i in range(3)]
        lerp_colour = [int(x) for x in ease_out(
            highlight_colour, idle_colour, lerp_time)]
        frame[top: top+3, left: right] = lerp_colour
        frame[bottom-3: bottom, left: right] = lerp_colour
        frame[top+3: bottom, left: left+3] = lerp_colour
        frame[top+3: bottom, right-3: right] = lerp_colour
        return frame
    video_final = video_final.fl(draw_needle)
    video_final.save_frame(output_manager.get_path('save_frame_0.png'), t=0.0)
    video_final.save_frame(output_manager.get_path('save_frame_1.png'), t=video_final.duration - 1)
    video_final.save_frame(output_manager.get_path('save_frame_fading.png'), t=1.0 - highlight_fade_time / 2.0)
    # def supersample(clip, d, nframes):
    #    """ Replaces each frame at time t by the mean of `nframes` equally spaced frames
    #    taken in the interval [t-d, t+d]. This results in motion blur.
    #    credit: Zulko https://gist.github.com/Zulko/f90674b2e64c5600370e"""
    #    def fl(gf, t):
    #        tt = np.linspace(t-d, t+d, nframes)
    #        avg = np.mean(1.0*np.array([gf(t_) for t_ in tt]),axis=0)
    #        return avg.astype("uint8")
    #    return clip.fl(fl)
#
    # video_final = supersample(video_final, d=0.008, nframes=3)
    # preset options (speed vs filesize): ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
    # audio_codec = "libmp3lame"
    video_final.write_videofile(output_manager.get_default_file_pathname() + '.mp4', preset='ultrafast', fps=fps, codec='mpeg4', audio_fps=44100,
                                audio_codec='libmp3lame', audio_bitrate="3000k", audio_nbytes=4, ffmpeg_params=["-b:v", "12000K", "-b:a", "3000k"])
    # files = glob.glob(target_folder + '/*.mp4')
    # files = glob.glob(NAME + "/" + NAME + '-*.wav')
#     for file in files:
#         os.remove(file)
