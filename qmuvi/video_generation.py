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

def reverse_cmap(cmap):
    from matplotlib.colors import ListedColormap
    newcolors = cmap(np.linspace(0, 1, 256))
    for i in reversed(range(256)):
        newcolors[i, :] = np.array([cmap(i/256)[0],cmap(i/256)[1],cmap(i/256)[2],1])
    return ListedColormap(newcolors)

def _plot_phase_wheel(fig, 
                     probabilities_0,
                     phase_angles_0,
                     cmap_phase, 
                     tick_colour, 
                     invert_colours
                     ):
    fig_gridspec_phase_wheel = {
            "bottom": 0.12,
            "top": 0.44,
            "left": 0.01,
            "right": 0.93,
            "wspace": 0.0,
            "hspace": 0.0,
            "height_ratios": [1]
        }
    plot_mosaic_phase_wheel_dict = fig.subplot_mosaic(
        [
            ["phase_wheel"],
        ],
        gridspec_kw = fig_gridspec_phase_wheel,
        subplot_kw={"projection": "polar"}
    )
    
    plt.sca(plot_mosaic_phase_wheel_dict["phase_wheel"])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    
    azimuths = np.arange(0, 361, 1)
    values = azimuths * np.ones((30, 361))
    azimuths = azimuths * np.pi / 180.0
    zeniths = np.arange(40, 70, 1)
    plot_mosaic_phase_wheel_dict["phase_wheel"].pcolormesh(azimuths, 
                                                           zeniths, 
                                                           np.roll(values, 180), 
                                                           cmap=cmap_phase
                                                           )
    plot_mosaic_phase_wheel_dict["phase_wheel"].fill_between(azimuths, 40, color = '#FFFFFF')
    
    plot_mosaic_phase_wheel_dict["phase_wheel"].plot(azimuths, [40] * 361, color=tick_colour, lw=1)
    for angle_iter, angle in enumerate(phase_angles_0):
        if probabilities_0[angle_iter] > 0.0001:
            plot_mosaic_phase_wheel_dict["phase_wheel"].plot([angle] * 40, np.arange(0, 40, 1), color=tick_colour, lw=2)
    if invert_colours == True:
        plot_mosaic_phase_wheel_dict["phase_wheel"].spines['polar'].set_color(tick_colour)
    
    plot_mosaic_phase_wheel_dict["phase_wheel"].set_yticks([])
    plot_mosaic_phase_wheel_dict["phase_wheel"].tick_params(axis='x', colors=tick_colour)
    plot_mosaic_phase_wheel_dict["phase_wheel"].tick_params(axis='y', colors=tick_colour)
    fig.text(0.82, 0.465, 'Phase', ha='right', va='bottom', fontsize=20)
    
    label_positions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    labels = ['0',r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{2}$']
    plot_mosaic_phase_wheel_dict["phase_wheel"].set_xticks(label_positions, labels)
    plot_mosaic_phase_wheel_dict["phase_wheel"].xaxis.set_tick_params(pad = 8)
    return plot_mosaic_phase_wheel_dict

def _plot_stat_bars(fig, 
                   fidelity, 
                   cmap_fidelity, 
                   c_gray, 
                   tick_colour, 
                   invert_colours
                   ):
    fig_gridspec_stat_bars = {
            "bottom": (0.95 - 0.08)/2 + 0.08 + 0.02,
            "top": 0.95,
            "left": 0.01,
            "right": 0.93,
            "wspace": 0.0,
            "hspace": 0.0,
            "height_ratios": [1]
        }
    plot_mosaic_stat_bars_dict = fig.subplot_mosaic(
        [
            ["fidelity"]
        ],
        gridspec_kw = fig_gridspec_stat_bars,
    )
    
    plt.sca(plot_mosaic_stat_bars_dict["fidelity"])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20) 
    
    plot_mosaic_stat_bars_dict["fidelity"].imshow(np.array(list(reversed([[val] * 6 for val in reversed(np.linspace(0,1,100))]))), cmap=cmap_fidelity, interpolation='bicubic')
    
    line_y = fig_gridspec_stat_bars["bottom"] + (fig_gridspec_stat_bars["top"] - fig_gridspec_stat_bars["bottom"]) * fidelity
    line_middle_x = fig_gridspec_stat_bars["left"] + (fig_gridspec_stat_bars["right"] - fig_gridspec_stat_bars["left"]) / 2
    line = Line2D([line_middle_x - 0.035, line_middle_x + 0.035], [line_y, line_y], lw=4, color=c_gray, alpha=1)
    line.set_clip_on(False)
    fig.add_artist(line)
    plot_mosaic_stat_bars_dict["fidelity"].tick_params(axis='x', colors=tick_colour)
    plot_mosaic_stat_bars_dict["fidelity"].tick_params(axis='y', colors=tick_colour)
    if invert_colours == True:
        plot_mosaic_stat_bars_dict["fidelity"].spines['bottom'].set_color(tick_colour)
        plot_mosaic_stat_bars_dict["fidelity"].spines['top'].set_color(tick_colour)
        plot_mosaic_stat_bars_dict["fidelity"].spines['left'].set_color(tick_colour)
        plot_mosaic_stat_bars_dict["fidelity"].spines['right'].set_color(tick_colour)
        for t in plot_mosaic_stat_bars_dict["fidelity"].xaxis.get_ticklines(): t.set_color(tick_colour)
        for t in plot_mosaic_stat_bars_dict["fidelity"].yaxis.get_ticklines(): t.set_color(tick_colour)
    
    fig.text(0.82, 0.945, 'Fidelity', ha='right', va='center', fontsize=20)
    fig.text(0.82, 0.905, format(fidelity, '.2f'), ha='right', va='center', fontsize=20)
    
    plot_mosaic_stat_bars_dict["fidelity"].xaxis.set_visible(False)
    plot_mosaic_stat_bars_dict["fidelity"].set_ylim((0,99))
    y_tick_positions = [0, 50, 99]
    y_tick_labels = [0.0, 0.5, 1.0]
    plot_mosaic_stat_bars_dict["fidelity"].set_yticks(y_tick_positions)
    plot_mosaic_stat_bars_dict["fidelity"].set_yticklabels(y_tick_labels)

    return plot_mosaic_stat_bars_dict

def convert_fig_pixels_to_inches(pixels):
    dpi = plt.rcParams['figure.dpi']
    return pixels / dpi

def generate_video_from_data(quantum_circuit: qiskit.QuantumCircuit, 
                             output_manager: data_manager.DataManager, 
                             rhythm: List[Tuple[int, int]], 
                             phase_instruments: List[List[int]] = [list(range(81, 89))], 
                             invert_colours: bool = False, 
                             fps: int = 60, 
                             vpr: Optional[float] = None, 
                             smooth_transitions: bool = True, 
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
        probability_distribution_only: Whether to only plot the basis state probabilities. (default: False) Bool
    """

    if vpr == None:
        def vpr(n): return 1/3
    else:
        def vpr(n): return vpr

    sounds_list = output_manager.load_json('sounds_list.json')
    fidelity_list = output_manager.load_json('fidelity_list.json')
    measured_probabilities_list = output_manager.load_json('meas_probs_list.json')

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
    empty_circuit_fig = empty_circuit.draw(filename=output_manager.get_path('partial_circ_empty.png'), output="mpl", fold=-1)

    # a barrier circuit is inserted between partial circuits
    barrier_circuit = QuantumCircuit(qubit_count)
    barrier_circuit.barrier()
    barrier_fig = barrier_circuit.draw(filename=output_manager.get_path('partial_circ_barrier.png'), output="mpl", fold=-1)
    
    # Fidelity colour bar
    cmap_fidelity = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "violet", "blue"])
    # Musical needle 
    cmap_needle = reverse_cmap(cmap_fidelity)
    # Phase for bar plot bars
    cmap_phase = cm.get_cmap('hsv')
    # Matplotlib plot ticks
    tick_colour = [0.4, 0.4, 0.4, 1.0]

    bg_color = [0xFF, 0xFF, 0xFF]
    bg_color_inverted = [0x00, 0x00, 0x00]

    if invert_colours == True:
        cmap_fidelity = invert_cmap(cmap_fidelity)
        cmap_phase = invert_cmap(cmap_phase)
        # Note: cmap_needle does not need to be inverted since needle is not part of the image that is inverted


    def plot_quantum_state(pure_probabilities_list: List[np.ndarray], 
                           pure_phase_angles_list: List[np.ndarray], 
                           measured_probabilities_list: List[np.ndarray], 
                           sampled_state_number: Union[int, float], 
                           interpolate: bool = False, 
                           fig: Optional[matplotlib.figure.Figure] = None, 
                           zero_noise: bool = False, 
                           probability_distribution_only: bool = probability_distribution_only
                           ):
        '''
        Args:
            pure_probabilities_list: a list containing a 2d array for each sampled state, where the array contains probabilities indexed by pure state number and basis state number
            pure_phase_angles_list: a list containing a 2d array for each sampled state, where the array contains phase angles indexed by pure state number and basis state number
            measured_probabilities_list: a list containing a 1d array for each sampled state, where the array contains measurement probabilities indexed by basis state number
            sampled_state_number: the index of the sampled state to be plotted. If interpolate is True, then this can be a decimal fraction
            interpolate: whether to interpolate plot numbers
            fig: an already created figure
            zero_noise: whether to only plot the data for the single pure state (no noise)
            probability_distribution_only: whether to only plot the basis-state probability distribution
        '''
        if fig == None:
            fig = plt.figure(figsize=(20, (1 - vpr(qubit_count)) * 13.5))
        
        if interpolate == True:
            base_state_count = pure_probabilities_list[0].shape[1]
            
            # create new vectors that lerp between the sampled states with respect to sampled_state_number
            pure_probabilities_start = pure_probabilities_list[math.floor(sampled_state_number)]
            pure_probabilities_end = pure_probabilities_list[math.ceil(sampled_state_number)]
            pure_probabilities = np.zeros((8, base_state_count))
            for i in range(pure_probabilities_start.shape[0]):
                pure_probabilities[i, :] = lerp(
                    pure_probabilities_start[i, :], pure_probabilities_end[i, :], sampled_state_number - math.floor(sampled_state_number))

            pure_phase_angles_start = pure_phase_angles_list[math.floor(sampled_state_number)]
            pure_phase_angles_end = pure_phase_angles_list[math.ceil(sampled_state_number)]
            pure_phase_angles = np.zeros((8, base_state_count))
            for i in range(pure_phase_angles.shape[0]):
                pure_phase_angles[i, :] = lerp(pure_phase_angles_start[i, :], pure_phase_angles_end[i, :], sampled_state_number - math.floor(sampled_state_number))

            measured_probabilities_start = measured_probabilities_list[math.floor(sampled_state_number)]
            measured_probabilities_end = measured_probabilities_list[math.ceil(sampled_state_number)]
            measured_probabilities = lerp(measured_probabilities_start[:], measured_probabilities_end[:], sampled_state_number - math.floor(sampled_state_number))

            # avoid animating the phases when the probability's start or end point is zero
            for i in range(pure_phase_angles.shape[0]):
                for j in range(pure_phase_angles.shape[1]):
                    if pure_probabilities_start[i, j] <= 0.0001 and pure_probabilities_end[i, j] > 0:
                        pure_phase_angles[i, j] = pure_phase_angles_end[i, j]
                    if pure_probabilities_start[i, j] > 0 and pure_probabilities_end[i, j] <= 0.0001:
                        pure_phase_angles[i, j] = pure_phase_angles_start[i, j]
        else:
            pure_probabilities = pure_probabilities_list[sampled_state_number]
            pure_phase_angles = pure_phase_angles_list[sampled_state_number]
            measured_probabilities = measured_probabilities_list[sampled_state_number]
            base_state_count = pure_probabilities.shape[1]
        
        # A list of plot names where their index in the list corresponds to which pure state they are plotting
        plot_pure_state_list = None
        # A dictionary of plot names where the key is the plot name and the value is a matplotlib Axes object with a set position and size within the figure
        plot_mosaic_dict = None
        # specify the plot mosaic layout
        if probability_distribution_only:
            # only plot the basis-state probability distribution. No phases or noise-induced pure states.
            fig_gridspec = {
                "bottom": 0.1,
                "top": 0.95,
                "left": 0.07,
                "right": 0.99,
                "wspace": 0.05,
                "hspace": 0.09,
            }
            plot_mosaic_dict = fig.subplot_mosaic(
                [
                    ["meas_probs", "meas_probs", "meas_probs", "meas_probs"],
                    ["meas_probs", "meas_probs", "meas_probs", "meas_probs"],
                ],
                gridspec_kw=fig_gridspec,
            )
        else:
            if zero_noise:
                # only plot the first and only (when no noise) pure state probability distribution.
                fig_gridspec = {
                    "bottom": 0.1,
                    "top": 0.95,
                    "left": 0.07,
                    "right": 0.99,
                    "wspace": 0.05,
                    "hspace": 0.09,
                }
                plot_mosaic_dict = fig.subplot_mosaic(
                    [
                        ["pure_state:0", "pure_state:0", "pure_state:0", "pure_state:0"],
                        ["pure_state:0", "pure_state:0", "pure_state:0", "pure_state:0"],
                    ],
                    gridspec_kw=fig_gridspec,
                )
            else:
                # Plot the probability distributions of the pure states in the mixed state
                fig_gridspec = {
                    "bottom": 0.08,
                    "top": 0.95,
                    "left": 0.07,
                    "right": 0.99,
                    "wspace": 0.05,
                    "hspace": 0.09,
                }
                plot_mosaic_dict = fig.subplot_mosaic(
                    [
                        ["pure_state:1", "pure_state:0", "pure_state:0", "pure_state:2"],
                        ["pure_state:3", "pure_state:4", "pure_state:5", "pure_state:6"],
                    ],
                    gridspec_kw=fig_gridspec,
                )
        
        num_qubits = (base_state_count - 1).bit_length()
        x_values = [x for x in range(base_state_count)]
        for plot_name, plot_axes in plot_mosaic_dict.items():

            pure_state_index = None
            if plot_name.split(":")[0] == "pure_state":
                pure_state_index = int(plot_name.split(":")[1])

            if plot_name == "meas_probs":
                y_values = measured_probabilities
            elif pure_state_index != None:
                y_values = pure_probabilities[pure_state_index, :]
            else:
                print("ERROR: plot_name not recognised: " + plot_name)
                exit(1)

            bar_list = plot_mosaic_dict[plot_name].bar(x_values, y_values, width=0.5)
            
            # set colour of the bars based on the phase
            if pure_state_index != None:
                for i, phase_angle in enumerate(pure_phase_angles[pure_state_index, :]):
                    # Map phase angles from (-pi/2, pi/2] to (0, 1]
                    colour_value = (phase_angle + np.pi) / (2 * np.pi)
                    bar_list[i].set_color(cmap_phase(colour_value))

            plot_mosaic_dict[plot_name].set_xlim((-0.5, base_state_count - 1 + 0.5))
            plot_mosaic_dict[plot_name].set_ylim([0, np.max(y_values)])
            plot_mosaic_dict[plot_name].tick_params(axis='x', colors=tick_colour)
            plot_mosaic_dict[plot_name].tick_params(axis='y', colors=tick_colour)
            plot_mosaic_dict[plot_name].tick_params(axis='y', labelsize=20)
            plot_mosaic_dict[plot_name].axes.xaxis.set_visible(False)
            plot_mosaic_dict[plot_name].axes.yaxis.set_visible(False)

            if (zero_noise and pure_state_index == 0) or plot_name == "meas_probs":
                plot_mosaic_dict[plot_name].axes.yaxis.set_visible(True)

            if pure_state_index == 1 or pure_state_index == 3:
                plot_mosaic_dict[plot_name].axes.yaxis.set_visible(True)

            # Set the x ticks and tick labels for important subplots
            if pure_state_index == 0 or plot_name == "meas_probs":
                # if there are too many qubits to fit all of the x ticks, then only draw a limited number
                if (zero_noise and num_qubits > 4) \
                        or ((not zero_noise) and num_qubits > 3) \
                        or (plot_name == "meas_probs" and num_qubits > 4):
                    x_ticks = [0]
                    x_ticks.append(int(base_state_count / 4.0))
                    x_ticks.append(int(2.0 * base_state_count / 4.0))
                    x_ticks.append(int(3.0 * base_state_count / 4.0))
                    x_ticks.append(int(base_state_count - 1))
                else:
                    x_ticks = list(range(base_state_count))
                
                # basis states are labelled in binary
                x_tick_labels = [bin(x)[2:].zfill(num_qubits)[::-1] for x in x_ticks]
                
                plot_mosaic_dict[plot_name].axes.xaxis.set_visible(True)
                plot_mosaic_dict[plot_name].set_xticks(x_ticks)
                plot_mosaic_dict[plot_name].set_xticklabels(x_tick_labels)
                plot_mosaic_dict[plot_name].tick_params(axis='x', labelsize=14)

        # Add axis labels
        fig.text(0.01, 
                 (fig_gridspec["top"] - fig_gridspec["bottom"])/2 + fig_gridspec["bottom"],
                 'Probability', 
                 va='center', 
                 rotation='vertical', 
                 fontsize=20
                )
        fig.text((fig_gridspec["right"] - fig_gridspec["left"])/2 + fig_gridspec["left"], 
                 0.035, 
                 'Quantum states', 
                 ha='center', 
                 fontsize=20
                )
        return fig

    def plot_info_panel(sampled_state_number, 
                        fidelity, 
                        pure_probabilities, 
                        pure_phase_angles, 
                        draw_phase_wheel = not probability_distribution_only
                        ):
        """ Plot the information panel corresponding to the given sampled state number and save it as a .png
        Args:
            sampled_state_number: Used to save the plotted info panel
            fidelity: The fidelity of the state
            pure_probabilities: a 2d array, where the array contains probabilities indexed by pure state number and basis state number (pure_state_numbers x basis_state_numbers)
            pure_phase_angles: a 2d array, where the array contains phase angles indexed by pure state number and basis state number (pure_state_numbers x basis_state_numbers)
            draw_phase_wheel: If True, only plot the phase wheel
        """
        probabilities_0 = list(pure_probabilities[0, :])
        phase_angles_0 = list(pure_phase_angles[0, :])

        c_gray = [0.6, 0.6, 0.6, 1]
        
        target_video_height_pixels = 1080
        dpi = plt.rcParams['figure.dpi']
        target_empty_circuit_height_pixels = target_video_height_pixels * (1 - vpr(qubit_count))
        fig_height_inches = (1 - vpr(qubit_count)) * 13.5
        fig_height_pixels = fig_height_inches * dpi
        scale = target_empty_circuit_height_pixels / fig_height_pixels
        
        scaled_dpi = dpi * scale

        target_video_height_in_inches = convert_fig_pixels_to_inches(1080)
        fig = plt.figure(figsize = (4, (1 - vpr(qubit_count)) * 13.5), dpi = scaled_dpi)

        if draw_phase_wheel == True:
            plot_mosaic_phase_wheel_dict = _plot_phase_wheel(fig, 
                                                             probabilities_0,
                                                             phase_angles_0,
                                                             cmap_phase, 
                                                             tick_colour, 
                                                             invert_colours
                                                            )
        plot_mosaic_stat_bars_dict = _plot_stat_bars(fig, 
                                                     fidelity,
                                                     cmap_fidelity,  
                                                     c_gray, 
                                                     tick_colour, 
                                                     invert_colours
                                                    )

        plt.savefig(output_manager.get_path(f'info_panel_{sampled_state_number}.png'))
        plt.close('all')
        return None
    
    print("Generating pieces...")

    # format probability and angle data for plotting
    pure_probabilities_list = []
    pure_phase_angles_list = []
    for sound_index, sound_data in enumerate(sounds_list):
        # sounds list has tuple elements with the following structure:
        # (
        #  pure_state_prob: float, 
        #  pure_state_info: Dict[int basis_state_number, Tuple[float basis_state_prob, float basis_state_angle]], # where (basis_state_prob > eps)
        #  all_basis_state_probs: List[float basis_state_prob], 
        #  all_basis_state_angles: List[float basis_state_angle]
        # )
        pure_state_count = len(sounds_list[sound_index])
        pure_probabilities = np.zeros((pure_state_count, len(sounds_list[sound_index][0][2])))
        pure_phase_angles = np.zeros((pure_state_count, len(sounds_list[sound_index][0][3])))
        for pure_state_index in range(pure_state_count):
            # only plot the pure states that are present in the sound data
            if pure_state_index < len(sound_data):
                pure_probabilities[pure_state_index, :] = np.array(sounds_list[sound_index][pure_state_index][2]) * sounds_list[sound_index][pure_state_index][0]
                pure_phase_angles[pure_state_index, :] = sounds_list[sound_index][pure_state_index][3]

        pure_probabilities_list.append(pure_probabilities)
        pure_phase_angles_list.append(pure_phase_angles)

    plot_clips = []
    for sound_index, sound_data in enumerate(sounds_list):
        # plot and save the info panel image
        plot_info_panel(sound_index, 
                        fidelity_list[sound_index], 
                        pure_probabilities_list[sound_index], 
                        pure_phase_angles_list[sound_index]
                       )

        # histograms
        target_video_height_pixels = 1080
        dpi = plt.rcParams['figure.dpi']
        target_empty_circuit_height_pixels = target_video_height_pixels * (1 - vpr(qubit_count))
        fig_height_inches = (1 - vpr(qubit_count)) * 13.5
        fig_height_pixels = fig_height_inches * dpi
        scale = target_empty_circuit_height_pixels / fig_height_pixels
        
        scaled_dpi = dpi * scale

        target_video_height_in_inches = convert_fig_pixels_to_inches(1080)
        anim_fig = plt.figure(figsize=(20, (1 - vpr(qubit_count)) * 13.5), dpi = scaled_dpi)

        if smooth_transitions == True:

            def make_plot_frame(time_since_frame, sound_index_temp=sound_index):
                anim_fig.clear()
                plt.cla()
                plt.clf()

                sound_time = (rhythm[sound_index_temp][0] + rhythm[sound_index_temp][1]) / 480.0
                time_between_frames = sound_time
                
                # animation interpolation time between frames. Target is 0.1 seconds but will shrink if needed
                transition_time = 0.1
                transition_time = min(0.1, time_between_frames * 0.4)
                
                transition_scale = 0
                if time_since_frame >= transition_time:
                    transition_scale = 1
                else:
                    transition_scale = time_since_frame / transition_time

                if sound_index_temp == 0:
                    interpolated_frame = sound_index_temp
                    interpolate = False
                else:
                    interpolated_frame = sound_index_temp - 1 + transition_scale
                    interpolate = True

                anim_fig = plot_quantum_state(pure_probabilities_list, 
                                              pure_phase_angles_list, 
                                              measured_probabilities_list,
                                              interpolated_frame, 
                                              interpolate = interpolate, 
                                              fig = anim_fig, 
                                              zero_noise = zero_noise
                                             )
                return mplfig_to_npimage(anim_fig)

            plot_clips.append(VideoClip(make_plot_frame, duration=(rhythm[sound_index][0] + rhythm[sound_index][1]) / 480))
        else:
            anim_fig.clear()
            plt.cla()
            plt.clf()

            frame_fig = plot_quantum_state(pure_probabilities_list, 
                                           pure_phase_angles_list, 
                                           measured_probabilities_list, 
                                           sound_index,
                                           fig = anim_fig, 
                                           zero_noise = zero_noise
                                          )

            frame_image = mplfig_to_npimage(frame_fig)
            plot_clips.append(ImageClip(frame_image).set_duration((rhythm[sound_index][0] + rhythm[sound_index][1]) / 480))
    
    # calculate the accumulated time for each sampled state in the animation from the rhythm
    accumulated_times = []
    accumulated_times.append(0)
    accumulated_time = 0
    for times in rhythm:
        sound_time = (times[0] + times[1]) / 480.0
        accumulated_time += sound_time
        accumulated_times.append(accumulated_time)

    paths = output_manager.glob('info_panel_*')

    # sort in ascending order of frame number in filename, e.g. info_panel_0, info_panel_1, ...
    paths.sort(key=lambda x: data_manager.extract_natural_number_from_string_end(os.path.splitext(x)[0]))

    # create info panel clips and stack them to the right of the plot clips
    for file_index, file in enumerate(paths):
        frame_time = (rhythm[file_index][0] + rhythm[file_index][1]) / 480.0
        info_panel_clip = ImageClip(file).set_duration(frame_time).resize(height = plot_clips[file_index].size[1])
        
        plot_clips[file_index] = clips_array([[plot_clips[file_index], info_panel_clip]], bg_color=bg_color)

    plot_info_clip = concatenate(plot_clips, method="compose")

    # for target height 1080 pixels, the plot_info_clip should now be 1920 x 720 pixels
    if invert_colours == True:
        plot_info_clip = invert_colors.invert_colors(plot_info_clip)

    video_duration = plot_info_clip.duration
    plot_info_clip_height = plot_info_clip.size[1]

    partial_circ_image_clips = []
    positions_x = []
    accumulated_width = 0

    image_barrier_clip = ImageClip(output_manager.get_path('partial_circ_barrier.png')).set_duration(video_duration)

    # crop the horizontal white space from the sides of the barrier image
    if image_barrier_clip.size[0] > 156:
        image_barrier_clip = crop.crop(image_barrier_clip, x1=133, x2=image_barrier_clip.size[0]-23)
    image_barrier_clip = image_barrier_clip.resize(height=1080 - plot_info_clip_height)

    barrier_image_width = image_barrier_clip.size[0]
    barrier_image_height = image_barrier_clip.size[1]
    barrier_start_y = int(43.0 * barrier_image_height / 454.0)
    barrier_end_y = int(25.0 * barrier_image_height / 454.0)

    image_empty_clip = ImageClip(output_manager.get_path('partial_circ_empty.png')).set_duration(video_duration)
    
    # crop the horizontal white space from the sides of the empty circuit image
    if image_empty_clip.size[0] > 156:
        image_empty_clip = crop.crop(
            image_empty_clip, x1=133, x2=image_empty_clip.size[0]-23)
    image_empty_clip = image_empty_clip.resize(height=1080 - plot_info_clip_height)

    image_empty_clip_array = clips_array([[image_empty_clip, image_empty_clip]], bg_color=bg_color)
    image_empty_clip_array = crop.crop(image_empty_clip_array, x1=0, x2=barrier_image_width)

    for i, partial_circ in enumerate(partial_circuit_list):
        new_image_clip = ImageClip(output_manager.get_path(f'partial_circ_{i}.png')).set_duration(video_duration)
        if new_image_clip.size[0] > 156:
            new_image_clip = crop.crop(
                new_image_clip, x1=133, x2=new_image_clip.size[0]-23)
        new_image_clip = new_image_clip.resize(height=1080 - plot_info_clip_height)
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
        [[x for x in all_clips]], bg_color=bg_color)
    circ_clip_arr.fps = fps
    composited_with_barrier_clips = []
    composited_with_barrier_clips.append(circ_clip_arr)
    note_accumulated_info = []  # accumulated time, note length, note rest
    accumulated_time = 0
    for iter in range(len(rhythm)):
        # new_barrier_clip = image_barrier_clip.set_start(accumulated_time).set_end(min(video_duration, accumulated_time + 1 / 4.0)).set_position((positions_x[iter]-barrier_image_width, 0))
        note_length = rhythm[iter][0] / 480.0
        new_barrier_clip = image_barrier_clip.set_start(0).set_end(min(
            accumulated_time, video_duration)).set_position((int(positions_x[iter]-barrier_image_width), 0))
        note_accumulated_info.append(
            (accumulated_time, rhythm[iter][0] / 480.0, rhythm[iter][1] / 480.0))
        accumulated_time += (rhythm[iter][0] + rhythm[iter][1]) / 480.0
        # new_barrier_clip.add_mask()
        # new_barrier_clip = new_barrier_clip.crossfadeout(note_length)
        composited_with_barrier_clips.append(new_barrier_clip)
    video_duration = video_duration
    video_size = plot_info_clip.size
    circ_clip_arr = CompositeVideoClip(composited_with_barrier_clips)
    circ_clip_arr = circ_clip_arr.resize(height=1080 - plot_info_clip_height)
    vertical_scale = 1080 / float(plot_info_clip_height + circ_clip_arr.size[1])
    circ_clip_target_width = int(1920 / vertical_scale)
    clip_orig_x = circ_clip_arr.size[0]
    circ_clip_arr = circ_clip_arr.margin(left=circ_clip_target_width, right=circ_clip_target_width, color=bg_color)
    # circ_clip_arr = crop.crop(circ_clip_arr, x1 = -clip_orig_x, x2 = 2 * clip_orig_x)
    # circ_clip_arr.save_frame("contatenated_circ.png", t = 0.1)
    h = circ_clip_arr.h
    w = circ_clip_target_width

    print("video_duration:", video_duration)
    print("circ_clip_arr.duration:", circ_clip_arr.duration)

    def f(gf, t):
        x_start = w/2
        time_fraction = t / video_duration
        accumulated_time = 0
        prev_accumulated_time = 0
        for iter in range(len(rhythm)):
            accumulated_time += (rhythm[iter][0] + rhythm[iter][1]) / 480.0
            if t <= accumulated_time:
                break
            prev_accumulated_time = accumulated_time

        if accumulated_time > video_duration or t > video_duration:
            print("ERROR: clip duration is longer than the total time specified by the rhythm: accumulated_time:", accumulated_time, "video_duration:", video_duration, "t:", t)
        
        target_iter = iter
        prev_iter = iter - 1
        prev_time = prev_accumulated_time
        target_time = accumulated_time
        if iter != len(rhythm)-1:
            target_pos = positions_x[iter+1] - barrier_image_width / 2
        else:
            target_pos = positions_x[iter+1]
        prev_pos = positions_x[iter] - barrier_image_width / 2

        if target_time == prev_time:
            print("ERROR: clip duration is longer than the total time specified by the rhythm")
            print("rhythm:", rhythm)
            print("accumulated_time:", accumulated_time)
            print("prev_accumulated_time:", prev_accumulated_time)
            print("video_duration:", video_duration)
            print("t:", t)
            print("target_iter:", target_iter)
        x = int(x_start + (prev_pos + (target_pos - prev_pos) * (t - prev_time) / (target_time - prev_time)))
        y = 0
        return gf(t)[y:y+h, x:x+w]

    circ_clip_arr = circ_clip_arr.fl(f, apply_to="mask")
    if invert_colours == True:
        circ_clip_arr = invert_colors.invert_colors(circ_clip_arr)
    # clip_arr = clips_array([[circuit_video.resize(circuit_rescale)], [video]], bg_color=bg_color)
    clip_arr = clips_array([[circ_clip_arr], [plot_info_clip]], bg_color=bg_color)
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
        
        idle_colour = [127, 127, 127]
        lerp_time = (time_since_note_played - time_to_start_fade) / highlight_fade_time
        scale = (2.0 * (1 - fidelity) - 1.0)  # -1 to 1
        scale = np.tanh(scale) / np.tanh(1)
        scale = (scale + 1.0) / 2.0
        highlight_colour = [int(255 * cmap_needle(scale)[i]) for i in range(3)]
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
