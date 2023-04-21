# Methods relating to the generation of video from sampled density matrices
import collections.abc
import math
import os
import re
from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import matplotlib
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import qiskit
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
from moviepy.editor import (
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    clips_array,
    concatenate,
)
from moviepy.video.fx import crop, freeze, invert_colors
from moviepy.video.io.bindings import mplfig_to_npimage
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

import qmuvi.data_manager as data_manager
import qmuvi.musical_processing as musical_processing

Colour = Iterable[Union[int, float]]


def convert_colour_to_uint8(colour: Colour) -> Colour:
    """Converts a colour to the uint8 format used by matplotlib.

    Parameters
    ----------
        colour
            The colour to convert. This can be an iterable of ints or floats.

    Returns
    -------
        The converted colour, represented as an iterable of uint8s.
    """
    if isinstance(colour[0], int):
        return colour
    return [int(c * 255) for c in colour]


def convert_colour_to_float(colour: Colour) -> Colour:
    """Converts a colour to the float range [0, 1] used by matplotlib.

    Parameters
    ----------
        colour
            The colour to convert. This can be an iterable of ints or floats.

    Returns
    -------
        The converted colour, represented as an iterable of floats.
    """
    if isinstance(colour[0], float):
        return colour
    return [c / 255.0 for c in colour]


def filter_blend_colours(image: Iterable[Iterable[Colour]], colour: Colour, alpha: float) -> Iterable[Iterable[Colour]]:
    """Applies alpha blending to each pixel in an image using a specified colour and alpha value.

    This function takes an image represented as a 2D array of colours, and applies alpha blending to
    each individual pixel in the bitmap. The blending is done using a specified colour and alpha value,
    with the resulting colour being a weighted average of the original pixel colour and the specified colour.
    The blended colour will have the minimum number of channels of the two input colours. For example, blending
    an RGB colour with an RGBA colour will result in an RGB colour.

    Parameters
    ----------
        image
            The image to be blended.
        colour
            The colour to blend with the image.
        alpha
            A value ranging from 0.0 (fully pixel colour) to 1.0 (fully colour) that specifies the weight of the
            colour in the blend. If the alpha value is outside the [0, 1] range, it will be clamped to this range.

    Returns
    -------
        The filtered image, represented as a 2D array of colours.
    """

    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = blend_colours(image[i][j], colour, alpha)

    return image


def blend_colours(colour1: Colour, colour2: Colour, alpha: float) -> Colour:
    """Blends two colours together using the specified alpha value as a weighted average.

    The blended colour will have the minimum number of channels of the two input colours. For example, blending an RGB colour
    with an RGBA colour will result in an RGB colour.

    Parameters
    ----------
        colour1
            The first colour to be blended.
        colour2
            The second colour to be blended.
        alpha
            A value ranging from 0.0 (fully colour1) to 1.0 (fully colour2) that specifies the weight of the
            second colour in the blend. If the alpha value is outside the [0, 1] range, it will be clamped to this range.

    Returns
    -------
        The blended colour with the lesser number of channels of the two input colours.
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    if isinstance(colour1[0], int):
        return [int((1 - alpha) * colour1[i] + (alpha) * colour2[i]) for i in range(min(len(colour2), len(colour1)))]
    else:
        return [((1 - alpha) * colour1[i] + (alpha) * colour2[i]) for i in range(min(len(colour2), len(colour1)))]


def filter_colour_multiply(image: Iterable[Iterable[Colour]], colour: Colour) -> Iterable[Iterable[Colour]]:
    """Multiplies each pixel in an image with a specified colour.

    This function takes an image represented as a 2D array of colours and multiplies each individual pixel in the
    bitmap with a specified colour. The resulting colour is obtained by multiplying the values of each colour channel
    of the original pixel with the corresponding values in the specified colour. The blended pixel colours will have
    the lesser number of channels of the two input colours. For example, blending an RGB colour
    with an RGBA colour will result in an RGB colour.

    Parameters
    ----------
        image
            The image to be filtered.
        colour
            The colour to multiply with the image.

    Returns
    -------
        The filtered image, represented as a 2D Iterable of colours. The blended pixel colours will have the lesser number of channels of the two input colours.
    """
    if isinstance(colour[0], int):
        for i in range(len(image)):
            for j in range(len(image[i])):
                image[i][j] = [int(image[i][j][k] * (colour[k] / 255.0)) for k in range(min(len(colour), len(image[i][j])))]
    else:
        for i in range(len(image)):
            for j in range(len(image[i])):
                image[i][j] = [image[i][j][k] * colour[k] for k in range(min(len(colour), len(image[i][j])))]

    return image


def lerp(start_values: Iterable[float], end_values: Iterable[float], t: float) -> List[float]:
    """Interpolates between two iterable sequences of values using linear interpolation.

    This function takes two iterables of start and end values and a float representing a time t in the range [0, 1],
    and performs linear interpolation between the two values based on the value of t. The result of the linear
    interpolation is an iterable of floats that is the same length as the input iterables.

    Parameters
    ----------
        start_values
            The start values for the linear interpolation. An iterable of floats.
        end_values
            The end values for the linear interpolation. An iterable of floats.
        t
            A float in the range [0, 1] that represents the time between the start and end values. If t=0, the result
            will be equal to the start values, if t=1, the result will be equal to the end values. If t is outside the
            range [0, 1], it will be clamped.

    Returns
    -------
        A list of floats representing the result of the linear interpolation between the start and end values.
        The resulting list will have the same length as the input iterables.
    """
    t = np.clip(t, 0.0, 1.0)

    return [((1 - t) * a) + (t * b) for a, b in zip(start_values, end_values)]


def ease_in(start_values: Iterable[float], end_values: Iterable[float], t: float) -> List[float]:
    """Interpolates between two values using an ease-in curve.

    This function takes two iterables of start and end values and a float representing a time t in the range [0, 1],
    and performs interpolation between the two values using an ease-in curve based on the value of t. The ease-in curve
    starts slowly and speeds up as it approaches the end value. The result of the interpolation is an iterable of floats
    that is the same length as the input iterables.

    Parameters
    ----------
        start_values
            The start values for the interpolation. An iterable of floats.
        end_values
            The end values for the interpolation. An iterable of floats.
        t
            A float in the range [0, 1] that represents the time between the start and end values. If t=0, the result
            will be equal to the start values, if t=1, the result will be equal to the end values. If t is outside the
            range [0, 1], it will be clamped.

    Returns
    -------
        A list of floats representing the result of the interpolation using an ease-in curve between the start
        and end values. The resulting list will have the same length as the input iterables.

    """
    t = np.clip(t, 0.0, 1.0)
    scaled_t = 1 - math.cos(t * math.pi / 2.0)

    return [((1 - scaled_t) * a) + (scaled_t * b) for a, b in zip(start_values, end_values)]


def ease_out(start_values: Iterable[float], end_values: Iterable[float], t: float) -> List[float]:
    """Interpolates between two values using an ease-out curve.

    This function takes two iterables of start and end values and a float representing a time t in the range [0, 1],
    and performs interpolation between the two values using an ease-out curve based on the value of t. The ease-out curve
    starts quickly and slows down as it approaches the end value. The result of the interpolation is an iterable of
    floats that is the same length as the input iterables.

    Parameters
    ----------
        start_values
            The start values for the interpolation. An iterable of floats.
        end_values
            The end values for the interpolation. An iterable of floats.
        t
            A float in the range [0, 1] that represents the time between the start and end values. If t=0, the result
            will be equal to the start values, if t=1, the result will be equal to the end values. If t is outside the
            range [0, 1], it will be clamped.

    Returns
    -------
        A list of floats representing the result of the interpolation using an ease-out curve between the start
        and end values. The resulting list will have the same length as the input iterables.

    """
    t = np.clip(t, 0.0, 1.0)
    scaled_t = np.sin(t * np.pi / 2.0)

    return [((1 - scaled_t) * a) + (scaled_t * b) for a, b in zip(start_values, end_values)]


def invert_cmap(cmap: matplotlib.colors.Colormap) -> matplotlib.colors.ListedColormap:
    """
    Inverts the colors of a Matplotlib colormap.

    This function takes a Matplotlib colormap object and returns a new colormap with the colors inverted. The inverted
    colormap is created by subtracting the red, green, and blue components of each color from 1.0.

    Parameters
    ----------
        cmap
            The colormap to invert.

    Returns
    -------
        The inverted colormap as a matplotlib.colors.ListedColormap object.
    """
    newcolors = cmap(np.linspace(0, 1, 256))
    for i in range(256):
        newcolors[i, :] = np.array([1 - cmap(i / 256)[0], 1 - cmap(i / 256)[1], 1 - cmap(i / 256)[2], 1])

    return ListedColormap(newcolors)


def reverse_cmap(cmap: matplotlib.colors.Colormap) -> matplotlib.colors.ListedColormap:
    """
    Reverses the order of a Matplotlib colormap.

    This function takes a Matplotlib colormap object and returns a new colormap with the order of the colors reversed.

    Parameters
    ----------
        cmap
            The colormap to reverse.

    Returns
    -------
        The reversed colormap as a matplotlib.colors.ListedColormap object.
    """
    # Generate a new set of colors for the reversed colormap using the existing colormap and a linearly spaced array
    # of values between 0 and 1.
    newcolors = cmap(np.linspace(0, 1, 256))

    # Use a reversed index to select the color at the opposite end of the colormap for each position in the new colormap.
    reversed_i = 255
    for i in range(256):
        # Dividing by 255 instead of 256 so that values include both 0 and 1.
        newcolors[i, :] = np.array([cmap(reversed_i / 255)[0], cmap(reversed_i / 255)[1], cmap(reversed_i / 255)[2], 1])
        reversed_i -= 1
    return ListedColormap(newcolors)


def convert_fig_pixels_to_inches(pixels: float) -> float:
    """Converts a size in pixels to inches based on the dpi setting in matplotlib.

    This function takes a float representing a size in pixels and returns a float representing the equivalent size in
    inches, based on the current dpi (dots per inch) setting for matplotlib figures.

    Parameters
    ----------
        pixels
            The size in pixels to be converted to inches.

    Returns
    -------
        The equivalent size in inches.
    """
    dpi = plt.rcParams["figure.dpi"]
    return pixels / dpi


def generate_video_from_data(
    quantum_circuit: qiskit.QuantumCircuit,
    output_manager: data_manager.DataManager,
    rhythm: Optional[List[Tuple[int, int]]],
    invert_colours: bool = False,
    fps: int = 24,
    vpr: Optional[Union[float, Callable[[int], float]]] = 1.0 / 3.0,
    smooth_transitions: bool = True,
    show_measured_probabilities_only: bool = False,
) -> None:
    """Uses generated qMuVi data and wav file to generate a video (.mp4).

    Parameters
    ----------
        quantum_circuit
            The qiskit QuantumCircuit.
        output_manager
            The data manager to use for saving the video and all of its pieces.
        rhythm
            A list of tuples for the length and rest times of each sound in units of ticks (480 ticks is 1 second).
            If None, then each sound length and rest time will be set to (note_sound, rest_time) = (240, 0)
        invert_colours
            Whether to render the video in dark mode.
        fps
            The frames per second of the output video.
        vpr
            Vertical proportion ratio. The propotion of vertical space that the circuit will occupy. Can be a float or a function that maps qubit_count (int) to float.
        smooth_transitions
            Whether to animate the plots by interpolating between qMuVi data samples. Significantly increases render time.
        show_measured_probabilities_only
            Whether to only plot the basis state probabilities.
    """

    if vpr is None:

        def vpr(n):
            return 1.0 / 3.0

    elif isinstance(vpr, float):
        vpr_temp = vpr

        def vpr(n):
            return vpr_temp

    elif not callable(vpr):
        raise TypeError("vpr must be a float, None or a Callable[[int], float]")

    sounds_list = output_manager.load_json("sounds_list.json")
    fidelity_list = output_manager.load_json("fidelity_list.json")
    measured_probabilities_list = output_manager.load_json("meas_probs_list.json")

    if rhythm is None:
        rhythm = [(240, 0)] * len(sounds_list)

    # format loaded data
    for sound_index in range(len(sounds_list)):
        for pure_state_info_index in range(len(sounds_list[sound_index])):
            # pure_state_prob: float
            sounds_list[sound_index][pure_state_info_index][0] = float(sounds_list[sound_index][pure_state_info_index][0])
            # pure_state_info: Dict[int basis_state_number, Tuple[float basis_state_prob, float basis_state_angle]], # where (basis_state_prob > eps)
            pure_state_info = {}
            for basis_state_number in sounds_list[sound_index][pure_state_info_index][1].keys():
                pure_state_info[int(basis_state_number)] = (
                    float(sounds_list[sound_index][pure_state_info_index][1][basis_state_number][0]),
                    float(sounds_list[sound_index][pure_state_info_index][1][basis_state_number][1]),
                )
            sounds_list[sound_index][pure_state_info_index][1] = pure_state_info
            # all_basis_state_probs: List[float basis_state_prob]
            for basis_state_index in range(len(sounds_list[sound_index][pure_state_info_index][2])):
                sounds_list[sound_index][pure_state_info_index][2][basis_state_index] = float(
                    sounds_list[sound_index][pure_state_info_index][2][basis_state_index]
                )
            # all_basis_state_angles: List[float basis_state_angle]
            for basis_state_index in range(len(sounds_list[sound_index][pure_state_info_index][3])):
                sounds_list[sound_index][pure_state_info_index][3][basis_state_index] = float(
                    sounds_list[sound_index][pure_state_info_index][3][basis_state_index]
                )

    zero_noise = True
    for i in range(len(sounds_list)):
        if len(sounds_list[i]) > 1:
            zero_noise = False
            break

    dag = circuit_to_dag(quantum_circuit)
    qubit_count = len(dag.qubits)

    circuit_layers_per_line = 50
    quantum_circuit.draw(filename=output_manager.get_path("circuit.png"), output="mpl", fold=circuit_layers_per_line)

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
        partial_circ.draw(filename=output_manager.get_path(f"partial_circ_{i}.png"), output="mpl", fold=-1)

    # an empty circuit is used to create space in the circuit
    empty_circuit = QuantumCircuit(qubit_count)
    empty_circuit.draw(filename=output_manager.get_path("partial_circ_empty.png"), output="mpl", fold=-1)

    # a barrier circuit is inserted between partial circuits
    barrier_circuit = QuantumCircuit(qubit_count)
    barrier_circuit.barrier()
    barrier_circuit.draw(filename=output_manager.get_path("partial_circ_barrier.png"), output="mpl", fold=-1)

    # Fidelity colour bar
    cmap_fidelity = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "violet", "blue"])
    # Musical needle
    cmap_needle = reverse_cmap(cmap_fidelity)
    # Phase for bar plot bars
    cmap_phase = cm.get_cmap("hsv")
    # Matplotlib plot ticks
    tick_colour = [0.4, 0.4, 0.4, 1.0]

    fidelity_line_colour = [0.6, 0.6, 0.6, 1.0]

    bg_color = [0xFF, 0xFF, 0xFF]

    if invert_colours is True:
        cmap_fidelity = invert_cmap(cmap_fidelity)
        cmap_phase = invert_cmap(cmap_phase)
        # Note: cmap_needle does not need to be inverted since needle is not part of the image that is inverted

    print("Preparing pieces for video generation...")

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
                pure_probabilities[pure_state_index, :] = (
                    np.array(sounds_list[sound_index][pure_state_index][2]) * sounds_list[sound_index][pure_state_index][0]
                )
                pure_phase_angles[pure_state_index, :] = sounds_list[sound_index][pure_state_index][3]

        pure_probabilities_list.append(pure_probabilities)
        pure_phase_angles_list.append(pure_phase_angles)

    plot_clips = []
    for sound_index, sound_data in enumerate(sounds_list):
        # plot and save the info panel image
        _plot_info_panel(
            output_manager,
            None,
            pure_probabilities_list[sound_index],
            pure_phase_angles_list[sound_index],
            fidelity_list[sound_index],
            sound_index,
            qubit_count,
            cmap_phase,
            cmap_fidelity,
            tick_colour,
            fidelity_line_colour,
            invert_colours,
            vpr,
            not show_measured_probabilities_only,
        )

        # histograms
        target_video_height_pixels = 1080
        dpi = plt.rcParams["figure.dpi"]
        target_empty_circuit_height_pixels = target_video_height_pixels * (1 - vpr(qubit_count))
        fig_height_inches = (1 - vpr(qubit_count)) * 13.5
        fig_height_pixels = fig_height_inches * dpi
        scale = target_empty_circuit_height_pixels / fig_height_pixels

        scaled_dpi = dpi * scale

        anim_fig = plt.figure(figsize=(20, (1 - vpr(qubit_count)) * 13.5), dpi=scaled_dpi)
        anim_fig.set_dpi(scaled_dpi)

        if smooth_transitions is True:

            def make_plot_frame(time_since_frame, sound_index_temp=sound_index, anim_fig=anim_fig):
                anim_fig.clear()
                plt.cla()
                plt.clf()

                sound_time = (rhythm[sound_index_temp][0] + rhythm[sound_index_temp][1]) / 480.0
                time_between_frames = sound_time

                # animation interpolation time between frames. Target is 0.10 seconds but will shrink if needed
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

                anim_fig = _plot_quantum_state(
                    anim_fig,
                    pure_probabilities_list,
                    pure_phase_angles_list,
                    measured_probabilities_list,
                    interpolated_frame,
                    qubit_count,
                    interpolate,
                    cmap_phase,
                    tick_colour,
                    vpr,
                    zero_noise,
                    show_measured_probabilities_only,
                )
                return mplfig_to_npimage(anim_fig)

            plot_clips.append(VideoClip(make_plot_frame, duration=(rhythm[sound_index][0] + rhythm[sound_index][1]) / 480))
        else:
            anim_fig.clear()
            plt.cla()
            plt.clf()

            anim_fig = _plot_quantum_state(
                anim_fig,
                pure_probabilities_list,
                pure_phase_angles_list,
                measured_probabilities_list,
                sound_index,
                qubit_count,
                False,
                cmap_phase,
                tick_colour,
                vpr,
                zero_noise,
                show_measured_probabilities_only,
            )

            frame_image = mplfig_to_npimage(anim_fig)
            plot_clips.append(ImageClip(frame_image).set_duration((rhythm[sound_index][0] + rhythm[sound_index][1]) / 480))

    # calculate the accumulated time for each sampled state in the animation from the rhythm
    accumulated_times = []
    accumulated_times.append(0)
    accumulated_time = 0
    for times in rhythm:
        sound_time = (times[0] + times[1]) / 480.0
        accumulated_time += sound_time
        accumulated_times.append(accumulated_time)

    paths = output_manager.glob("info_panel_*")

    # sort in ascending order of frame number in filename, e.g. info_panel_0, info_panel_1, ...
    paths.sort(key=lambda x: data_manager.extract_natural_number_from_string_end(os.path.splitext(x)[0]))

    # create info panel clips and stack them to the right of the plot clips
    for file_index, file in enumerate(paths):
        frame_time = (rhythm[file_index][0] + rhythm[file_index][1]) / 480.0
        info_panel_clip = ImageClip(file).set_duration(frame_time).resize(height=plot_clips[file_index].size[1])

        plot_clips[file_index] = clips_array([[plot_clips[file_index], info_panel_clip]], bg_color=bg_color)

    plot_info_clip = concatenate(plot_clips, method="compose")

    # for target height 1080 pixels, the plot_info_clip should now be 1920 x 720 pixels
    if invert_colours is True:
        plot_info_clip = invert_colors.invert_colors(plot_info_clip)

    video_duration = plot_info_clip.duration
    plot_info_clip_height = plot_info_clip.size[1]
    circuit_anim_height = 1080 - plot_info_clip_height

    image_barrier_clip_base = ImageClip(output_manager.get_path("partial_circ_barrier.png")).set_duration(video_duration)

    # crop the horizontal white space from the sides of the barrier image
    if image_barrier_clip_base.size[0] > 156:
        image_barrier_clip_base = crop.crop(image_barrier_clip_base, x1=133, x2=image_barrier_clip_base.size[0] - 23)
    image_barrier_clip_base = image_barrier_clip_base.resize(height=circuit_anim_height)

    barrier_image_width = image_barrier_clip_base.size[0]
    barrier_image_height = image_barrier_clip_base.size[1]
    barrier_start_y = int(43.0 * barrier_image_height / 454.0)
    barrier_end_y = int(25.0 * barrier_image_height / 454.0)

    image_empty_clip = ImageClip(output_manager.get_path("partial_circ_empty.png")).set_duration(video_duration)

    # crop the horizontal white space from the sides of the empty circuit image
    if image_empty_clip.size[0] > 156:
        image_empty_clip = crop.crop(image_empty_clip, x1=133, x2=image_empty_clip.size[0] - 23)
    image_empty_clip = image_empty_clip.resize(height=circuit_anim_height)

    image_empty_clip_array = clips_array([[image_empty_clip, image_empty_clip]], bg_color=bg_color)
    image_empty_clip_array = crop.crop(image_empty_clip_array, x1=0, x2=barrier_image_width)

    partial_circuit_clip_list = []
    positions_x = []
    accumulated_width = 0
    for i, partial_circ in enumerate(partial_circuit_list):
        partial_circuit_clip = ImageClip(output_manager.get_path(f"partial_circ_{i}.png")).set_duration(video_duration)
        if partial_circuit_clip.size[0] > 156:
            partial_circuit_clip = crop.crop(partial_circuit_clip, x1=133, x2=partial_circuit_clip.size[0] - 23)
        partial_circuit_clip = partial_circuit_clip.resize(height=circuit_anim_height)

        if i != len(partial_circuit_list) - 1:
            accumulated_width += partial_circuit_clip.size[0] + barrier_image_width
        else:
            accumulated_width += partial_circuit_clip.size[0]

        positions_x.append(accumulated_width)
        partial_circuit_clip_list.append(partial_circuit_clip)

    all_circuit_clip_list = []
    for i in range(len(partial_circuit_clip_list)):
        all_circuit_clip_list.append(partial_circuit_clip_list[i])
        if i != len(partial_circuit_clip_list) - 1:
            all_circuit_clip_list.append(image_empty_clip_array)

    full_circuit_clip = clips_array([[x for x in all_circuit_clip_list]], bg_color=bg_color)

    full_circuit_clip.fps = fps
    full_circuit_and_barrier_clips_list = []
    full_circuit_and_barrier_clips_list.append(full_circuit_clip)

    accumulated_time_info = []  # tuple: (accumulated_time, note_length, note_rest). Used in drawing of needle
    accumulated_time = 0
    for sound_index in range(len(rhythm)):
        note_length = rhythm[sound_index][0] / 480.0
        note_rest = rhythm[sound_index][1] / 480.0
        barrier_clip = (
            image_barrier_clip_base.set_start(0)
            .set_end(min(accumulated_time, video_duration))
            .set_position((int(positions_x[sound_index] - barrier_image_width), 0))
        )
        accumulated_time_info.append((accumulated_time, note_length, note_rest))
        accumulated_time += note_length + note_rest
        full_circuit_and_barrier_clips_list.append(barrier_clip)

    full_circuit_with_barriers_clip = CompositeVideoClip(full_circuit_and_barrier_clips_list)

    output_width = 1920

    # add margins so that the circuit can slide over the rendered output (I don't think it's necessary)
    full_circuit_with_barriers_clip = full_circuit_with_barriers_clip.margin(left=output_width, right=output_width, color=bg_color)

    # TODO: change pan effect so that position is changed similar to
    # clip.set_position(lambda t: ('center', 50+t) ) https://zulko.github.io/moviepy/getting_started/compositing.html
    # This is so that the full extended margin video above doesn't needs to be rendered and a time and frame effect doesn't need to be applied, which can be slow.
    def circuit_clip_animate_pan_effect(gf: Callable[[float], np.ndarray], t: float) -> np.ndarray:
        """Animates the circuit clip panning to the rhythm.

        Parameters
        ----------
            gf
                the clips get_frame method that returns the frame of the clip at time t.
            t
                the time in seconds.

        Returns
        -------
            the frame of the clip at time t.
        """
        x_start = output_width / 2
        next_accumulated_time = 0
        prev_accumulated_time = 0
        frame_sound_index = len(rhythm) - 1
        for sound_index in range(len(rhythm)):
            prev_accumulated_time = next_accumulated_time
            next_accumulated_time += (rhythm[sound_index][0] + rhythm[sound_index][1]) / 480.0
            if t <= next_accumulated_time:
                frame_sound_index = sound_index
                break

        if frame_sound_index != len(rhythm) - 1:
            next_pos = positions_x[frame_sound_index + 1] - barrier_image_width / 2
        else:
            next_pos = positions_x[frame_sound_index + 1]
        prev_pos = positions_x[frame_sound_index] - barrier_image_width / 2

        x = int(x_start + (prev_pos + (next_pos - prev_pos) * (t - prev_accumulated_time) / (next_accumulated_time - prev_accumulated_time)))
        y = 0
        return gf(t)[y : y + circuit_anim_height, x : x + output_width]

    full_circuit_animated_clip = full_circuit_with_barriers_clip.fl(circuit_clip_animate_pan_effect, apply_to="mask")
    if invert_colours is True:
        full_circuit_animated_clip = invert_colors.invert_colors(full_circuit_animated_clip)

    complete_clip = clips_array([[full_circuit_animated_clip], [plot_info_clip]], bg_color=bg_color)

    video_final_silent = crop.crop(
        complete_clip, x1=int(complete_clip.size[0] / 2 - output_width / 2), x2=int(complete_clip.size[0] / 2 + output_width / 2)
    )

    # add audio
    paths = output_manager.glob(f"{output_manager.default_name}-*.wav")
    audio_file_clips = []
    for path in paths:
        path_multiplatform = path.replace("\\", "/")
        audio_file_clip = mpy.AudioFileClip(path_multiplatform, nbytes=4, fps=44100)
        audio_file_clips.append(audio_file_clip)
    composed_audio_clip = CompositeAudioClip(audio_file_clips)

    video_final = video_final_silent.set_audio(composed_audio_clip)

    video_final_width = video_final.size[0]

    def draw_needle_effect(get_frame, t) -> np.ndarray:
        """Draw a rectangle in the frame on top of the circuit animated clip.

        Parameters
        ----------
            get_frame
                the clips get_frame method that returns the frame of the clip at time t.
            t
                the time in seconds.

        Returns
        -------
            the frame of the clip at time t.
        """
        # change (top, bottom, left, right) to the coordinates
        top = 1
        # make the gap between the bottom and the barrier the same as the gap between the top and the barrier.
        bottom = int(full_circuit_animated_clip.size[1] - 1 + (barrier_start_y - barrier_end_y))
        left = int(video_final_width / 2 - 9)
        right = int(video_final_width / 2 + 9)
        # get accumulated time, sound times and fidelity for the current frame
        current_time_info = accumulated_time_info[-1]
        fidelity = fidelity_list[-1]
        for sound_index, sound_info in enumerate(accumulated_time_info):
            if sound_info[0] > t:
                current_time_info = accumulated_time_info[sound_index - 1]
                fidelity = fidelity_list[sound_index - 1]
                break

        time_since_sound_played = t - current_time_info[0]
        highlight_fade_time = current_time_info[1]

        idle_colour = [127, 127, 127]
        lerp_time = time_since_sound_played / highlight_fade_time
        scale = 2.0 * (1 - fidelity) - 1.0  # fidelity from [0, 1] to scale [1, -1]
        scale = np.tanh(scale) / np.tanh(1)
        scale = (scale + 1.0) / 2.0
        highlight_colour = [int(255 * cmap_needle(scale)[i]) for i in range(3)]
        lerp_colour = [int(x) for x in ease_out(highlight_colour, idle_colour, lerp_time)]

        frame = get_frame(t)
        frame[top : top + 3, left:right] = lerp_colour
        frame[bottom - 3 : bottom, left:right] = lerp_colour
        frame[top + 3 : bottom, left : left + 3] = lerp_colour
        frame[top + 3 : bottom, right - 3 : right] = lerp_colour

        return frame

    video_final = video_final.fl(draw_needle_effect)

    script_path = os.path.dirname(os.path.abspath(__file__))

    asset_path = os.path.join(script_path, "package_data", "assets")

    inverted_img = "white.png"
    non_inverted_img = "black.png"

    if invert_colours is True:
        generated_title = (
            ImageClip(os.path.join(asset_path, inverted_img))
            .set_duration(video_final.duration)
            .set_pos(("right", "bottom"))
            .resize(height=30, width=205)
            .margin(right=28, bottom=2, opacity=0)
        )
    else:
        generated_title = (
            ImageClip(os.path.join(asset_path, non_inverted_img))
            .set_duration(video_final.duration)
            .set_pos(("right", "bottom"))
            .resize(height=30, width=205)
            .margin(right=28, bottom=2, opacity=0)
        )

    video_final = CompositeVideoClip([video_final, generated_title])

    # preset options (speed vs filesize): ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
    video_final.write_videofile(
        output_manager.get_default_file_pathname() + ".mp4",
        preset="ultrafast",
        fps=fps,
        codec="mpeg4",
        audio_fps=44100,
        audio_codec="libmp3lame",
        audio_bitrate="3000k",
        audio_nbytes=4,
        ffmpeg_params=["-b:v", "12000K", "-b:a", "3000k"],
    )


def _plot_phase_wheel(
    fig: matplotlib.figure.Figure,
    probabilities: List[float],
    phase_angles: List[float],
    cmap_phase: matplotlib.colors.Colormap,
    tick_colour: Colour,
    invert_colours: bool,
) -> Dict[str, matplotlib.axes.Axes]:
    """Plots a phase wheel with line indicators on it for the given phase_angles and returns a dictionary of the axes.

    This method plots a phase wheel with line indicators on it for the given phase_angles. The phase wheel is a polar
    plot with the total probability of each phase as the radial axis and the phase angle as the angular axis. The line
    indicators are plotted as a series of lines that extend from the centre of the plot to the outer edge of the plot
    at the given phase angle. The line indicators are coloured according to the angle of the phase.

    Parameters
    ----------
        fig
            a matplotlib figure object.
        probabilities
            a list of floats representing the probability of basis state.
        phase_angles
            a list of floats representing the phase angle of basis state.
        cmap_phase
            a matplotlib colormap object representing the colour map to use for the phase wheel.
        tick_colour
            the colour of the tick labels.
        invert_colours
            whether to invert the colours of the phase wheel.

    Returns
    -------
        A dictionary mapping the labels to the matplotlib Axes objects generated for the phase wheel. The order of
        the axes is left-to-right and top-to-bottom of their position in the total layout.
    """

    fig_gridspec_phase_wheel = {"bottom": 0.12, "top": 0.44, "left": 0.01, "right": 0.93, "wspace": 0.0, "hspace": 0.0, "height_ratios": [1]}
    plot_mosaic_phase_wheel_dict = fig.subplot_mosaic(
        [
            ["phase_wheel"],
        ],
        gridspec_kw=fig_gridspec_phase_wheel,
        subplot_kw={"projection": "polar"},
    )

    plt.sca(plot_mosaic_phase_wheel_dict["phase_wheel"])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    azimuths = np.arange(0, 361, 1)
    values = azimuths * np.ones((30, 361))
    azimuths = azimuths * np.pi / 180.0
    zeniths = np.arange(40, 70, 1)
    plot_mosaic_phase_wheel_dict["phase_wheel"].pcolormesh(azimuths, zeniths, np.roll(values, 180), cmap=cmap_phase)
    plot_mosaic_phase_wheel_dict["phase_wheel"].fill_between(azimuths, 40, color="#FFFFFF")

    plot_mosaic_phase_wheel_dict["phase_wheel"].plot(azimuths, [40] * 361, color=tick_colour, lw=1)

    # Plot the phase angle indicators.
    # Need to bin the angles to determine the total probability of each bin, then plot the lines with the correct length.
    angle_bin_count = 180
    angles_bin = list(range(angle_bin_count))
    phase_angle_probabilities = {}
    for angle in angles_bin:
        phase_angle_probabilities[angle] = 0

        for angle_iter, phase_angle in enumerate(phase_angles):
            converted_angle = round((angle_bin_count * (phase_angle + np.pi) / (2 * np.pi))) % angle_bin_count
            if angle == converted_angle:
                phase_angle_probabilities[angle] += probabilities[angle_iter]

    highest_probability = max(phase_angle_probabilities.values())
    for angle_iter, phase_angle in enumerate(phase_angles):
        if probabilities[angle_iter] > 0.0001:
            colour_value = (phase_angle + np.pi) / (2 * np.pi)
            converted_angle = round((angle_bin_count * (phase_angle + np.pi) / (2 * np.pi))) % angle_bin_count
            length = phase_angle_probabilities[converted_angle] / highest_probability
            plot_mosaic_phase_wheel_dict["phase_wheel"].plot([phase_angle] * 40, length * np.arange(0, 40, 1), color=cmap_phase(colour_value), lw=2)

    if invert_colours is True:
        plot_mosaic_phase_wheel_dict["phase_wheel"].spines["polar"].set_color(tick_colour)

    plot_mosaic_phase_wheel_dict["phase_wheel"].set_yticks([])
    plot_mosaic_phase_wheel_dict["phase_wheel"].tick_params(axis="x", colors=tick_colour)
    plot_mosaic_phase_wheel_dict["phase_wheel"].tick_params(axis="y", colors=tick_colour)
    fig.text(0.82, 0.465, "Phase", ha="right", va="bottom", fontsize=20)

    label_positions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"]
    plot_mosaic_phase_wheel_dict["phase_wheel"].set_xticks(label_positions, labels)
    plot_mosaic_phase_wheel_dict["phase_wheel"].xaxis.set_tick_params(pad=8)

    return plot_mosaic_phase_wheel_dict


def _plot_stat_bars(
    fig: matplotlib.figure.Figure,
    fidelity: float,
    cmap_fidelity: matplotlib.colors.Colormap,
    c_gray: Colour,
    tick_colour: Colour,
    invert_colours: bool,
) -> Dict[str, matplotlib.axes.Axes]:
    """Plots a fidelity bar and returns a dictionary of the axes.

    Parameters
    ----------
        fig
            The figure to add the bar to.
        fidelity
            The fidelity value to plot.
        cmap_fidelity
            The colormap to use for the fidelity bar.
        c_gray
            The color of the fidelity line.
        tick_colour
            The color of the ticks.
        invert_colours
            Whether or not to invert the colors of the plot.

    Returns
    -------
        A dictionary mapping the labels to the matplotlib Axes objects generated for the fidelity bar. The order of
        the axes is left-to-right and top-to-bottom of their position in the total layout.
    """

    fig_gridspec_stat_bars = {
        "bottom": (0.95 - 0.08) / 2 + 0.08 + 0.02,
        "top": 0.95,
        "left": 0.01,
        "right": 0.93,
        "wspace": 0.0,
        "hspace": 0.0,
        "height_ratios": [1],
    }
    plot_mosaic_stat_bars_dict = fig.subplot_mosaic(
        [["fidelity"]],
        gridspec_kw=fig_gridspec_stat_bars,
    )

    plt.sca(plot_mosaic_stat_bars_dict["fidelity"])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plot_mosaic_stat_bars_dict["fidelity"].imshow(
        np.array(list(reversed([[val] * 6 for val in reversed(np.linspace(0, 1, 100))]))), cmap=cmap_fidelity, interpolation="bicubic"
    )

    line_y = fig_gridspec_stat_bars["bottom"] + (fig_gridspec_stat_bars["top"] - fig_gridspec_stat_bars["bottom"]) * fidelity
    line_middle_x = fig_gridspec_stat_bars["left"] + (fig_gridspec_stat_bars["right"] - fig_gridspec_stat_bars["left"]) / 2
    line = Line2D([line_middle_x - 0.035, line_middle_x + 0.035], [line_y, line_y], lw=4, color=c_gray, alpha=1)
    line.set_clip_on(False)
    fig.add_artist(line)
    plot_mosaic_stat_bars_dict["fidelity"].tick_params(axis="x", colors=tick_colour)
    plot_mosaic_stat_bars_dict["fidelity"].tick_params(axis="y", colors=tick_colour)
    if invert_colours is True:
        plot_mosaic_stat_bars_dict["fidelity"].spines["bottom"].set_color(tick_colour)
        plot_mosaic_stat_bars_dict["fidelity"].spines["top"].set_color(tick_colour)
        plot_mosaic_stat_bars_dict["fidelity"].spines["left"].set_color(tick_colour)
        plot_mosaic_stat_bars_dict["fidelity"].spines["right"].set_color(tick_colour)
        for t in plot_mosaic_stat_bars_dict["fidelity"].xaxis.get_ticklines():
            t.set_color(tick_colour)
        for t in plot_mosaic_stat_bars_dict["fidelity"].yaxis.get_ticklines():
            t.set_color(tick_colour)

    fig.text(0.82, 0.945, "Fidelity", ha="right", va="center", fontsize=20)
    fig.text(0.82, 0.905, format(fidelity, ".2f"), ha="right", va="center", fontsize=20)

    plot_mosaic_stat_bars_dict["fidelity"].xaxis.set_visible(False)
    plot_mosaic_stat_bars_dict["fidelity"].set_ylim((0, 99))
    y_tick_positions = [0, 50, 99]
    y_tick_labels = [0.0, 0.5, 1.0]
    plot_mosaic_stat_bars_dict["fidelity"].set_yticks(y_tick_positions)
    plot_mosaic_stat_bars_dict["fidelity"].set_yticklabels(y_tick_labels)

    return plot_mosaic_stat_bars_dict


def _plot_quantum_state(
    fig: Optional[matplotlib.figure.Figure],
    pure_probabilities_list: List[np.ndarray],
    pure_phase_angles_list: List[np.ndarray],
    measured_probabilities_list: List[np.ndarray],
    sampled_state_number: Union[int, float],
    qubit_count: int,
    interpolate: bool,
    cmap_phase: matplotlib.colors.Colormap,
    tick_colour: Colour,
    vpr: Optional[Union[float, Callable[[int], float]]],
    zero_noise: bool,
    show_measured_probabilities_only: bool,
) -> matplotlib.figure.Figure:
    """Plots the quantum state for given probability and phase distributions.

    Parameters
    ----------
        fig
            The figure to plot the quantum state on. If None, a new figure is created.
        pure_probabilities_list
            A list containing a 2d array for each sampled state, where the array contains probabilities indexed by pure state number and basis state number.
        pure_phase_angles_list
            A list containing a 2d array for each sampled state, where the array contains phase angles indexed by pure state number and basis state number.
        measured_probabilities_list
            A list containing a 1d array for each sampled state, where the array contains measurement probabilities indexed by basis state number.
        sampled_state_number
            The index of the sampled state to be plotted. If interpolate is True, then this can be a decimal fraction.
        qubit_count
            The number of qubits in the quantum state.
        interpolate
            Whether to interpolate `sampled_state_number`.
        cmap_phase
            A colourmap for the phase angle colours in the phase wheel. The colourmap should map normalised phase angles [0,1] to colours.
        tick_colour
            The color of the tick marks.
        vpr
            Propotion of vertical space that the circuit will occupy. Can be a float or a function that maps qubit_count (int) to float.
        zero_noise
            Whether to only plot the data for the single pure state (no noise).
        show_measured_probabilities_only
            Whether to only plot the basis-state probability distribution.

    Returns
    -------
        The figure containing the plot of the quantum state.
    """
    if vpr is None:

        def vpr(n):
            return 1.0 / 3.0

    elif isinstance(vpr, float):
        vpr_temp = vpr

        def vpr(n):
            return vpr_temp

    elif not callable(vpr):
        raise TypeError("vpr must be a float, None or a Callable[[int], float].")

    if fig is None:
        fig = plt.figure(figsize=(20, (1 - vpr(qubit_count)) * 13.5))

    tick_colour = convert_colour_to_float(tick_colour)

    if interpolate is True:
        base_state_count = pure_probabilities_list[0].shape[1]

        # create new vectors that lerp between the sampled states with respect to sampled_state_number
        pure_probabilities_start = pure_probabilities_list[math.floor(sampled_state_number)]
        pure_probabilities_end = pure_probabilities_list[math.ceil(sampled_state_number)]
        pure_probabilities = np.zeros((pure_probabilities_start.shape[0], base_state_count))
        for i in range(pure_probabilities_start.shape[0]):
            pure_probabilities[i, :] = lerp(
                pure_probabilities_start[i, :], pure_probabilities_end[i, :], sampled_state_number - math.floor(sampled_state_number)
            )

        pure_phase_angles_start = pure_phase_angles_list[math.floor(sampled_state_number)]
        pure_phase_angles_end = pure_phase_angles_list[math.ceil(sampled_state_number)]
        pure_phase_angles = np.zeros((pure_phase_angles_start.shape[0], base_state_count))
        for i in range(pure_phase_angles.shape[0]):
            pure_phase_angles[i, :] = lerp(
                pure_phase_angles_start[i, :], pure_phase_angles_end[i, :], sampled_state_number - math.floor(sampled_state_number)
            )
        measured_probabilities_start = measured_probabilities_list[math.floor(sampled_state_number)]
        measured_probabilities_end = measured_probabilities_list[math.ceil(sampled_state_number)]
        measured_probabilities = lerp(
            measured_probabilities_start[:], measured_probabilities_end[:], sampled_state_number - math.floor(sampled_state_number)
        )

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

    # A dictionary of plot names where the key is the plot name and the value is a matplotlib Axes object with a set position and size within the figure
    plot_mosaic_dict = None
    # specify the plot mosaic layout
    if show_measured_probabilities_only:
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
        elif pure_state_index is not None:
            if pure_state_index < pure_probabilities.shape[0]:
                y_values = pure_probabilities[pure_state_index, :]
            else:
                y_values = np.zeros(base_state_count)
        else:
            print("ERROR: plot_name not recognised: " + plot_name)
            exit(1)

        bar_list = plot_mosaic_dict[plot_name].bar(x_values, y_values, width=0.5)

        # set colour of the bars based on the phase
        if pure_state_index is not None:
            if pure_state_index < pure_probabilities.shape[0]:
                for i, phase_angle in enumerate(pure_phase_angles[pure_state_index, :]):
                    # Map phase angles from (-pi/2, pi/2] to (0, 1]
                    colour_value = (phase_angle + np.pi) / (2 * np.pi)
                    bar_list[i].set_color(cmap_phase(colour_value))

        plot_mosaic_dict[plot_name].set_xlim((-0.5, base_state_count - 1 + 0.5))
        plot_mosaic_dict[plot_name].set_ylim([0, np.max(y_values)])
        plot_mosaic_dict[plot_name].tick_params(axis="x", colors=tick_colour)
        plot_mosaic_dict[plot_name].tick_params(axis="y", colors=tick_colour)
        plot_mosaic_dict[plot_name].tick_params(axis="y", labelsize=20)
        plot_mosaic_dict[plot_name].axes.xaxis.set_visible(False)
        plot_mosaic_dict[plot_name].axes.yaxis.set_visible(False)

        if (zero_noise and pure_state_index == 0) or plot_name == "meas_probs":
            plot_mosaic_dict[plot_name].axes.yaxis.set_visible(True)

        if pure_state_index == 1 or pure_state_index == 3:
            plot_mosaic_dict[plot_name].axes.yaxis.set_visible(True)

        # Set the x ticks and tick labels for important subplots
        if pure_state_index == 0 or plot_name == "meas_probs":
            # if there are too many qubits to fit all of the x ticks, then only draw a limited number
            if (zero_noise and num_qubits > 4) or ((not zero_noise) and num_qubits > 3) or (plot_name == "meas_probs" and num_qubits > 4):
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
            plot_mosaic_dict[plot_name].tick_params(axis="x", labelsize=14)

    # Add axis labels
    fig.text(
        0.01,
        (fig_gridspec["top"] - fig_gridspec["bottom"]) / 2 + fig_gridspec["bottom"],
        "Probability",
        va="center",
        rotation="vertical",
        fontsize=20,
    )
    fig.text((fig_gridspec["right"] - fig_gridspec["left"]) / 2 + fig_gridspec["left"], 0.035, "Quantum states", ha="center", fontsize=20)
    return fig


def _plot_info_panel(
    output_manager: data_manager.DataManager,
    fig: Optional[matplotlib.figure.Figure],
    pure_probabilities: np.ndarray,
    pure_phase_angles: np.ndarray,
    fidelity: float,
    sampled_state_number: int,
    qubit_count: int,
    cmap_phase: matplotlib.colors.Colormap,
    cmap_fidelity: matplotlib.colors.Colormap,
    tick_colour: Colour,
    fidelity_line_colour: Colour,
    invert_colours: bool,
    vpr: Optional[Union[float, Callable[[int], float]]],
    draw_phase_wheel: bool,
) -> matplotlib.figure.Figure:
    """Plot the information panel corresponding to the given sampled state number and save it as a .png.

    Parameters
    ----------
        output_manager
            Instance of DataManager for saving the figure.
        fig
            Matplotlib Figure instance to plot on. If None, a new figure is created.
        pure_probabilities
            A 2d array for the sampled state, where the array contains probabilities indexed by pure state number and basis state number.
        pure_phase_angles
            A 2d array for the sampled state, where the array contains phse angles indexed by pure state number and basis state number.
        fidelity
            The fidelity of the sampled state relative to the noiseless state.
        sampled_state_number
            The index number of the sampled state to be used when saving image to file.
        qubit_count
            The number of qubits in the quantum state.
        cmap_phase
            A colourmap for the phase angle colours in the phase wheel. The colourmap should map normalised phase angles [0,1] to colours.
        cmap_fidelity
            A colourmap for the fidelity bar. The colourmap should map normalised phase angles [0,1] to colours.
        tick_colour
            The colour of the tick marks.
        fidelity_line_colour
            The colour of the line marker on the fidelity bar.
        invert_colours
            Whether the colours are being inverted. Used to ensure that the colours are correct when the colours are inverted.
        vpr
            Propotion of vertical space that the circuit will occupy. Can be a float or a function that maps qubit_count (int) to float.
        draw_phase_wheel
            If True, only plot the phase wheel.

    Returns
    -------
        The figure containing the plot of the quantum state.
    """
    if vpr is None:

        def vpr(n):
            return 1.0 / 3.0

    elif isinstance(vpr, float):
        vpr_temp = vpr

        def vpr(n):
            return vpr_temp

    elif not callable(vpr):
        raise TypeError("vpr must be a float, None or a Callable[[int], float].")

    probabilities_0 = list(pure_probabilities[0, :])
    phase_angles_0 = list(pure_phase_angles[0, :])
    tick_colour = convert_colour_to_float(tick_colour)
    fidelity_line_colour = convert_colour_to_float(fidelity_line_colour)

    target_video_height_pixels = 1080
    dpi = plt.rcParams["figure.dpi"]
    target_empty_circuit_height_pixels = target_video_height_pixels * (1 - vpr(qubit_count))
    fig_height_inches = (1 - vpr(qubit_count)) * 13.5
    fig_height_pixels = fig_height_inches * dpi
    scale = target_empty_circuit_height_pixels / fig_height_pixels

    scaled_dpi = dpi * scale

    if fig is None:
        fig = plt.figure(figsize=(4, (1 - vpr(qubit_count)) * 13.5), dpi=scaled_dpi)
        fig.set_dpi(scaled_dpi)

    if draw_phase_wheel is True:
        _plot_phase_wheel(fig, probabilities_0, phase_angles_0, cmap_phase, tick_colour, invert_colours)
    _plot_stat_bars(fig, fidelity, cmap_fidelity, fidelity_line_colour, tick_colour, invert_colours)

    plt.savefig(output_manager.get_path(f"info_panel_{sampled_state_number}.png"))
    plt.close("all")
    return fig
