import glob
import math
import os
import platform
import sys
import warnings
from typing import Any, AnyStr, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
import qiskit.quantum_info as qi
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error

from . import data_manager, musical_processing, quantum_simulation, video_generation
from .musical_processing import note_map_c_major_arpeggio

__version__ = "0.1.0"


def get_instrument_collection(collection_name: str) -> List[int]:
    """Returns a list of instruments as ints defined by the General MIDI standard.

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

    Parameters
    ----------
        collection_name
            The name of the instrument collection.

    Returns
    -------
        A list of instruments from the given instrument collection name.
    """
    instrument_dict = {
        "piano": list(range(1, 9)),
        "tuned_perc": list(range(9, 17)),
        "organ": list(range(17, 25)),
        "guitar": list(range(25, 33)),
        "bass": list(range(33, 41)),
        "strings": list(range(41, 49)),
        "ensemble": list(range(49, 57)),
        "brass": list(range(57, 65)),
        "reed": list(range(65, 73)),
        "pipe": list(range(73, 81)),
        "synth_lead": list(range(81, 89)),
        "synth_pad": list(range(89, 97)),
        "synth_effects": list(range(97, 105)),
        "ethnic": list(range(105, 113)),
        "percussive": list(range(113, 121)),
        "sound_effects": list(range(120, 128)),
        "windband": [74, 69, 72, 67, 57, 58, 71, 59],
    }
    return instrument_dict[collection_name]


# TODO: make rhythm units of seconds with 480 units of precision rather than units of ticks.
def generate_qmuvi(
    quantum_circuit: QuantumCircuit,
    qmuvi_name: str,
    noise_model: Optional[NoiseModel] = None,
    rhythm: Optional[List[Tuple[int, int]]] = None,
    instruments: Union[List[List[int]], List[int]] = [list(range(81, 89))],
    note_map: Callable[[int], int] = note_map_c_major_arpeggio,
    invert_colours: bool = False,
    fps: int = 24,
    vpr: Optional[Union[float, Callable[[int], float]]] = 1.0 / 3.0,
    smooth_transitions: bool = True,
    log_to_file: bool = False,
    show_measured_probabilities_only: bool = False,
) -> None:
    """Samples the quantum circuit at every barrier and uses the state properties to create a music video (.mp4).

    Parameters
    ----------
        quantum_circuit
            The qiskit QuantumCircuit.
        name
            The name of the qMuVi.
        noise_model
            To sample quantum states, qMuVi uses the qiskit AerSimulator to simulate the circuits. A NoiseModel can be
            passed to the simulator to include noise in the computations, which is translated to the generated output in
            qMuVi. If None, then no noise will be used in the simulations.
        rhythm
            A is a list of tuples in the form (sound_time, rest_time), one for each qmuvi sampled state in the circuit.
            The sound_time is how long the sound will play for and the rest_time is the wait time afterwards
            before playing the next sound. Times are in units of ticks  where 480 ticks is 1 second.
            If None then each sound will be assigned (240, 0).
        instruments
            The collections of instruments for each pure state in the mixed state (up to 8 collections) (defaults to 'synth_lead').
            Computational basis state phase determines which instrument from the collection is used.
        note_map
            A callable object that maps state numbers to note numbers. The note map is used to convert the basis state numbers to
            the note number in the MIDI. A note number of 60 is middle C.
        invert_colours
            Whether to render the video in dark mode.
        fps
            The frames per second of the output video.
        vpr
            Propotion of vertical space that the circuit will occupy. Can be a float or a function that maps qubit_count (int) to float.
        smooth_transitions
            Whether to smoothly animate between histogram frames. Significantly increased render time.
        log_to_file
            Whether to output the timidity synth midi conversion log files.
        show_measured_probabilities_only
            Whether to only show the total basis state measurement probability distribution in the video.

    Returns
    -------
        This function does not return a value. The resulting output files are saved in a folder in the current working directory.
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

    if isinstance(instruments, list) and isinstance(instruments[0], list):
        if len(instruments) > 8:
            raise ValueError("The maximum number of instrument collections is 8.")
    elif isinstance(instruments, list) and isinstance(instruments[0], int):
        instruments = [instruments]
    else:
        raise TypeError("instruments must be a list of lists of ints or a list of ints.")

    output_path = data_manager.get_unique_pathname(qmuvi_name + "-output", os.getcwd())
    output_manager = data_manager.DataManager(output_path, default_name=qmuvi_name)

    # generate data from quantum circuit
    sounds_list, fidelity_list, meas_prob_samples = generate_qmuvi_data(quantum_circuit, output_manager, noise_model=noise_model)

    if rhythm is None:
        rhythm = [(240, 0)] * len(sounds_list)

    # generate midi files from data
    musical_processing.generate_midi_from_data(output_manager, instruments=instruments, note_map=note_map, rhythm=rhythm)
    # convert midi files to a wav file
    musical_processing.convert_midi_to_wav_timidity(output_manager, log_to_file=log_to_file)
    # generate video from data
    video_generation.generate_video_from_data(
        quantum_circuit,
        output_manager=output_manager,
        rhythm=rhythm,
        invert_colours=invert_colours,
        fps=fps,
        vpr=vpr,
        smooth_transitions=smooth_transitions,
        show_measured_probabilities_only=show_measured_probabilities_only,
    )


def generate_qmuvi_data(
    quantum_circuit: QuantumCircuit, output_manager: data_manager.DataManager, noise_model: NoiseModel = None
) -> Tuple[List[List[Tuple[float, Mapping[int, Tuple[float, float]], List[float], List[float]]]], List[float], List[List[float]]]:
    """Samples the quantum circuit at every barrier and generates data from the state properties.

    Parameters
    ----------
        quantum_circuit
            The qiskit QuantumCircuit.
        output_manager
            The DataManager object for the output folder.
        noise_model
            A qiskit NoiseModel. If None then no noise will be used in the simulations.

    Returns
    -------
        sounds_list
            a list of lists of sound data where the first index is the sample_number and the second is the
            pure state index in the eigendecomposition in descending order with respect to the pure_state_prob (eigenvalue).
            Only pure states with pure_state_prob > eps are kept. The sound data is a tuple of form:
            (
            pure_state_prob: float,
            pure_state_info: Dict[int basis_state_number, Tuple[float basis_state_prob, float basis_state_angle]], # where (basis_state_prob > eps)
            all_basis_state_probs: List[float basis_state_prob],
            all_basis_state_angles: List[float basis_state_angle]
            ).
        fidelity_list
            A list of the fidelity of each of the sampled quantum states.
        meas_prob_samples
            A list of lists of the measurement probabilities where the first index is the sample_number and the second is the basis_state number.
    """
    print("Generating qMuVi data...")

    if noise_model is None:
        noise_model = NoiseModel()

    # TODO: When no noise: switch to statevector simulation to reduce resource requirements.
    density_matrices_pure = quantum_simulation.sample_circuit_barriers(quantum_circuit)
    # TODO: When no noise: remove redundant quantum simulation.
    density_matrices_noisy = quantum_simulation.sample_circuit_barriers(quantum_circuit, noise_model)

    # calculate the fidelity of each of the sampled quantum states
    fidelity_list = [qi.state_fidelity(dm_noisy, density_matrices_pure[i]) for i, dm_noisy in enumerate(density_matrices_noisy)]

    # used to keep the phase of the pure states consistent among each of the sampled states
    global_phasors = np.ones(density_matrices_noisy[0].shape[0], dtype=complex)

    # generate the sound data for each sampled state
    # The music is composed of a list of sounds, one for each sample of the quantum circuit
    # Each sound is a list of tuples, one for each pure state in the sampled state
    # Each tuple is of the form:
    # (
    #  pure_state_prob: float,
    #  pure_state_info: Dict[int basis_state_number, Tuple[float basis_state_prob, float basis_state_angle]], # where (basis_state_prob > eps)
    #  all_basis_state_probs: List[float basis_state_prob],
    #  all_basis_state_angles: List[float basis_state_angle]
    # )
    sounds_list = []
    for rho in density_matrices_noisy:
        sound_data = musical_processing.get_sound_data_from_density_matrix(rho, global_phasors)
        sounds_list.append(sound_data)

    # gather the probabilities for each sampled state
    meas_prob_samples = []
    for rho in density_matrices_noisy:
        # get the measurement probabilities from each basis state
        measurement_probabilities = []
        for i in range(rho.shape[0]):
            measurement_probabilities.append(rho[i, i].real)
        meas_prob_samples.append(measurement_probabilities)

    # sort the sounds by probability
    for i, sound in enumerate(sounds_list):
        # the first index of an element of sound is the probability of the pure state in the
        # linear combination making up the density matrix
        sounds_list[i] = sorted(sound, key=lambda a: a[0], reverse=True)

    output_manager.save_json(sounds_list, filename="sounds_list.json")
    output_manager.save_json(fidelity_list, filename="fidelity_list.json")
    output_manager.save_json(meas_prob_samples, filename="meas_probs_list.json")

    return sounds_list, fidelity_list, meas_prob_samples


def generate_qmuvi_music(
    quantum_circuit: QuantumCircuit,
    qmuvi_name: str,
    noise_model: NoiseModel = None,
    rhythm: Optional[List[Tuple[int, int]]] = None,
    instruments: List[List[int]] = [list(range(81, 89))],
    note_map: Callable[[int], int] = note_map_c_major_arpeggio,
    log_to_file: bool = False,
):
    """Samples the quantum circuit at every barrier and uses the state properties to create a song (.wav).

    Parameters
    ----------
        quantum_circuit
            The qiskit QuantumCircuit.
        qmuvi_name
            The name of the qMuVi.
        noise_model
            A qiskit NoiseModel. If None then no noise will be used in the simulations.
        rhythm
            A list of tuples for the sound and rest times of each played sound in units of ticks (480 ticks is 1 second).
            If None, then for each sample, (sound_time, rest_time) = (240, 0)
        instruments
            The collections of instruments for each pure state in the mixed state (up to 8 collections) (defaults to 'synth_lead').
            Computational basis state phase determines which instrument from the collection is used.
        note_map
            Converts state number to a note number where 60 is middle C.
        log_to_file
            Whether to output the timidity synth midi conversion log files.
    """
    output_path = data_manager.get_unique_pathname(qmuvi_name + "-output", os.getcwd())
    output_manager = data_manager.DataManager(output_path, default_name=qmuvi_name)

    # generate data from quantum circuit
    sounds_list, fidelity_list, meas_prob_samples = generate_qmuvi_data(quantum_circuit, output_manager, noise_model=noise_model)

    if rhythm is None:
        rhythm = [(240, 0)] * len(sounds_list)

    # generate midi files from data
    musical_processing.generate_midi_from_data(output_manager, instruments=instruments, note_map=note_map, rhythm=rhythm)
    # convert midi files to a wav file
    musical_processing.convert_midi_to_wav_timidity(output_manager, log_to_file=log_to_file)


def generate_qmuvi_video(
    quantum_circuit: QuantumCircuit,
    output_manager: data_manager.DataManager,
    rhythm: Optional[List[Tuple[int, int]]] = None,
    noise_model: NoiseModel = None,
    note_map: Callable[[int], int] = note_map_c_major_arpeggio,
    invert_colours: bool = False,
    fps: int = 60,
    vpr: Optional[Union[float, Callable[[int], float]]] = 1.0 / 3.0,
    smooth_transitions: bool = True,
    phase_marker: bool = True,
    show_measured_probabilities_only: bool = False,
):
    """Samples the quantum circuit at every barrier and uses the state properties to create a silent video (.mp4). No music is generated using this method.

    Parameters
    ----------
        quantum_circuit
            The qiskit QuantumCircuit.
        output_manager
            The data manager to use for saving the video and all of its pieces.
        rhythm
            A list of tuples for the length and rest times of each sound in units of ticks (480 ticks is 1 second).
            If None, then each sound length and rest time will be set to (note_sound, rest_time) = (240, 0).
        noise_model
             qiskit NoiseModel. If None then no noise will be used in the simulations.
        note_map
            Converts state number to a note number where 60 is middle C.
        invert_colours
            Whether to render the video in dark mode.
        fps
            The frames per second of the output video.
        vpr
            Propotion of vertical space that the circuit will occupy. Can be a float or a function that maps qubit_count (int) to float.
        smooth_transitions
            Whether to smoothly animate between histogram frames. Significantly increased render time.
        phase_marker
            Whether to draw lines on the phase wheel indicating phases of the primary pure state.
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

    # generate data from quantum circuit
    sounds_list, fidelity_list, meas_prob_samples = generate_qmuvi_data(quantum_circuit, output_manager, noise_model=noise_model)

    if rhythm is None:
        rhythm = [(240, 0)] * len(sounds_list)

    # generate video from data
    video_generation.generate_video_from_data(
        quantum_circuit,
        output_manager=output_manager,
        rhythm=rhythm,
        invert_colours=invert_colours,
        fps=fps,
        vpr=vpr,
        smooth_transitions=smooth_transitions,
        phase_marker=phase_marker,
        show_measured_probabilities_only=show_measured_probabilities_only,
    )
