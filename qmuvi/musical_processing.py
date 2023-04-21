# Methods relating to the generation of music from sampled density matrices
# Path: qmuvi\musical_processing.py

import glob
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
from typing import Any, AnyStr, Callable, Dict, List, Mapping, Optional, Tuple, Union
from shutil import which

import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

import qmuvi
import qmuvi.data_manager as data_manager
import qmuvi.quantum_simulation as quantum_simulation


def note_map_chromatic_middle_c(n: int) -> int:
    """Map `n` to a note in the chromatic scale. It starts at middle C.

    Parameters
    ----------
        n
            The integer to be mapped to a note.

    Returns
    -------
        The note mapped to from `n`. Returns `n` + 60.
    """
    return n + 60


def note_map_c_major(n: int) -> int:
    """Map `n` to a note in the C major scale. It starts at middle C.

    Parameters
    ----------
        n
            The integer to be mapped to a note.

    Returns
    -------
        The note mapped to from `n`.
    """
    C_MAJ = [
        0,
        2,
        4,
        5,
        7,
        9,
        11,
        12,
        14,
        16,
        17,
        19,
        21,
        23,
        24,
        26,
        28,
        29,
        31,
        33,
        35,
        36,
        38,
        40,
        41,
        43,
        45,
        47,
        48,
        50,
        52,
        53,
        55,
        57,
        59,
        60,
        62,
        64,
        65,
        67,
        69,
        71,
        72,
        74,
        76,
        77,
        79,
        81,
        83,
        84,
        86,
        88,
        89,
        91,
        93,
        95,
        96,
        98,
        100,
        101,
        103,
        105,
        107,
        108,
        110,
        112,
        113,
        115,
        117,
        119,
        120,
        122,
        124,
        125,
        127,
    ]
    CURR_MODE = C_MAJ
    note = 60 + n
    # Shifting
    if CURR_MODE and note < CURR_MODE[0]:
        note = CURR_MODE[0]
    else:
        while CURR_MODE and note not in CURR_MODE:
            note -= 1
    return note


def note_map_f_minor(n: int) -> int:
    """Map `n` to a note in the F minor scale. It starts at middle C.

    Parameters
    ----------
        n
            The integer to be mapped to a note.

    Returns
    -------
        The note mapped to from `n`.
    """
    F_MIN = [
        5,
        7,
        8,
        10,
        12,
        13,
        15,
        17,
        19,
        20,
        22,
        24,
        25,
        27,
        29,
        31,
        32,
        34,
        36,
        37,
        39,
        41,
        43,
        44,
        46,
        48,
        49,
        51,
        53,
        55,
        56,
        58,
        60,
        61,
        63,
        65,
        67,
        68,
        70,
        72,
        74,
        75,
        77,
        79,
        80,
        82,
        84,
        86,
        87,
        89,
        91,
        92,
        94,
        96,
        97,
        99,
        101,
        103,
        104,
        106,
        108,
        109,
        111,
        113,
        115,
        116,
        118,
        120,
        121,
        123,
        125,
        127,
    ]
    CURR_MODE = F_MIN
    note = 60 + n
    # Shifting
    if CURR_MODE and note < CURR_MODE[0]:
        note = CURR_MODE[0]
    else:
        while CURR_MODE and note not in CURR_MODE:
            note -= 1
    return note


def note_map_c_major_arpeggio(n: int) -> int:
    """Map `n` to a note in the C major arpeggio. It starts at middle C.

    Parameters
    ----------
        n
            The integer to be mapped to a note.

    Returns
    -------
        The note mapped to from `n`.
    """
    if (n + 1) % 3 == 0:
        return (4 * n - 1) + 60
    return (4 * n) + 60


def get_sound_data_from_density_matrix(
    density_matrix: np.ndarray, default_pure_state_global_phasors: Mapping[int, complex], eps: float = 1e-8
) -> List[Tuple[float, Dict[int, Tuple[float, float]], List[float], List[float]]]:
    """Extract a list of sound data from a density matrix by eigendecomposing it into a mixture of pure states.

    Parameters
    ----------
        density_matrix
            A density matrix in the form of a numpy array.
        default_pure_state_global_phasors
            A dictionary of default global phasors for each pure state in the density matrix.
            Used to keep the global phase consistent during eigendecompositions of different density matrices.
        eps
            A small probability threshold. Used to filter out pure states and some basis states with negligible probability.

    Returns
    -------
        A list of sound data for pure states (under the condition pure_state_prob > eps) in the form of tuples, each tuple containing:
        (
        pure_state_prob,
        pure_state_info: A Dict with key: basis_state_number, and value: (basis_state_prob, basis_state_angle), # where (basis_state_prob > eps)
        all_basis_state_probs,
        all_basis_state_angles
        ).
    """
    sound_data = []
    # eigenvalues are not necessarily ordered
    mixture_eigenvalues, mixture_statevectors = np.linalg.eig(density_matrix)
    nonzero_probability_indices = []
    for i, eigenvalue in enumerate(mixture_eigenvalues):
        if eigenvalue.real > eps:
            nonzero_probability_indices.append(i)

    # each pure state in the mixture has a global phase introduced during the eigendecomposition
    for pure_state_index in nonzero_probability_indices:
        pure_state_probability = mixture_eigenvalues[pure_state_index].real
        notes_by_basis_state_index = {}
        pure_statevector = mixture_statevectors[:, pure_state_index]
        zero_state_amplitude = complex(pure_statevector[0])
        zero_state_probability = (zero_state_amplitude * np.conj(zero_state_amplitude)).real
        if zero_state_probability > eps:
            pure_global_phasor = np.conj(zero_state_amplitude)
            default_pure_state_global_phasors[pure_state_index] = pure_global_phasor
        else:
            pure_global_phasor = default_pure_state_global_phasors[pure_state_index]

        # get the angles for each of the amplitudes (-pi, pi]
        basis_state_phase_angles = list(np.angle(pure_statevector * pure_global_phasor))
        basis_state_probabilities = []
        # get the (probability, angle) pair for the basis states in the statevector
        for basis_state_index, basis_state_amplitude in enumerate(pure_statevector):
            basis_state_probability = (complex(basis_state_amplitude) * np.conj(complex(basis_state_amplitude))).real
            basis_state_probabilities.append(basis_state_probability)
            if basis_state_probability > eps:
                angle = basis_state_phase_angles[basis_state_index]
                notes_by_basis_state_index[basis_state_index] = (basis_state_probability, angle)

        pure_state_sound_data = (pure_state_probability, notes_by_basis_state_index, basis_state_probabilities, basis_state_phase_angles)
        sound_data.append(pure_state_sound_data)
    return sound_data


def convert_midi_to_wav_timidity(output_manager: data_manager.DataManager, log_to_file: bool = False) -> None:
    """Convert a MIDI file to a WAV file using the Timidity++ library.

    This function takes the path to generated MIDI files and uses the Timidity++ library to
    convert the files to a single WAV file.

    Parameters
    ----------
        output_manager
            A data manager object for finding the MIDI files and handling the output file.
        log_to_file
            Whether to log the output of the Timidity process to a file.

    Returns
    -------
        This function does not return anything, but writes the output WAV file to disk.
    """
    files = output_manager.glob(output_manager.default_name + "-*.mid")
    if len(files) == 0:
        files = output_manager.glob(output_manager.default_name + ".mid")

    # documentation found here: https://www.mankier.com/1/timidity#Input_File
    options = []
    options.append("-Ow")
    if log_to_file is True:
        options.append("--verbose=3")
    options.append("--preserve-silence")
    options.append("-A,120")
    # anti-aliasing seems to cause some crackling
    options.append("--no-anti-alias")
    options.append("--mod-wheel")
    options.append("--portamento")
    options.append("--vibrato")
    options.append("--no-ch-pressure")
    options.append("--mod-envelope")
    options.append("--trace-text-meta")
    options.append("--overlap-voice")
    # options.append("--temper-control")
    options.append("--default-bank=0")
    options.append("--default-program=0")
    # d: disabled, l: left, r: right, b: swap l&r
    options.append("--delay=d,0")
    # chorus options
    # d: disabled,
    # n: enable MIDI chorus effect control,
    # s: surround sound, chorus detuned to a lesser degree.
    # last number is chorus level 0-127
    ###
    options.append("--chorus=n,64")
    # reverb options
    # d: disabled,
    # n: enable MIDI reverb effect control,
    # g: global reverb effect,
    # f: Freeverb MIDI reverb effect control,
    # G: global, Freeverb effect.
    # num 1: reverb level 0-127
    # num 2: reverb scaleroom [0,1], roomsize = C * scaleroom + offsetroom, where C is reverberation character
    # num 3: reverb offsetroom [0,1]
    # num 4: reverb factor for pre-delay time of reverberation in percent
    ###
    options.append("--reverb=f,40,0.28,0.7,100")
    # d: disable, c: Chamberlin resonant LPF (12dB/oct), m: Moog resonant low-pass VCF (24dB/oct)
    options.append("--voice-lpf=c")
    # 0: no shaping, 1: trad, 2: Overdrive-like soft-clipping + new noise shaping, 3: Tube-amplifier-like soft-clipping + new noise shaping, 4: New noise shaping
    options.append("--noise-shaping=4")
    options.append("--resample=5")  # 0-5, 5 is highest quality
    options.append("--voice-queue=0")
    # options.append("--fast-decay") # 0 means no voices will be killed even when there's a delay due to so many in the queue
    options.append("--decay-time=0")
    # options.append("-R 100")
    options.append("--interpolation=gauss")
    options.append("-EFresamp=34")  # for interpolation gauss: 0-34
    options.append("--output-stereo")
    options.append("--output-24bit")
    options.append("--polyphony=15887")
    options.append("--sampling-freq=44100")
    #options.append("--audio-buffer=5/100") # causes error in timidity version 2.15 (linux)
    # (regular: 0, linear: 1, ideal: ~1.661, GS: ~2)
    options.append("--volume-curve=1.661")
    # options.append("--module=4")

    package_path = os.path.dirname(os.path.realpath(__file__))
    config_filepath = os.path.join(package_path, "package_data", "resources", "timidity", "timidity.cfg")
    options.append(f'--config-file="{config_filepath}"')

    options_string = " ".join(options)

    def timidity_convert_subprocess(
        filename: str,
        options_string: str,
        thread_results: Mapping[int, subprocess.CompletedProcess],
        thread_errors: Mapping[int, Any],
        thread_index: int,
    ) -> None:
        try:
            if log_to_file is True:
                # setup logger for thread
                log = logging.getLogger(threading.current_thread().name)
                log.setLevel(logging.DEBUG)

                # add file handler
                log_filepath = os.path.join(os.getcwd(), f"log-timidity-{thread_index}.log")
                # os.remove(log_filepath)
                file_handler = logging.FileHandler(log_filepath, mode="w")
                formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
                file_handler.setFormatter(formatter)
                log.addHandler(file_handler)

            # Get the absolute path of the package
            package_path = os.path.dirname(os.path.realpath(__file__))

            shell = False

            if log_to_file is True:
                log.info(f"{platform.system()}")
                log.info(f"cwd: {os.getcwd()}")
                log.info(f"package_path: {package_path}")
                log.debug(f'midi filename: "{filename}.mid"')
                log.debug(f'wav filename: "{filename}.wav"')

            if platform.system() == "Windows":
                # Join the path to the binary file
                binary_path = os.path.join(package_path, "package_data", "binaries", "TiMidity-2.15.0", "windows", "timidity.exe")
                command = " ".join([binary_path, options_string, f'-o "{filename}.wav"', f'"{filename}.mid"'])
            elif platform.system() == "Darwin":
                binary_path = os.path.join(package_path, "package_data", "binaries", "TiMidity-2.15.0", "macos", "timidity")
                command = [binary_path] + options + ["-o", f'"{filename}.wav"', f'"{filename}.mid"']
            else:
                # Assume the binary is in the PATH on other platforms (installable on Linux)
                binary_path = "timidity"
                shell = True
                if which(binary_path) is None:
                    if log_to_file is True:
                        log.exception(f"Error: Could not find timidity binary in PATH: {binary_path}. Please install timidity.")
                    raise Exception(f"Could not find timidity binary in PATH: {binary_path}. Please install timidity.")

                command = " ".join([binary_path, options_string, f'-o "{filename}.wav"', f'"{filename}.mid"'])

            if log_to_file is True:
                log.info(f"timidity binary path: {binary_path}")
                log.info(f"timidity options: {options_string}")
                log.debug(f"executing subprocess command: {command}")

            thread_results[thread_index] = subprocess.run(command, cwd=package_path, capture_output=True, check=True, shell=shell)

            if log_to_file is True:
                log.debug("completed subprocess")

        except BaseException as e:
            thread_errors[thread_index] = (e, sys.exc_info())
            if log_to_file is True:
                log.exception("An error occurred:")
        if log_to_file is True:
            log.info(thread_results[thread_index].stdout.decode())
            log.info(thread_results[thread_index].stderr.decode())

    convert_files_mid_to_wav_timidity_threading(files, options_string, timidity_convert_subprocess, timeout=8)

    files = output_manager.glob(output_manager.default_name + "-*.wav")
    if len(files) > 1:
        print("Mixing .wav track files to a single file...")
        mix_wav_files(files, output_manager)


def mix_wav_files(file_paths: List[str], output_manager: data_manager.DataManager) -> None:
    """Mixe multiple WAV files into a single file.

    This function takes a list of file paths for WAV files and a data manager object for handling the output file, and
    mixes the audio content of the input files into a single file. The resulting file is saved to the output file path
    returned by the data manager.

    Parameters
    ----------
    file_paths
        A list of file paths for the WAV files to be mixed.
    output_manager
        A data manager object for handling the output file.

    Returns
    -------
        This function does not return a value. The resulting mixed audio file is saved to disk.
    """
    import moviepy.editor as mpy
    from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip

    audio_clips = []
    audio_file_clips = []
    for path in file_paths:
        path_multiplatform = path.replace("\\", "/")
        audio_file_clip = mpy.AudioFileClip(path_multiplatform, nbytes=4, fps=44100)
        audio_file_clips.append(audio_file_clip)
        audio_array = audio_file_clip.to_soundarray(nbytes=4)
        total_time = audio_file_clip.duration
        audio_array_clip = AudioArrayClip(audio_array[0 : int(44100 * total_time)], fps=44100)
        audio_clips.append(audio_array_clip)

    composed_audio_clip = CompositeAudioClip(audio_file_clips)
    output_filename = output_manager.get_default_file_pathname() + ".wav"
    composed_audio_clip.write_audiofile(output_filename, codec="pcm_s16le", fps=44100)


def convert_files_mid_to_wav_timidity_threading(
    pathnames: List[str],
    options_string: str,
    timidity_convert_method: Callable[[str, str, Dict[int, subprocess.CompletedProcess], Dict[int, Any], int], None],
    timeout: float = 8,
) -> List[str]:
    """Convert a list of MIDI files to WAV format using TiMidity++ in a threaded manner.

    Given a list of file paths to MIDI files, an options string for TiMidity++, a function to perform the conversion, and a timeout,
    this function will convert each file to WAV format using TiMidity++ in a separate thread. Each thread runs the conversion function
    with the given options string and the input file path. If a timeout is specified and the conversion function takes longer than the
    timeout to complete, the function will raise an exception.

    Parameters
    ----------
        pathnames
            A list of file paths to MIDI files that will be converted to WAV format.
        options_string
            An options string to be passed to the TiMidity++ conversion function.
        timidity_convert_method
            The function to be used to perform the TiMidity++ conversion.
        timeout
            The number of seconds to wait for each thread to complete before raising a timeout exception. Default is 8.

    Returns
    -------
        A list of file paths to the converted WAV files.

    Raises
    ------
    Exception
        If a timeout occurs while converting a MIDI file to WAV format.
    """
    output_wav_files = []

    base_pathnames = [os.path.splitext(pathname.replace("\\", "/"))[0] for pathname in pathnames]

    # instantiate lists that store data from each thread
    thread_results = [None] * len(pathnames)
    thread_errors = [None] * len(pathnames)

    # start a thread for each file to be converted
    threads = []
    for file_index in range(len(pathnames)):
        t = threading.Thread(
            target=timidity_convert_method,
            name=f"timidity-{file_index}",
            args=((base_pathnames[file_index], options_string, thread_results, thread_errors, file_index)),
        )
        t.daemon = True
        threads.append(t)
        t.start()

    print("Converting to .wav files using TiMidity++")

    # Wait for all threads to finish, with a timeout
    thread_has_timed_out = False
    for thread_index, t in enumerate(threads):
        t.join(timeout=timeout)
        wav_file = f"'{base_pathnames[file_index]}" + ".wav'"
        if t.is_alive() and not os.path.isfile(wav_file):
            # thread is still running and .wav file was not created, a conversion timeout has occured
            t._stop()
            print(f"Timidity++ timed out for {wav_file}, skipping conversion")
            thread_has_timed_out = True
        if os.path.isfile(wav_file):
            output_wav_files.append(wav_file)

    # wait a small amount of time to ensure that the .wav file conversion has been finalised
    time.sleep(0.1)

    # kill the remaining threads that have successfully created their .wav file
    for t in threads:
        if t.is_alive():
            t._stop()

    # Output any errors that may have occured within the threads
    for thread_index, thread_result in enumerate(thread_results):
        filename = pathnames[thread_index]
        if thread_errors[thread_index] is not None:
            # an error occured in the thread, show the error
            print(f"Errors in thread {thread_index}:")
            raise thread_errors[thread_index][0]

        if thread_result.returncode != 0:
            # timidity command failed in subprocess, show subprocess stdout and stderr
            print(f"Error from Timidity++ converting file {filename}")
            print("Output:")
            print(f'"{thread_result.stdout.decode().strip()}"')
            print("stderr:")
            print(f'"{thread_result.stderr.decode().strip()}"')
            print("")

    if len(pathnames) == 1 and thread_has_timed_out is True:
        raise Exception(f"ERROR: Thread timed out ({timeout}s) during timidity mid to wav conversion on the single file {pathnames[0]}")

    return output_wav_files


def generate_midi_from_data(
    output_manager: data_manager.DataManager,
    instruments: List[List[int]] = [list(range(81, 89))],
    note_map: Callable[[int], int] = note_map_c_major_arpeggio,
    rhythm: Optional[List[Tuple[int, int]]] = None,
) -> List[str]:
    """Use the qmuvi generated data files to create a song as a midi file (.mid).

    Parameters
    ----------
        output_manager
            The DataManager object for the output folder.
        instruments
            The collections of instruments for each pure state in the mixed state (up to 8 collections).
            Computational basis state phase determines which instrument from the collection is used.
        note_map
            Converts state number to a note number where 60 is middle C.
        rhythm
            A list of tuples for the length and rest times of each sound in units of ticks (480 ticks is 1 second).
            If None, then each sound length and rest time will be set to (note_sound, rest_time) = (240, 0).

    Returns
    -------
        A list of file paths to the generated MIDI files.
    """
    print("Generating MIDI track files...")

    sounds_list = output_manager.load_json(filename="sounds_list.json")

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

    for rhythm_index in range(len(rhythm)):
        rhythm[rhythm_index] = (int(rhythm[rhythm_index][0]), int(rhythm[rhythm_index][1]))

    track_count = 8
    mid_files = []
    tracks = []
    for i in range(track_count):
        mid_files.append(MidiFile())
        tracks.append(MidiTrack())

    # if less than the the track_count number of instruments are provided, then the last instrument will be used for the remaining tracks
    instruments_by_track = []
    for instrument_index in range(track_count):
        if instrument_index < len(instruments):
            instruments_by_track.append(instruments[instrument_index])
        else:
            instruments_by_track.append(instruments[-1])

    for track in tracks:
        track.append(MetaMessage("set_tempo", tempo=bpm2tempo(60)))

    # not sure where these numbers come from, might be from the GeneralMidi specification.
    available_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]
    largest_track_instruments_count = max([len(instrument_list) for instrument_list in instruments_by_track])
    channels = [available_channels[instrument_index % len(available_channels)] for instrument_index in range(largest_track_instruments_count)]
    for rhythm_index, sound in enumerate(sounds_list):
        active_notes = []
        # TODO: check if this is redundant (already done above)
        sorted_pure_state_infos = sorted(sound, key=lambda a: a[0], reverse=True)
        # the largest probability of the pure states in the sampled state
        largest_chord_prob = sorted_pure_state_infos[0][0]

        # start playing the notes with the correct velocity and instrument for each track
        for track_number, track in enumerate(tracks):
            channel = 0

            # if there are no more pure states to add to the track, then add a rest
            if track_number >= len(sorted_pure_state_infos):
                track.append(Message("note_on", channel=channel, note=0, velocity=0, time=0))
                active_notes.append((track_number, 0, channel))
                continue

            # chord is Dict[int basis_state_number, Tuple[float basis_state_prob, float basis_state_angle]]
            chord_prob, pure_state_info, _, _ = sorted_pure_state_infos[track_number]
            largest_note_prob = max([pure_state_info[basis_state_number][0] for basis_state_number in pure_state_info.keys()])

            for basis_state_number in pure_state_info.keys():
                note_number = note_map(basis_state_number)
                note_prob, note_angle = pure_state_info[basis_state_number]
                note_velocity = (note_prob / largest_note_prob) * (chord_prob / largest_chord_prob)

                # converts the velocity from a float [0, 1] to an int [0, 127]
                velocity_128 = int(128 * (note_velocity))
                # handle the special case where note_velocity >= 1
                if velocity_128 > 127:
                    velocity_128 = 127

                track_instruments = instruments_by_track[track_number]

                # bound the angle between 0 and 2pi taking into consideration the periodic nature of the phase
                note_angle_bounded = (note_angle + np.pi / len(track_instruments)) % (2 * np.pi)
                # convert the angle to a fraction of 2pi, resulting in a value in the range [0, 1)
                note_angle_fraction = note_angle_bounded / (2 * np.pi)
                instrument_index = int(note_angle_fraction * len(track_instruments))

                instrument = track_instruments[instrument_index]
                channel = channels[instrument_index]
                # Set the instrument for the track
                track.append(Message("program_change", channel=channel, program=instrument, time=0))
                # start playing the note_number with the given velocity, the note stops playing when the 'note_off' message is sent
                track.append(Message("note_on", channel=channel, note=note_number, velocity=velocity_128, time=0))
                active_notes.append((track_number, note_number, channel))

        # move the tracks forward in time according to the rhythm
        # TODO: Check if this assumes that sound notes cannot be 0, if so, then modify the approach to allow for 0 notes
        for track_number, track in enumerate(tracks):
            channel = 0
            track.append(Message("note_on", channel=channel, note=0, velocity=0, time=rhythm[rhythm_index][0]))
            track.append(Message("note_off", channel=channel, note=0, velocity=0, time=0))

        # turn off all of the notes for this sound
        for track_number, note_number, channel in active_notes:
            track = tracks[track_number]
            track.append(Message("note_off", channel=channel, note=note_number, velocity=0, time=0))

        # Add a rest, specified by the rhythm, to each track
        for track_number, track in enumerate(tracks):
            channel = 0
            track.append(Message("note_on", channel=channel, note=0, velocity=0, time=rhythm[rhythm_index][1]))
            track.append(Message("note_off", channel=channel, note=0, velocity=0, time=0))

    output_manager.remove_files(output_manager.default_name + "-*.mid")

    for track_number, track in enumerate(tracks):
        mid_files[track_number].tracks.append(track)
        mid_files[track_number].save(output_manager.get_path(f"{output_manager.default_name}-{track_number}.mid"))

    return mid_files
