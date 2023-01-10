# Methods relating to the generation of music from sampled density matrices
# Path: qmuvi\simulation.py

import qmuvi.quantum_simulation as quantum_simulation
import qmuvi.data_manager as data_manager

import os
import glob
import time
import numpy as np
import logging
import threading

def get_sound_data_from_density_matrix(density_matrix, default_pure_state_global_phasors, eps = 1E-8):
    ''' Extracts a list of sound data from a density matrix by eigendecomposing it into a mixture of pure states
    Args: 
        density_matrix: a density matrix in the form of a numpy array
        default_pure_state_global_phasors: a dictionary of default global phasors for each pure state in the density matrix. 
            Used to keep the global phase consistent during eigendecompositions of different density matrices
        eps: a small probability threshold. Used to filter out pure states and some basis states with negligible probability
    Returns: a list of sound data for (prob > eps) pure states in the form of tuples, each tuple containing:
        - the probability of the pure state
        - a dictionary of (probability, angle) pairs for each (prob > eps) basis state in the pure state, 
            keys are indices of basis states in the statevector
        - a list of all basis state probabilities in the pure state
        - a list of all basis state angles in the pure state
    '''

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
        
        # get the angles for each of the amplitudes
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

def convert_midi_to_wav_timidity(files, timeout = 8, log_to_file = False):
    """ Converts a list of midi files to wav files using the Timidity program
    Args:
        files: A list of midi files to convert
        timeout: The amount of time to wait for the Timidity threads to complete before terminating
        log_to_file: Whether to log the output of the Timidity process to a file
    """

    # documentation found here: https://www.mankier.com/1/timidity#Input_File
    options = []
    options.append("-Ow")
    if log_to_file == True:
        options.append("--verbose=3")
    options.append("--preserve-silence")
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
    ### chorus options
    # d: disabled, 
    # n: enable MIDI chorus effect control, 
    # s: surround sound, chorus detuned to a lesser degree. 
    # last number is chorus level 0-127
    ###
    options.append("--chorus=n,64")
    ### reverb options
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
    options.append("--polyphony=15887")
    options.append("--sampling-freq=44100")
    options.append("--audio-buffer=5/100")
    options.append("--volume-curve=1.661") # (regular: 0, linear: 1, ideal: ~1.661, GS: ~2)
    #options.append("--module=4")
    options_string = " ".join(options)

    def timidity_convert_subprocess(filename, options_string, thread_results, thread_errors, thread_index):
        try:
            import os
            import platform
            import subprocess
            import re
            import sys

            base, ext = os.path.splitext(filename)
            num_string_match = re.search(r'\d+$', base)

            if num_string_match:
                # Number was found, extract it
                num_string = num_string_match.group()
            else:
                num_string = ""

            if log_to_file == True:
                # setup logger for thread
                log = logging.getLogger(threading.current_thread().name)
                log.setLevel(logging.DEBUG)

                # add file handler
                file_handler = logging.FileHandler(f"{os.getcwd()}/log-timidity-{thread_index}.log")
                formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
                file_handler.setFormatter(formatter)
                log.addHandler(file_handler)

            if platform.system() == 'Windows':
                # Get the absolute path of the package
                package_path = os.path.dirname(os.path.abspath(__file__))

                # Join the path to the binary file
                binary_path = os.path.join(package_path, 'bin', 'windows', 'TiMidity-2.15.0-w32', 'timidity.exe')
            else:
                # Assume the binary is in the PATH on other platforms (installable on MacOS)
                binary_path = 'timidity'

            command = [
                binary_path,
                options_string,
                f'-o {filename}.wav',
                f'{filename}.mid'
            ]

            command_string = " ".join(command)

            if log_to_file == True:
                log.info(f"{platform.system()}")
                log.info(f"cwd: {os.getcwd()}")
                log.info(f"package_path: {package_path}")
                log.info(f"timidity binary path: {binary_path}")
                log.debug(f"midi filename: {filename}.mid")
                log.debug(f"wav filename: {filename}.wav")
                log.info(f"timidity options: {options_string}")
                log.debug(f"executing subprocess command: {command_string}")
            
            thread_results[thread_index] = subprocess.run(command_string, cwd=package_path, capture_output = True, check=True)
            if log_to_file == True:
                log.debug(f"completed subprocess")

        except BaseException as e:
            thread_errors[thread_index] = (e, sys.exc_info())
            if log_to_file == True:
                log.exception("An error occurred:")
        if log_to_file == True:
            log.info(thread_results[thread_index].stdout.decode())
            log.info(thread_results[thread_index].stderr.decode())

    convert_files_mid_to_wav_timidity_threading(files, options_string, timidity_convert_subprocess, timeout=timeout)

def mix_wav_files(files, output_filename):
    import moviepy.editor as mpy
    from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
    
    audio_clips = []
    audio_file_clips = []
    for file in files:
        filename = file.replace("\\", "/")
        audio_file_clip = mpy.AudioFileClip(filename, nbytes=4, fps=44100)
        audio_file_clips.append(audio_file_clip)
        audio_array = audio_file_clip.to_soundarray(nbytes=4)
        total_time = audio_file_clip.duration
        audio_array_clip = AudioArrayClip(audio_array[0:int(44100 * total_time)], fps=44100)
        audio_clips.append(audio_array_clip)
    
    composed_audio_clip = CompositeAudioClip(audio_file_clips)
    output_filename_base = os.path.splitext(output_filename)[0]
    composed_audio_clip.write_audiofile(output_filename_base + ".wav", codec='pcm_s16le', fps=44100)

def convert_files_mid_to_wav_timidity_threading(files, options_string, timidity_convert_method, timeout=8):
    output_wav_files = []

    base_filenames = [os.path.splitext(filename.replace("\\", "/"))[0] for filename in files]
    
    # instantiate lists that store data from each thread
    thread_results = [None] * len(files)
    thread_errors = [None] * len(files)

    # start a thread for each file to be converted 
    threads = []
    for file_index, file in enumerate(files):
        t = threading.Thread(
                             target=timidity_convert_method, 
                             name=f"timidity-{file_index}", 
                             args=((base_filenames[file_index], options_string, thread_results, thread_errors, file_index))
                            )
        t.daemon = True
        threads.append(t)
        t.start()

    if len(files) > 1:
        print("Converting tracks " + str(base_filenames) + " to .wav using TiMidity++...")
    else:
        print("Converting " + str(base_filenames) + ".mid to .wav using TiMidity++...")

    # Wait for all threads to finish, with a timeout
    thread_has_timed_out = False
    for thread_index, t in enumerate(threads):
        t.join(timeout = timeout)
        base_filename = base_filenames[thread_index]
        wav_file = base_filename + ".wav"
        if t.is_alive() and not os.path.isfile(wav_file):
            # thread is still running and .wav file was not created, a conversion timeout has occured
            t._stop()
            print("Timidity++ timed out for " + base_filename + ".mid, skipping conversion")
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
        filename = files[thread_index]
        if thread_errors[thread_index] != None:
            # an error occured in the thread, show the error
            print(f"Errors in thread {thread_index}:")
            raise thread_errors[thread_index][0]

        if thread_result.returncode != 0:
            # timidity command failed in subprocess, show subprocess stdout and stderr
            print(f"Error from Timidity++ converting file {filename}")
            print(f"Output:")
            print(f"\"{thread_result.stdout.decode().strip()}\"")
            print(f"stderr:")
            print(f"\"{thread_result.stderr.decode().strip()}\"")
            print(f"")

    if len(files) == 1 and thread_has_timed_out == True:
        raise Exception(f"ERROR: Thread timed out ({timeout}s) during timidity mid to wav conversion for the only file {files[0]}")
    
    return output_wav_files

#def convert_midi_to_mp3_vlc(midi_filename_no_ext, wait_time = 3):
#    """ Uses headless VLC to convert a midi file to mp3 in the working directory.
#    Args:
#        midi_filename_no_ext: the name of the midi file in the working dir.
#        wait_time: The amount of time to wait after the VLC process has started. Used to make sure the process is finished before continuing execution.
#    """
#
#    string = 'vlc ' + midi_filename_no_ext + '.mid -I dummy --no-sout-video --sout-audio --no-sout-rtp-sap --no-sout-standard-sap --ttl=1 --sout-keep --sout "#transcode{acodec=mp3,ab=256}:std{access=file,mux=dummy,dst=./' + midi_filename_no_ext + '.mp3}"'
#    command_string = f"{string}"
#
#    def run_vlc():
#        import os
#        #print(string)
#        directories = os.system(command_string)
#
#    import threading
#    t = threading.Thread(target=run_vlc,name="vlc",args=())
#    t.daemon = True
#    t.start()
#
#    import time
#    print("Converting " + midi_filename_no_ext + ".mid midi to " + midi_filename_no_ext + ".mp3...")
#    time.sleep(wait_time)
#
#def convert_midi_to_wav_vlc(midi_filename_no_ext, wait_time = 3, separate_audio_files = True, output_combined_audio = True):
#    """ Uses headless VLC to convert a midi file to wav in the working directory.
#    Args:
#        midi_filename_no_ext: the name of the midi file in the working dir.
#        wait_time: The amount of time to wait after the VLC process has started. Used to make sure the process is finished before continuing execution.
#    """
#
#    string = 'vlc ' + midi_filename_no_ext + '.mid -I dummy --no-sout-video --sout-audio --no-sout-rtp-sap --no-sout-standard-sap --ttl=1 --synth-polyphony="65535" --sout-keep --sout "#transcode{acodec=s16l,channels=2}:std{access=file,mux=wav,dst=./' + midi_filename_no_ext + '.wav}"'
#    command_string = f"{string}"
#
#    import os
#
#
#    def run_vlc(midi_filename_no_ext):
#        import os
#
#        string = 'vlc ' + midi_filename_no_ext + '.mid -I dummy --no-sout-video --sout-audio --no-sout-rtp-sap --no-sout-standard-sap --ttl=1 --synth-polyphony="65535" --sout-keep --sout "#transcode{acodec=s16l,channels=2}:std{access=file,mux=wav,dst=./' + midi_filename_no_ext + '.wav}"'
#        command_string = f"{string}"
#
#        os.system(command_string)
#
#    import threading
#
#    if separate_audio_files == True:
#        try:
#            wav_files = glob.glob(midi_filename_no_ext + '-*.wav')
#            for file in wav_files:
#                os.remove(file)
#        except:
#            pass
#
#        files = glob.glob("./" + midi_filename_no_ext + '-*.mid')
#    else:
#        files = glob.glob("./" + midi_filename_no_ext + '.mid')
#    
#    filenames = []
#    for file in files:
#        filename = file.replace("\\", "/")
#        filename = os.path.splitext(filename)[0]
#        filenames.append(filename)
#        t = threading.Thread(target=lambda : run_vlc(filename),name="vlc convert",args=())
#        t.daemon = True
#        t.start()
#
#    import time
#    print("Converting " + str(filenames) + " midi files to .wav using VLC...")
#    time.sleep(wait_time)
#    if output_combined_audio == True:
#        if separate_audio_files == True:
#            import moviepy.editor as mpy
#            from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
#            files = glob.glob(midi_filename_no_ext + '-*.wav')
#            audio_clips = []
#            audio_file_clips = []
#            for file in files:
#                filename = file.replace("\\", "/")
#                audio_file_clip = mpy.AudioFileClip(filename, nbytes=4, fps=44100)
#                audio_file_clips.append(audio_file_clip)
#                audio_array = audio_file_clip.to_soundarray(nbytes=4)
#                total_time = audio_file_clip.duration
#                audio_array_clip = AudioArrayClip(audio_array[0:int(44100 * total_time)], fps=44100)
#                audio_clips.append(audio_array_clip)
#            
#            composed_audio_clip = CompositeAudioClip(audio_file_clips)
#            composed_audio_clip.write_audiofile(midi_filename_no_ext + ".wav",codec='pcm_s16le', fps=44100)
