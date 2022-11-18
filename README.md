
<img src="https://user-images.githubusercontent.com/6459545/182747409-52b2b800-4c01-45ca-a826-2120aa50fc5c.png" width="256">

An open-source **q**uantum **Mu**sic **Vi**deo tool 

<img src="https://user-images.githubusercontent.com/6459545/182753376-bf01d486-6310-4e17-bee5-37ff5b2cb088.png" width="700">

qMuVi is a python library that can be used in your [qiskit](https://qiskit.org/) project to transform quantum circuits into music videos. 

Quantum computing is notorious for being unintuitive and difficult to imagine. This tool attempts to create a connection between a human observer and the complex workings of quantum computation. By transforming quantum circuits into music videos, it allows you to "hear" and "see" how a quantum state evolves as it is processed by a quantum algorithm.  

**Showcase video:**
<div align="left">
      <a href="https://www.youtube.com/watch?v=seersj3W-hg" target="_blank" rel="noopener noreferrer">
         <img width="700" alt="youtube-dp" src="https://user-images.githubusercontent.com/6459545/186303286-0e0aec10-53e1-4ecd-9380-fe8d8ed9372b.png">
      </a>
</div>

# Qiskit Hackathon winning project
qMuVi was the winning project of the Qiskit Hackathon Melbourne 2022 hosted by [IBM Quantum Network Hub @ the University of Melbourne](https://www.unimelb.edu.au/quantumhub). It has continued to be developed since the competition.  

<img src="https://user-images.githubusercontent.com/6459545/179168389-ee36690b-0cc8-4192-becd-1e699b179ce3.png" width="512">
From left to right, our qiskit hackathon team was Yang Yang, Gary Mooney (team leader), Harish Vallury, and Gan Yu Pin.

# How it works
The quantum circuits are run on [Qiskit Aer’s simulator](https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html) which supports noise models ([NoiseModel](https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.NoiseModel.html)) that are used to mimic noise in a physical quantum device. For your quantum circuit, you are able to specify a noise model to be applied to the quantum simulation during qMuVi's sampling of the states. This is particularly useful in understanding how the noise present on a real quantum computer will affect the outcome of your states.

Various instruments that play your music can be selected easily using the get_instruments() method, with a range of predefined collections, including piano, tuned percussion, organ, guitar etc. Behind the scenes, the instruments are assigned as integers according to the [General MIDI](https://en.wikipedia.org/wiki/General_MIDI) standard, which the [Mido Python library](https://mido.readthedocs.io/en/latest/index.html) uses to generate the music using the specified digital instruments.

There are three note maps that are provided by default, the chromatic C scale, the C major scale and the F minor scale, which are used to map the quantum basis state numbers (e.g. |0>, |2>, |7>, etc...) to MIDI note numbers. For example, the C major scale map works by adding 60 to the state number so that the |0> basis state is mapped to middle C, then rounds the note number down to the nearest note in the C major scale. This mapping is readily customised by defining your own method that maps an _int_ to an _int_.

Another important part to music is the rhythm. The note play durations, as well as the rest times after the notes, are defined as a list of tuples. The tuples specify the play and rest time in units of ticks (where 480 ticks is 1 second) for the sound samples of the quantum state.

The [MoviePy Python library](https://zulko.github.io/moviepy/) is used to render the music videos which display a visual representation of your input circuit. The quantum state is visualised by animating figures that show various important information, such as the probability distribution of basis states for each pure state, with colours representing their phases.

Once your quantum circuit, instruments and rhythm are defined (and optionally noise model and note map), you can input these parameters into methods such as  _make_music_video()_ or _make_music_midi()_ to generate a music video file or a raw MIDI file respectively. See below for code examples.  

# Mapping quantum to music
<img src="https://user-images.githubusercontent.com/6459545/179481509-843ede43-20a9-4392-916e-3e6b4757bbe7.png" width="220">
A quantum simulator is used to sample density matrices representing physical quantum states at various time steps. Density matrices can keep track of the uncertainty of the system's actual quantum state by containing all of the possible states that could have been produced by the noise model. The exact quantum state of a system is called a pure state. As incoherent noise is introduced to a density matrix representing a pure state, it can be thought of as changing from a single pure state to a statistical mixture of pure states due to the uncertainty introduced by the noise. When this happens, we say that the density matrix represents a mixed state. A density matrix can be written as its statistical distribution of pure states by performing an eigendecomposition where the eigenvectors are the pure states and the eigenvalues are the corresponding probabilities in the distribution.
<br />
<br />

<img src="https://user-images.githubusercontent.com/6459545/177944433-b3ea5a8e-d750-47c6-a1e2-58357f3db3ce.png" height="110">

<img src="https://user-images.githubusercontent.com/6459545/177961479-e6dc704e-9fb1-43b1-858c-674c414a743a.png" height="160">

Each pure state is a statevector representing a superposition of basis states, e.g. (statevector) = (a|00> + b|11>) / sqrt(2) = (a|0> + b|3>) / sqrt(2). The basis state numbers (e.g. 0 and 3) of the superposition of states are mapped (using a note map) to the MIDI note numbers to be played. The volume of each note is calculated as the probability of the basis state in the statevector (e.g. |a|^2 and |b|^2) multiplied by the probability of the pure state in the statistical distribution. So each pure state is a chord and because the statistical distribution can have multiple pure state terms, this means that multiple chords can be playing at the same time.

Each pure state of the statistical distribution is assigned a instrument collection. The instrument in the collection that will be used to play a note is determined by the corresponding basis state's phase in the superposition. The angles are discretised to match the size of the collection, where an angle of zero corresponds to the first instrument. A list of up to 8 instrument collections can be specified when making the music video (see below example). The collections from the list will be assigned to pure states in the statistical distribution in order of decreasing probability. If there are less than 8 collections specified, the remaining pure states will use the last instrument collection in the list. 

# Example:
```
import quantum_music
from quantum_music import make_music_midi, make_music_video, get_instruments, chromatic_middle_c, get_depolarising_noise
import qiskit
from qiskit import QuantumCircuit

circ = QuantumCircuit(4)
circ.h(0)
circ.barrier() # signals to the quantum music transpiler to sample the state at this time step
circ.cx(0, 1)
circ.barrier()
circ.cx(1, 2)
circ.barrier()
circ.cx(2, 3)
circ.barrier()
circ.h(3)
circ.barrier()
circ.cx(3,2)
circ.barrier()
circ.cx(2,1)
circ.barrier()
circ.cx(1,0)
circ.barrier()

rhythm = [(120, 60)]*8 # sound length and rest time for each sample (length must match the number of barriers)

single_qubit_error = 0.02
two_qubit_error = 0.05
depolarising_noise_model = get_depolarising_noise(single_qubit_error, two_qubit_error)
                    
intruments = []
intruments.append([73]) # a pipe
intruments.append(get_instruments('tuned_perc'))

# Uncomment to convert the circuit to music and outputs a MIDI file
# make_music_midi(circ, "my_quantum_midi", rhythm, depolarising_noise_model, intruments, note_map=chromatic_middle_c, separate_audio_files=True)

# Converts the circuit to music and video and outputs all generated content and an .mp4 file
make_music_video(circ, "my_quantum_video", rhythm, depolarising_noise_model, intruments, note_map=chromatic_middle_c, invert_colours = False, fps=60, smooth_transitions=True, synth="timidity", output_logs = False)
```

Run the python script and it should output all the content into a folder with the specified name (e.g. "my_quantum_video").  

# Methods
### make_music_video(qc, name, rhythm, noise_model, instrument_collections, note_map, invert_colours, fps, smooth_transitions, synth, output_logs, probability_distribution_only)
Generates a music video from a qiskit quantum algorithm with barriers  
_**qc:**_ quantum circuit (qiskit QuantumCircuit)  
_**name:**_ name of the music video file and the folder that the data will be saved to (string)  
_**rhythm:**_ the timings for each sample (list of tuples (int, int). First element is note length, second element is rest time. Units of ticks 480 = 1 sec).  
_**noise_model:**_ a qiskit noise model to use for the simulation (default: None, will perform a noiseless simulation) ([NoiseModel](https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.NoiseModel.html))  
_**instrument_collections:**_ list of instrument collections for each pure state. Instrument for note is chosen from collection based on state phase (default: [list(range(81,89))], synth_lead) (list of list of ints)  
_**note_map:**_ the note map to convert from state number to note number. Middle C is 60 (default: chromatic_middle_c) (map from int to int)  
_**invert_colours:**_ whether to invert the colours of the video so that the background is black (default: False) (bool)  
_**fps:**_ the fps of the output video (default: 60) (int)  
_**smooth_transitions:**_ Whether to animate smooth transitions between histogram states. This slows down rendering significantly (default: True) (bool)  
_**synth:**_ the synth to use to convert mid to wav. Options are "timidity" and "vlc" (default: "timidity") (string)  
_**output_logs:**_ Outputs log files generated by timidity when converting mid files to wav. (default: False) (bool)  
_**probability_distribution_only:**_ Whether to only plot the final measurement probabilities rather than for each pure state. (default: False) (bool)  

### make_music_midi(qc, name, rhythm, noise_model, instrument_collections, note_map, separate_audio_files)
Generates music from a qiskit quantum algorithm with barriers and outputs it as a mid file  
_**qc:**_ quantum circuit (qiskit QuantumCircuit)  
_**name:**_ name of the music file and the folder that the data will be saved to (string)  
_**rhythm:**_ the timings for each sample (list of tuples (int, int). First element is note length, second element is rest time. Units of ticks 480 = 1 sec).  
_**noise_model:**_ a qiskit noise model to use for the simulation (default: None, will perform a noiseless simulation) ([NoiseModel](https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.NoiseModel.html))  
_**instrument_collections:**_ list of instrument collections for each pure state. Instrument for note is chosen from collection based on state phase (default: [list(range(81,89))], synth_lead) (list of list of ints)  
_**note_map:**_ the note map to convert from state number to note number. Middle C is 60 (default: chromatic_middle_c) (map from int to int)  
_**separate_audio_files:**_ whether to output separate mid files for each pure state in the mixed state ensemble. Good for maximum audio quality (up to 8 pure states). (default: True) (bool)  

### make_video(qc, name, rhythm, noise_model, instrument_collections, note_map, invert_colours, fps, smooth_transitions, separate_audio_files, probability_distribution_only)
Generates a video from data that was already generated using make_music_midi or make_music_video.  
_**qc:**_ quantum circuit (qiskit QuantumCircuit)  
_**name:**_ name of the music video file and the folder that the data will be saved to (string)  
_**rhythm:**_ the timings for each sample (list of tuples (int, int). First element is note length, second element is rest time. Units of ticks 480 = 1 sec).  
_**noise_model:**_ a qiskit noise model to use for the simulation (default: None, will perform a noiseless simulation) ([NoiseModel](https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.NoiseModel.html))  
_**instrument_collections:**_ list of instrument collections for each pure state. Instrument for note is chosen from collection based on state phase (default: [list(range(81,89))], synth_lead) (list of list of ints)  
_**note_map:**_ the note map to convert from state number to note number. Middle C is 60 (default: chromatic_middle_c) (map from int to int)  
_**invert_colours:**_ whether to invert the colours of the video so that the background is black (default: False) (bool)  
_**fps:**_ the fps of the output video (default: 60) (int)  
_**smooth_transitions:**_ Whether to animate smooth transitions between histogram states. This slows down rendering significantly (default: True) (bool)  
_**separate_audio_files:**_ whether to look for separate mid files for each pure state in the mixed state ensemble and combine them together for the video output. Good for maximum audio quality (up to 8 pure states). (default: True) (bool)  
_**probability_distribution_only:**_ Whether to only plot the final measurement probabilities rather than for each pure state. (default: False) (bool)  

### get_instruments(instrument_collection_name)
Gets a list of integers corresponding to instruments according to the standard General MIDI (see https://en.wikipedia.org/wiki/General_MIDI)  
_**instrument_collection_name:**_ the name of a predefined collection (string)  
  
**Predefined collections:**  
'piano': list(range(1,9))  
'tuned_perc': list(range(9,17))  
'organ': list(range(17,25))  
'guitar': list(range(25,33))  
'bass': list(range(33,41))  
'strings': list(range(41,49))  
'ensemble': list(range(49,57))  
'brass': list(range(57,65))  
'pipe': list(range(73,81))  
'synth_lead': list(range(81,89))  
'synth_pad': list(range(89,97))  
'synth_effects': list(range(97,105))  
'ethnic': list(range(105,113))  
'percussive': list(range(113,121))  
'sound_effects': list(range(121,128))  
'windband': [74,69,72,67,57,58,71,59]  

### Note Maps
**chromatic_middle_c(state_number)**  
**c_major(state_number)**  
**f_minor(state_number)**  

The predefined note maps. Returns a note number calculated as the input state number + 60 so that 0 is middle C, then rounded down to the nearest note of the scale.  
_**state_number:**_ the state number (int)  

# Setup
Download the repo and use the example .py files as a starting point (python 3). 

There are three conversion processes in qMuVi to configure. Each of the conversions below can be used independently without needing to configure the other ones (apart from video, this will be made independent soon). The conversions are: 
- **Output to mid file:** Process and output the quantum circuit as data files and MIDI (mid) file(s). MIDI is a standard format that describes the sounds to play and when to play them.  
- **Convert mid to wav:** wav files are the typical audio files that can be directly listened to using any audio playback software.
- **Convert to video:** Generate a video using the data files and wav file(s). If there were no wav files generated, the video will be silent.
  
## Output to mid file (MIDI):
Python packages: qiskit==0.37.0, and mido==1.2.10. Earlier versions will probably work too.

## Convert mid to wav:
The conversion can either be configured to be automatic using the TiMidity++ or VLC programs, or the conversion can be done manually (see **Manual Conversion** below) 

### Windows
The project already comes with a verion of TiMidity++ that runs on windows. So qMuVi should already be configured to automatically convert mid files to wav files.

#### (optional) Use VLC player instead:
If preferred, you can use headless VLC to convert mid files to wav files instead. To do this, install VLC player and add its install path to the PATH system variable. 

VLC needs to be configured to use a sound font (.sf2 file). If VLC can play a midi file successfully then it is already configured. There is one located in the folder "GeneralUser GS 1.471", however you can use whichever modern SoundFont you like. Go to VLC preferences (show all settings) -> Input/Codecs -> Audio codecs -> FluidSynth -> Browse to SoundFont location.  

### Mac OS
You should install either TiMidity++ or VLC player to do the MIDI to wav conversions.

#### TiMidity++
Use your package manager to install timidity. (I'll add instructions to set up the soundfont in TiMidity++ for MacOs later, there are instructions online somewhere)

#### VLC player
Download and install VLC player. Create a symlink of VLC in the usr/local/bin/ directory (or some other directory already in the PATH environment variable) with the following command.

```ln -s Application/VLC.app/Contents/MacOS/VLC usr/local/bin/```

Note: if you haven't done this before you may need to create the usr/local/bin/ directory first.

VLC needs to be configured to use a sound font (.sf2 file). If VLC can play a mid file successfully then it is already configured. There is one located in the folder "GeneralUser GS 1.471", however you can use whichever modern SoundFont you like. Setting it up would be something like VLC preferences (show all settings) -> Input/Codecs -> Audio codecs -> (whatever MIDI synth program VLC MacOs uses) -> Browse to SoundFont location.  

### Manual Conversion (mid -> wav)
You can do this process manually using whatever method you like instead of using TiMidity++ or VLC to automatically do it. Just output the circuit as a MIDI file first using _make_music_midi_, then convert to WAV using whatever method you like, then put the WAV file in the folder with the MIDI (and other content) and give it the same name. Then use the _make_video_ method to finish converting the quantum algorthm to video. The _make_video_ method will use the WAV file in the folder instead of generating a new one.

## Convert to video:
Python packages: qiskit==0.37.0, moviepy==1.0.3, mido==1.2.10, matplotlib==3.5.2, and pylatexenc. Earlier versions will probably work too.

#### Mac OS
The ffmpeg codec for video processing may not be installed after installing MoviePy. If so, install it manually using your package manager. E.g.

```brew install ffmpeg```
