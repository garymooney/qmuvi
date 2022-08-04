
<img src="https://user-images.githubusercontent.com/6459545/182747409-52b2b800-4c01-45ca-a826-2120aa50fc5c.png" width="256">

An open-source **q**uantum **Mu**sic **Vi**deo tool 

<img src="https://user-images.githubusercontent.com/6459545/182753376-bf01d486-6310-4e17-bee5-37ff5b2cb088.png" width="700">

qMuVi is a python library that can be used by your qiskit project to convert your quantum circuits into music video files. 

Quantum computing is notorious for being unintuitive and difficult to imagine. This tool attempts to create some kind of connection between a human observer and the complex workings of quantum computation. By transforming quantum circuits into music videos, it allows you to "hear" and "see" how a quantum state evolves as it is processed by a quantum algorithm.

# Qiskit Hackathon winning project
qMuVi was originally created for the Qiskit Hackathon Melbourne 2022 and won first place by the judges and also won the community vote.  


<img src="https://user-images.githubusercontent.com/6459545/179168389-ee36690b-0cc8-4192-becd-1e699b179ce3.png" width="512">
From left to right, our team was Yang Yang, Gary Mooney (team leader), Harish Vallury, and Gan Yu Pin.

# How it works
The quantum circuits are run on Qiskit Aer’s simulator which supports noise models that are used to mimic noise in a physical quantum device. For your quantum circuit, you are able to specify a noise model to be applied to the quantum simulation during qMuVi's sampling of the states. This is particularly useful in understanding how the noise present on a real quantum computer will affect the outcome of your states.

Various instruments that play your music can be selected easily using the get_instruments() method, with a range of predefined collections, including piano, tuned percussion, organ, guitar etc. Behind the scenes, the instruments are assigned as integers according to the [General MIDI](https://en.wikipedia.org/wiki/General_MIDI) standard, which the [Mido Python library](https://mido.readthedocs.io/en/latest/index.html) uses to generate the music using the specified digital instruments.

There are three note maps that are provided by default, the chromatic C scale, the C major scale and the F minor scale, which are used to map the quantum basis state numbers (e.g. |0>, |2>, |7>, etc...) to MIDI note numbers. For example, the C major scale map works by adding 60 to the state number so that the |0> basis state is mapped to middle C, then rounds the note number down to the nearest note in the C major scale. This mapping is readily customised by defining your own method that maps an _int_ to an _int_.

Another important part to music is the rhythm. The note play duration as well as the rest time after the note is defined as a list of tuples. The tuples specify the play and rest time in units of ticks (where 480 ticks is 1 second) for the sound samples of the quantum state.

The music video output will display a visual representation of your input circuit. The quantum state is visualised by animating plots that show various important information, such as the fidelity and the probability distribution of basis states with colours representing their phases.

Once your quantum circuit, instruments and rhythm are defined (and optionally noise model and note map), you can input these parameters into methods such as  _make_music_video()_ or _make_music_midi()_ to generate a music video file or a raw MIDI file respectively. See below for code examples.  

# Mapping quantum to music
<img src="https://user-images.githubusercontent.com/6459545/179481509-843ede43-20a9-4392-916e-3e6b4757bbe7.png" width="220">
A quantum simulator is used to sample density matrices representing physical quantum states at various time steps. Density matrices can keep track of the uncertainty of the system's actual quantum state by containing all of the possible states that could have been produced by the noise model. The exact quantum state of a system is called a pure state. As incoherent noise is introduced to a density matrix representing a pure state, it can be thought of as changing from a single pure state to a statistical mixture of pure states due to the uncertainty introduced by the noise. When this happens, we say that the density matrix represents a mixed state. A density matrix can be written as its statistical distribution of pure states by performing an eigendecomposition where the eigenvectors are the pure states and the eigenvalues are the corresponding probabilities in the distribution.
<br />
<br />

<img src="https://user-images.githubusercontent.com/6459545/177944433-b3ea5a8e-d750-47c6-a1e2-58357f3db3ce.png" height="110">

<img src="https://user-images.githubusercontent.com/6459545/177961479-e6dc704e-9fb1-43b1-858c-674c414a743a.png" height="160">

Each pure state is a statevector representing a superposition of basis states, e.g. (statevector) = (a|00> + b|11>) / sqrt(2) = (a|0> + b|3>) / sqrt(2). The basis state numbers (e.g. 0 and 3) of the superposition of states are mapped (using a note map) to the MIDI note numbers to be played. The volume of each note is calculated as the probability of the basis state in the statevector (e.g. |a|^2 and |b|^2) multiplied by the probability of the pure state in the statistical distribution. So each pure state is a chord and because the statistical distribution can have multiple pure state terms, this means that multiple chords can be playing at the same time.

Each pure state of the statistical distribution is assigned a instrument collection. The instrument in the collection that will be used to play a note is determined by the corresponding basis state's phase in the superposition. The angles are discretised to match the size of the collection, where an angle of zero corresponds to the first instrument. A list of up to 8 instrument collections should be specified when making the music video. The collections from the list will be assigned to pure states in the statistical distribution in order of decreasing probability. If there are less than 8 collections specified, the remaining pure states will use the last instrument collection in the list. 

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
# make_music_midi(circ, "my_quantum_midi", rhythm, depolarising_noise_model, intruments, note_map=chromatic_middle_c)

# Converts the circuit to music and video and outputs all generated content and an .mp4 file
make_music_video(circ, "my_quantum_video", rhythm, depolarising_noise_model, intruments, note_map=chromatic_middle_c, invert_colours = False, fps=60)
```

Run the python script and it should output all the content into a folder with the specified name (e.g. "my_quantum_video").  

# Methods
### make_music_video(qc, name, rhythm, single_qubit_error, two_qubit_error, instrument_collections, note_map, invert_colours, fps)
Generates a music video from a qiskit quantum algorithm with barriers  
_**qc:**_ quantum circuit (qiskit QuantumCircuit)  
_**name:**_ name of the music video file and the folder that the data will be saved to (string)  
_**rhythm:**_ the timings for each sample (list of tuples (int, int). First element is note length, second element is rest time. Units of ticks 480 = 1 sec).  
_**single_qubit_error:**_ single qubit depolarisation noise (float)  
_**two_qubit_error:**_ two-qubit depolarisation noise (float)  
_**instrument_collections:**_ list of instrument collections for each pure state. Instrument for note is chosen from collection based on state phase (list of list of ints)  
_**note_map:**_ the note map to convert from state number to note number. Middle C is 60 (default: chromatic_middle_c) (map from int to int)  
_**invert_colours:**_ whether to invert the colours of the video so that the background is black (default: False) (bool)  
_**fps:**_ the fps of the output video (default: 60) (int)

### make_music_midi(qc, name, rhythm, single_qubit_error, two_qubit_error, instrument_collections, note_map)
Generates music from a qiskit quantum algorithm with barriers and outputs it as a midi file  
_**qc:**_ quantum circuit (qiskit QuantumCircuit)  
_**name:**_ name of the music file and the folder that the data will be saved to (string)  
_**rhythm:**_ the timings for each sample (list of tuples (int, int). First element is note length, second element is rest time. Units of ticks 480 = 1 sec).  
_**single_qubit_error:**_ single qubit depolarisation noise (float)  
_**two_qubit_error:**_ two-qubit depolarisation noise (float)  
_**instrument_collections:**_ list of instrument collections for each pure state. Instrument for note is chosen from collection based on state phase (list of list of ints)  
_**note_map:**_ the note map to convert from state number to note number. Middle C is 60 (default: chromatic_middle_c) (map from int to int)

### get_instruments(instrument_collection_name)
Gets a list of integers corresponding to instruments according to the standard General MIDI (see https://en.wikipedia.org/wiki/General_MIDI)  
_**instrument_collection_name:**_ the name of a predefined collection (string)  
  
**Current predefined collections:**  
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
This project uses Python 3. Download the repo and use the example .py files as a starting point.
  
## Convert to midi:
Python packages: qiskit==0.37.0, and mido==1.2.10. Earlier versions will probably work too.

## Convert to wav:
### Windows
The project already comes with a verion of TiMidity++ that runs on windows. So you don't need to do anything.

#### (optional) Use VLC player instead:
If preferred, you can use headless VLC to convert midi files to wav files instead. To do this, install VLC player and add its install path to the PATH system variable. 

VLC needs to be configured to use a sound font (.sf2 file). If VLC can play a midi file successfully then it is already configured. There is one located in the folder "GeneralUser GS 1.471", however you can use whichever modern SoundFont you like. Go to VLC preferences (show all settings) -> Input/Codecs -> Audio codecs -> FluidSynth -> Browse to SoundFont location.  

### Mac OS
You should install either TiMidity++ or VLC player to do the MIDI to wav conversions.

#### TiMidity++
Use your package manager to install timidity. (I'll add instructions to set up the soundfont in TiMidity++ for MacOs later, there are instructions online somewhere)

#### VLC player
Download and install VLC player. Create a symlink of VLC in the usr/local/bin/ directory (or some other directory already in the PATH environment variable) with the following command.

```ln -s Application/VLC.app/Contents/MacOS/VLC usr/local/bin/```

Note: if you haven't done this before you may need to create the usr/local/bin/ directory first.

VLC needs to be configured to use a sound font (.sf2 file). If VLC can play a midi file successfully then it is already configured. There is one located in the folder "GeneralUser GS 1.471", however you can use whichever modern SoundFont you like. Setting it up would be something like VLC preferences (show all settings) -> Input/Codecs -> Audio codecs -> (whatever MIDI synth program VLC MacOs uses) -> Browse to SoundFont location.  

### Alternative method
You can do this process manually using whatever method you like instead of using TiMidity++ or VLC to automatically do it. Just output the circuit as a MIDI file first using _make_music_midi_, then convert to WAV using whatever method you like, then put the WAV file in the folder with the MIDI (and other content) and give it the same name. Then use the _make_video_ method to finish converting the quantum algorthm to video. The _make_video_ method will use the WAV file in the folder instead of generating a new one.

## Convert to video:
Python packages: qiskit==0.37.0, moviepy==1.0.3, mido==1.2.10, matplotlib==3.5.2, and pylatexenc. Earlier versions will probably work too.

#### Mac OS
The ffmpeg codec for video processing may not be installed after installing MoviePy. If so, install it manually using your package manager. E.g.

```brew install ffmpeg```
