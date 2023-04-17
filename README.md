
<img src="https://user-images.githubusercontent.com/6459545/182747409-52b2b800-4c01-45ca-a826-2120aa50fc5c.png" width="256">

An open-source **q**uantum **Mu**sic **Vi**deo tool

<img src="https://user-images.githubusercontent.com/6459545/182753376-bf01d486-6310-4e17-bee5-37ff5b2cb088.png" width="700">

qMuVi is a python library that can be used in your [qiskit](https://qiskit.org/) project to transform quantum circuits into music videos.

Quantum computing is notorious for being unintuitive and difficult to imagine. This tool attempts to create a connection between a human observer and the complex workings of quantum computation. By transforming quantum circuits into music videos, it allows you to "hear" and "see" how a quantum state evolves as it is processed by a quantum algorithm.

<video src="https://user-images.githubusercontent.com/14325614/227772719-a341769d-b578-4d23-9e7d-c18462bd1c22.webm" alt="grover_Cmaj"></video>

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

# Setup
**Python version:** 3.10.8 (should work for &ge;3.7 and possibly lower)

**Python packages:** `pip install -r requirements.txt`  (to replicate our dev environment)

or install the following libs: `qiskit==0.37.0`, `mido==1.2.10`, `moviepy==1.0.3`, `matplotlib==3.5.2`, and `pylatexenc`,

# How to use:
Just add barrier gates to your quantum circuit and call the `generate_qmuvi` method, that's it! Examples are found in the [demos](https://github.com/garymooney/qmuvi/blob/main/demos/) folder. Documentation can be found on the GitHub Pages site here: [https://garymooney.github.io/qmuvi](https://garymooney.github.io/qmuvi)

## Example:
A simple example of the 3-qubit Quantum Fourier Transform:
```python
import qmuvi
import qiskit
from qiskit import QuantumCircuit
from math import pi

circ = QuantumCircuit(3)
circ.x(0)
circ.x(1)
circ.barrier() # qMuVi will play the state as a sound when it encounters a barrier gate.
circ.h(0)
circ.barrier()
circ.crz(pi/2, 1, 0)
circ.barrier()
circ.crz(pi/4, 2, 0)
circ.barrier()
circ.h(1)
circ.barrier()
circ.crz(pi/2, 2, 1)
circ.barrier()
circ.h(2)
circ.barrier()
circ.barrier()

qmuvi.generate_qmuvi(circ, "simple_qft3")
```

Running the script will output the generated files, including the MP4 video, into a folder with the specified name "simple_qft3".

## Customisation
Properties in qMuVi can be customised using optional arguments in the `generate_qmuvi` method.

qMuVi provides simple customisation options such as `invert_colours`, `fps`, `smooth_transitions`, and `show_measured_probabilities_only`, along with more advanced options which are described below.

### _noise_model_
A [qiskit.providers.aer.noise.NoiseModel](https://qiskit.org/documentation/apidoc/aer_noise.html) object. A simple depolarising noise model can be obtained using the `get_simple_noise_model` method in the `qmuvi.quantum_simulation` module, or you can make your own.

**How it works:** To sample quantum states, qMuVi uses the qiskit AerSimulator to simulate the circuits. A noise model can be
passed to the simulator to include noise in the computations, which is translated to the generated output in qMuVi.

**Example:**
```python
import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
# define quantum circuit ...
noise_model = get_simple_noise_model(gate_error_rate_1q = 0.01,
                                     gate_error_rate_cnot = 0.02)

qmuvi.generate_qmuvi(circ, "simple_qft3", noise_model = noise_model)
```

### _rhythm_
A list of tuples in the form `(sound_time, rest_time)`.

**How it works:** Each tuple `(sound_time, rest_time)` in the list corresponds to a sampled quantum
            state in the circuit. The `sound_time` is how long the sound will play for and the `rest_time` is the wait
            time afterwards before playing the next sound. Times are in units of ticks where 480 ticks is 1 second.

**Example:**
```python
import qmuvi
# define quantum circuit ...
rhythm = [[200,40]]*7+[[960,0]]

qmuvi.generate_qmuvi(circ, "simple_qft3", rhythm = rhythm)
```

### _instruments_
A list of instruments as _int_s defined by the General MIDI standard. Premade instrument collections can be obtained using the `get_instrument_collection` method, or you can make your own.

<table>
  <tr>
    <td colspan="3"><b>Instrument collections</b></td>
  </tr>
  <tr>
    <td>'piano'</td> <td>'tuned_perc'</td> <td>'organ'</td>
  </tr>
  <tr>
    <td>'guitar'</td> <td>'bass'</td> <td>'strings'</td>
  </tr>
  <tr>
    <td>'ensemble'</td> <td>'brass'</td> <td>'reed'</td>
  </tr>
  <tr>
    <td>'pipe'</td> <td>'synth_lead'</td> <td>'synth_pad'</td>
  </tr>
  <tr>
    <td>'synth_effects'</td> <td>'ethnic'</td> <td>'percussive'</td>
  </tr>
  <tr>
    <td>'sound_effects'</td> <td>'windband'</td> <td></td>
  </tr>
</table>

**How it works:** An instrument collection is assigned to each pure state in the quantum state decomposition and the phase of the basis state determines the instrument in the collection.

**Example:**
```python
import qmuvi
# define quantum circuit ...
instruments = [qmuvi.get_instrument_collection("pipe"),
               qmuvi.get_instrument_collection("reed"),
               qmuvi.get_instrument_collection("brass"),
               qmuvi.get_instrument_collection("organ")]

qmuvi.generate_qmuvi(circ, "simple_qft3", instruments = instruments)
```
### _note_map_
A callable object that maps state numbers to note numbers. Premade note maps can be found in the `qmuvi.musical_processing` module, or you can make your own.

| Note Maps |
| --------- |
| `note_map_chromatic_middle_c` |
| `note_map_c_major` |
| `note_map_f_minor` |
| `note_map_c_major_arpeggio` |

**How it works:** The note map is used to convert the basis state numbers to the note numbers when generating the MIDI. A note number of 60 is middle C. You can make your own by for example defining a lambda function: `note_map = lambda n: n+60` or a method with the signiture `note_map(n: int) -> int`.

**Example:**
```python
import qmuvi
from qmuvi.musical_processing import note_map_c_major_arpeggio
# define quantum circuit ...

qmuvi.generate_qmuvi(circ, "simple_qft3", note_map = note_map_c_major_arpeggio)
```

# Features
* Generate a music video from a Qiskit quantum circuit.
