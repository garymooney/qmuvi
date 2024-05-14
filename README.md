[![code tests](https://github.com/garymooney/qmuvi/actions/workflows/code-tests.yml/badge.svg)](https://github.com/garymooney/qmuvi/actions/workflows/code-tests.yml)
[![docs](https://github.com/garymooney/qmuvi/actions/workflows/docs.yml/badge.svg)](https://github.com/garymooney/qmuvi/actions/workflows/docs.yml)

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
<br>


# Setup
To install qMuVi, you can use `pip`:

```bash
pip install qmuvi
```

## Linux
For Linux, qMuVi requires timidity to be installed separately. Instructions for common distributions:
```bash
pacman -S timidity++ # Arch Linux
dnf install timidity++ # Fedora
apt-get install timidity # Debian and Ubuntu
zypper install timidity # OpenSUSE
```

## Install From Source
For the latest version from source, you can install using poetry:

1. Ensure [poetry](https://python-poetry.org/docs/) is installed.
2. Clone the qMuVi repo and navigate to root.
```
git clone https://github.com/garymooney/qmuvi.git
cd qmuvi
```
3. Install qMuVi using poetry.
```
poetry install -E test -E doc -E dev
```
Note. Use `poetry run` to execute commands within the virtual environment managed by poetry (if not already in a virtual environment).

4. Run the tests to verify that everything is working.
```
poetry run pytest -s tests
```

# Usage
```python
import qmuvi
from qiskit import QuantumCircuit

circ = QuantumCircuit(2)
# Barrier gates tell qMuVi where to sample the state in the circuit
circ.barrier()
circ.h(0)
circ.barrier()
circ.cx(0, 1)
circ.barrier()

qmuvi.generate_qmuvi(circ, "bell_state")
```

Running the script will output the generated files, including the MP4 video, into a folder with the specified name "bell_state".

Examples can be found in the [examples](https://github.com/garymooney/qmuvi/tree/main/examples) folder with rendered outputs displayed in [qmuvi_gallery](https://github.com/garymooney/qmuvi/tree/main/qmuvi_gallery).

Documentation: [https://garymooney.github.io/qmuvi](https://garymooney.github.io/qmuvi)

# Properties
Properties in qMuVi can be customised using optional arguments in the `generate_qmuvi` method.

qMuVi provides simple customisation options such as `invert_colours`, `fps`, `smooth_transitions`, and `show_measured_probabilities_only`, along with more advanced options which are described below.

## Property - _noise_model_
A [Qiskit NoiseModel](https://qiskit.org/documentation/apidoc/aer_noise.html) object. A simple depolarising noise model can be obtained using the `get_simple_noise_model` method in the `qmuvi.quantum_simulation` module, or you can make your own.

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

## Property - _rhythm_
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

## Property - _instruments_
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
## Property - _note_map_
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
* **Quantum Circuit** (Qiskit) $\rightarrow$ **Music Video File** (as MP4).
* **Quantum Circuit** (Qiskit) $\rightarrow$ **Music Files** (as MIDI and WAV).
