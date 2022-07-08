# Expressing Quantum Computation Through Generated Art

A python module that can transform Qiskit quantum circuits into music videos!

Add a barrier to a quantum circuit for each time step to sample the quantum state for, then call the _make_music_video_ method. This will create a folder with all of the generated content inside it.

# Mapping quantum to music
  
![image](https://user-images.githubusercontent.com/6459545/177944433-b3ea5a8e-d750-47c6-a1e2-58357f3db3ce.png)  
![image](https://user-images.githubusercontent.com/6459545/177961479-e6dc704e-9fb1-43b1-858c-674c414a743a.png)

A noisy quantum density matrix simulator is used to sample mixed states (a probabilistic mixture of pure states) at various time steps. Each pure state is a state vector representing a superposition of computational basis states. A state with no noise is a pure state and so will only have a single non-zero element in the mixture. As noise is introduced, more terms with non-zero probability will appear. The superposition of states in a pure state will determine the notes that are played. The mapping from integer representation of the state to note number can be customised by passing in a method that takes an int and returns an int (60 is middle C), this allows the possibility to map states to notes of musical scales. A list of instrument collections (see example below) can be specified to assign, in order of decreasing probability, instrument collections to the pure states of the mixture (the instrument collection list can be manually specified in the method call). A maximum of 8 instrument collections can be specified (if there are less than 8, the remaining pure states will use the last instrument collection in the list). The instrument for each note is chosen from the collection by the state's phase angle in the superposition. The angles are discretised to match the size of the collection, where an angle of zero corresponds to the first instrument. The velocity (which is proportional to volume) of each note is calculated by multiplying the propability of the pure state in the mixture and the probability of the computational basis state of the pure state's superposition, normalised such that there is always a note with velocity equal to 1.  

# Methods
### make_music_video(qc, name, rhythm, single_qubit_error, two_qubit_error, instrument_collections, note_map)
Generates a music video from a qiskit quantum algorithm with barriers  
_**qc:**_ quantum circuit (qiskit QuantumCircuit)  
_**name:**_ name of the music video (string)  
_**rhythm:**_ the timings for each sample (list of tuples (int, int). First element is note length, second element is rest time. Units of ticks 480 = 1 sec).  
_**single_qubit_error:**_ single qubit depolarisation noise (float)  
_**two_qubit_error:**_ two-qubit depolarisation noise (float)  
_**instrument_collections:**_ list of instrument collections for each pure state. Instrument for note is chosen from collection based on state phase (list of list of ints)  
_**note_map:**_ the note map to convert from state number to note number. Middle C is 60 (map from int to int)  

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
'windband': [74,69,72,67,57,58,71,59]  

### chromatic_middle_c(state_number)
Used for the note map. Returns a note number calculated as the input state number + 60 such that state number zero is middle C.  
_**state_number:**_ the state number (int)  

### c_major(state_number)
Used for the note map. Returns a note number calculated as the input state number + 60 then rounded down to a note in the C major scale.  
_**state_number:**_ the state number (int)  

### f_minor(state_number)
Used for the note map. Returns a note number calculated as the input state number + 60 then rounded down to a note in the F minor scale.  
_**state_number:**_ the state number (int)  

# Example:
```
import quantum_music
from quantum_music import make_music_video, get_instruments, chromatic_middle_c
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
rhythm = [(120, 60)]*8 # sound length and rest time for each sample
single_qubit_error = 0.02
two_qubit_error = 0.05

                    
intruments = []
intruments.append([73]) # a pipe
intruments.append(get_instruments('tuned_perc'))

make_music_video(circ, "my_quantum_video", rhythm, single_qubit_error, two_qubit_error, intruments, note_map=chromatic_middle_c)
```

Run the python script and it should output all the content into a folder with the specified name (e.g. "my_quantum_video").  
**Warning:** Using numbers in the name sometimes causes an error.

# Setup
Install VLC player and add its install path to the PATH system variable so that headless VLC player can be used to convert MIDI to MP3. For the moment, before the conversion will work, VLC needs to be configured to use a sound font (.sf2 file). There is one currently in the repo. VLC should tell you how to configure the sound font by asking it to play a midi file. If it plays the midi file successfully then it should be configured correctly.  
  
This project uses Python 3 and a few python packages, notably qiskit==0.37.0, moviepy==1.0.3, mido==1.2.10, and matplotlib==3.5.2 (there might be others). Earlier versions will probably work too.
