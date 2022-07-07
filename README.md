# Expressing Quantum Computation Through Generated Art

A library that can turn Qiskit quantum circuits into music videos!

Add a barrier to the circuit for each time step that you would like to be sampled, then call the _make_music_video_ method. This will create a folder with all of the generated content inside it.

**make_music_video**  
_Arg 0:_ quantum circuit  
_Arg 1:_ name of the music video  
_Arg 2:_ list of tuples corresponding to the timings of each sample. first element is note length, second element is the rest time.  
_Arg 3:_ single qubit depolarisation noise  
_Arg 4:_ two-qubit depolarisation noise  
_Arg 5:_ list of instrument collections for each pure state. Instrument for note is chosen from collection based on state phase.  


For Example:
```
import quantum_music
from quantum_music import make_music_video, get_instruments
import qiskit
from qiskit import QuantumCircuit

circ = QuantumCircuit(4)
circ.h(0)
circ.barrier()
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
rhythm = [(120, 60)]*8
single_qubit_error = 0.02
two_qubit_error = 0.05

# instrument collections for get_instruments method
# 'piano': list(range(1,9))
# 'tuned_perc': list(range(9,17))
# 'organ': list(range(17,25))
# 'guitar': list(range(25,33))
# 'bass': list(range(33,41))
# 'strings': list(range(41,49))
# 'ensemble': list(range(49,57))
# 'brass': list(range(57,65))
# 'pipe': list(range(73,81))
                    
intruments = []
intruments.append([73]) # a pipe

intruments.append(get_instruments('tuned_perc'))
make_music_video(circ, "my_quantum_video", rhythm, single_qubit_error, two_qubit_error)
```
