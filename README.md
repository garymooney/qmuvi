# expressing-quantum-computation-through-generated-art

A library that can turn Qiskit quantum circuits into music videos!

Add a barrier to the circuit for each time step that you would like to be sampled, then call the _make_music_video_ method.

**make_music_video**  
_Arg 0:_ quantum circuit  
_Arg 1:_ name of the music video  
_Arg 2:_ list of tuples corresponding to the timings of each sample. first element is note length, second element is the rest time.  
_Arg 3:_ single qubit depolarisation noise  
_Arg 4:_ two-qubit depolarisation noise  

For Example:
```
import quantum_music
from quantum_music import make_music_video
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

make_music_video(circ, "my_quantum_video", rhythm, single_qubit_error, two_qubit_error)
```
