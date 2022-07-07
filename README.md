# expressing-quantum-computation-through-generated-art

A library that can turn Qiskit quantum circuits into music videos!

Add a barrier to the circuit for each time step that you would like to be samples, then call the make_music_video method.

Method: make_music_video
Arg 0: quantum circuit
Arg 1: name of the music video
Arg 2: list of tuples corresponding to the timings of each sample. first element is note length, second element is the rest time.
Arg 3: single qubit depolarisation noise
Arg 4: two-qubit depolarisation noise

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
make_music_video(circ, "hi", [(120, 60)]*8, 0.02, 0.05)
```
