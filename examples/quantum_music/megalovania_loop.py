from qiskit import QuantumCircuit

# Add the qmuvi path so that we can import qmuvi (if you have installed qmuvi, you can skip this step)
import sys
sys.path.append(r"../..")
import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model


loop_repeats = 8

circ = QuantumCircuit(5)

circ.x(0)
circ.x(1)
circ.x(2)
circ.h(4)
circ.cx(4,2)
i = 0
while i < loop_repeats:
    i += 1
    circ.barrier()
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,1)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    circ.x(1)
    circ.cx(4,3)
    circ.cx(4,1)
    circ.cx(4,0)
    circ.barrier()
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,1)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    circ.x(0)
    circ.cx(4,3)
    circ.cx(4,0)
    circ.barrier()
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,1)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    circ.x(0)
    circ.x(1)
    circ.x(2)
    circ.cx(3,4)
    circ.barrier()
    circ.barrier()
    circ.cx(3,4)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,0)
    circ.x(1)
    circ.x(2)
    circ.cx(4,2)
    circ.barrier()
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,2)
    circ.cx(4,0)
    circ.barrier()
    circ.cx(4,3)
    circ.cx(4,2)
    circ.cx(4,1)
    circ.barrier()
    if i != loop_repeats:
        circ.x(1)
        circ.cx(4,3)
        circ.cx(4,0)


time_list = [(60,0),(60,0),(60,60),(60,120),(60,60),(60,60),(120,0),(60,0),(60,0),(60,0)] * 4 * loop_repeats
chromatic_G1 = lambda n: n + 31

qmuvi.generate_qmuvi(circ,
                     "megalovania_loop",
                     noise_model = get_simple_noise_model(0.01, 0.02),
                     rhythm = time_list,
                     instruments = [[81],[80]],
                     note_map = chromatic_G1,
                     invert_colours = True,
                     fps = 24,
                     smooth_transitions = True
                     )
