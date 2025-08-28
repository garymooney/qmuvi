import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate

# Add the qmuvi path so that we can import qmuvi (if you have installed qmuvi, you can skip this step)
import sys
sys.path.append(r"../..")
import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
from qmuvi.musical_processing import note_map_f_minor


circ = QuantumCircuit(4)

# Equal superposition
circ.h(0)
circ.h(1)
circ.h(2)
circ.h(3)
circ.barrier()

# Fmin7 Oracle
circ.x(0)
circ.x(1)
circ.x(2)
circ.compose(MCMTGate(ZGate(),3,1),inplace=True)
circ.barrier()
circ.x(2)
circ.barrier()
circ.compose(MCMTGate(ZGate(),3,1),inplace=True)
circ.barrier()
circ.x(0)
circ.x(3)
circ.barrier()
circ.compose(MCMTGate(ZGate(),3,1),inplace=True)
circ.barrier()
circ.x(1)
circ.x(2)
circ.barrier()
circ.compose(MCMTGate(ZGate(),3,1),inplace=True)
circ.barrier()
circ.x(2)
circ.x(3)
circ.barrier()

# Inversion
circ.h(0)
circ.h(1)
circ.h(2)
circ.h(3)
circ.barrier()
circ.x(0)
circ.x(1)
circ.x(2)
circ.x(3)
circ.barrier()
circ.compose(MCMTGate(ZGate(),3,1),inplace=True)
circ.barrier()
circ.x(0)
circ.x(1)
circ.x(2)
circ.x(3)
circ.barrier()
circ.h(0)
circ.h(1)
circ.h(2)
circ.h(3)
circ.barrier()

time_list = [(60,0)]*8+[(960,0)]+[(240,0)]*4+[(1920,0)]

qmuvi.generate_qmuvi(circ,
                     "grover_Fmin_seven",
                     noise_model = None,
                     rhythm = time_list,
                     instruments = [qmuvi.get_instrument_collection("windband")],
                     note_map = note_map_f_minor,
                     invert_colours = False,
                     fps = 24,
                     smooth_transitions = True
                     )
