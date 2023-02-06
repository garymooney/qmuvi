import qmuvi
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT


circ = QuantumCircuit(4)
# Equal superposition
circ.h(0)
circ.h(1)
circ.h(2)
circ.h(3)
circ.barrier()
# Cmaj Oracle
circ.x(0)
circ.x(1)
circ.x(2)
circ.x(3)
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
circ.x(2)
circ.barrier()
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
circ.x(3)
circ.barrier()
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
circ.x(0)
circ.x(1)
circ.x(3)
circ.barrier()
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
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
circ.compose(MCMT('z',3,1),inplace=True)
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

time_list = [[60,0]]*8+[[960,0]]+[[240,0]]*4+[[1920,0]]
qmuvi.generate_qmuvi(circ, 
                     "grover_Cmaj", 
                     noise_model = None, 
                     rhythm = time_list, 
                     phase_instruments = [qmuvi.get_instruments("windband")], 
                     invert_colours = False, 
                     fps = 5, 
                     smooth_transitions = True, 
                     log_to_file = False
                     )