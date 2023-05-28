import qiskit
from qiskit import QuantumCircuit
from numpy import pi

import qmuvi
from qmuvi.musical_processing import note_map_c_major_arpeggio


circ = QuantumCircuit(1)

circ.x(0)
circ.ry(pi/8, 0)
circ.barrier()
circ.rz(pi/4, 0)
circ.barrier()
circ.rz(pi/4, 0)
circ.barrier()
circ.rz(pi/4, 0)
circ.barrier()
circ.rz(pi/4, 0)
circ.barrier()
circ.rz(pi/4, 0)
circ.barrier()
circ.rz(pi/4, 0)
circ.barrier()
circ.rz(pi/4, 0)
circ.barrier()
circ.rz(pi/4, 0)
circ.barrier()

time_list = [[200,0]]*8 + [[400,0]]

instruments = []
instruments.append(qmuvi.get_instrument_collection("organ"))

qmuvi.generate_qmuvi(circ,
                     "single_qubit",
                     noise_model = None,
                     rhythm = time_list,
                     instruments = instruments,
                     note_map = note_map_c_major_arpeggio,
                     invert_colours = True,
                     fps = 10,
                     smooth_transitions = False
                     )
