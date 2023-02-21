import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
from qmuvi.music import chromatic_middle_c
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

qmuvi.generate_qmuvi_music(circ, 
                           "example_music_only", 
                           noise_model = get_simple_noise_model(0.02, 0.05), 
                           rhythm = [(120, 60)]*8, 
                           phase_instruments = [[57], qmuvi.get_instrument_collection('tuned_perc')], 
                           note_map = chromatic_middle_c
                           )