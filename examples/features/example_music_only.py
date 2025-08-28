import qiskit
from qiskit import QuantumCircuit

# Add the qmuvi path so that we can import qmuvi (if you have installed qmuvi, you can skip this step)
import sys
sys.path.append(r"../..")
import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
from qmuvi.musical_processing import note_map_chromatic_middle_c

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
                           instruments = [[57], qmuvi.get_instrument_collection('tuned_perc')],
                           note_map = note_map_chromatic_middle_c
                           )
