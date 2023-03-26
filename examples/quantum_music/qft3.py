import qiskit
from qiskit import QuantumCircuit
from numpy import pi

# Add the qmuvi path so that we can import qmuvi (if you have installed qmuvi, you can skip this step)
import sys
sys.path.append(r"../..")
import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
from qmuvi.musical_processing import note_map_c_major_arpeggio


circ = QuantumCircuit(3)

circ.x(0)
circ.x(1)
circ.barrier()
circ.h(0)
circ.barrier()
circ.crz(pi/2,1,0)
circ.barrier()
circ.crz(pi/4,2,0)
circ.barrier()
circ.h(1)
circ.barrier()
circ.crz(pi/2,2,1)
circ.barrier()
circ.h(2)
circ.barrier()
circ.barrier()

time_list = [[200,40]]*7+[[960,0]]

instruments = []
instruments.append(qmuvi.get_instrument_collection("synth_lead"))
instruments.append(qmuvi.get_instrument_collection("ethnic"))
instruments.append(qmuvi.get_instrument_collection("ensemble"))
instruments.append(qmuvi.get_instrument_collection("synth_pad"))
instruments.append(qmuvi.get_instrument_collection("percussive"))
instruments.append(qmuvi.get_instrument_collection("sound_effects"))

qmuvi.generate_qmuvi(circ, 
                     "qft3", 
                     noise_model = get_simple_noise_model(0.2, 0.4), 
                     rhythm = time_list, 
                     instruments = instruments, 
                     note_map = note_map_c_major_arpeggio,
                     invert_colours = True, 
                     fps = 24, 
                     smooth_transitions = True
                     )