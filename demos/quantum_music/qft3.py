import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
import qiskit
from qiskit import QuantumCircuit
from numpy import pi

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

def Cmaj_arpeggio(n):
    if (n+1)%3==0:
        return (4*n-1)+60
    return (4*n)+60
time_list = [[200,40]]*7+[[960,0]]

instruments = []
instruments.append(qmuvi.get_instruments("synth_lead"))
instruments.append(qmuvi.get_instruments("ethnic"))
instruments.append(qmuvi.get_instruments("ensemble"))
instruments.append(qmuvi.get_instruments("synth_pad"))
instruments.append(qmuvi.get_instruments("percussive"))
instruments.append(qmuvi.get_instruments("sound_effects"))

qmuvi.generate_qmuvi(circ, 
                     "qft3", 
                     noise_model = get_simple_noise_model(0.2, 0.4), 
                     rhythm = time_list, 
                     phase_instruments = instruments, 
                     note_map = Cmaj_arpeggio,
                     invert_colours = True, 
                     fps = 60, 
                     smooth_transitions = True
                     )