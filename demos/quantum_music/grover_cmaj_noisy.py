import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
from qmuvi.musical_processing import note_map_c_major
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

instruments = []
#instruments.append([57])
instruments.append(qmuvi.get_instrument_collection("windband"))
instruments.append(qmuvi.get_instrument_collection("windband"))
instruments.append(qmuvi.get_instrument_collection("ethnic"))
instruments.append(qmuvi.get_instrument_collection("ethnic"))
instruments.append(qmuvi.get_instrument_collection("percussive"))
instruments.append(qmuvi.get_instrument_collection("sound_effects"))
#make_music_midi(circ, "grover_Cmaj_noisy", time_list, get_simple_depolarising_noise_model(0.1, 0.2), instruments)
#convert_midi_to_wav_vlc("grover_Cmaj_noisy/grover_Cmaj_noisy")
#convert_midi_to_wav_timidity("grover_Cmaj_noisy/grover_Cmaj_noisy")

qmuvi.generate_qmuvi(circ, 
                     "grover_Cmaj_noisy", 
                     noise_model = get_simple_noise_model(0.01, 0.02), 
                     rhythm = time_list, 
                     instruments = instruments, 
                     note_map = note_map_c_major,
                     invert_colours = True, 
                     fps =  60, 
                     smooth_transitions = False
                     )