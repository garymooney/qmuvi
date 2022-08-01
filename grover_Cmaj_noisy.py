import quantum_music
from quantum_music import make_music_video, get_instruments, get_depolarising_noise
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
instruments.append(get_instruments("windband"))
instruments.append(get_instruments("windband"))
instruments.append(get_instruments("ethnic"))
instruments.append(get_instruments("ethnic"))
instruments.append(get_instruments("percussive"))
instruments.append(get_instruments("sound_effects"))
#make_music_midi(circ, "grover_Cmaj_noisy", time_list, get_simple_depolarising_noise_model(0.1, 0.2), instruments)
#convert_midi_to_wav_vlc("grover_Cmaj_noisy/grover_Cmaj_noisy")
#convert_midi_to_wav_timidity("grover_Cmaj_noisy/grover_Cmaj_noisy")
make_music_video(circ, "grover_Cmaj_noisy", time_list, get_depolarising_noise(0.01, 0.02), instruments, invert_colours = True, fps = 60, smooth_transitions=False)