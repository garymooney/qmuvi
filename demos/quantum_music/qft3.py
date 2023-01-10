import quantum_music
from quantum_music import make_music_video, get_instruments, get_depolarising_noise
import qiskit
from qiskit import QuantumCircuit
from numpy import pi

circ = QuantumCircuit(3)

#circ.h(0)
#circ.x(1)


#circ.barrier()
#for i in range(31):
#    circ.crz(pi/16, 0,1)
#
#    circ.barrier()
#
#rhythm = [[240,0]]*32

circ.x(0)
circ.x(1)
#circ.x(2)

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
instruments.append(get_instruments("synth_lead"))
instruments.append(get_instruments("ethnic"))
instruments.append(get_instruments("ensemble"))
instruments.append(get_instruments("synth_pad"))
instruments.append(get_instruments("percussive"))
instruments.append(get_instruments("sound_effects"))

make_music_video(circ, "qft", time_list, get_depolarising_noise(0.2, 0.4), instruments, Cmaj_arpeggio, invert_colours = True, fps = 60, smooth_transitions=True)