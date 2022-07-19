import quantum_music
from quantum_music import make_music_video, get_instruments, make_music_midi
import qiskit
from qiskit import QuantumCircuit
from numpy import pi

circ = QuantumCircuit(2)

circ.h(0)
circ.x(1)


circ.barrier()
for i in range(31):
    circ.crz(pi/16, 0,1)

    circ.barrier()

rhythm = [[240,0]]*32

#circ.x(1)
#circ.x(2)
#circ.barrier()
#circ.h(0)
#circ.barrier()
#circ.crz(pi/2,1,0)
#circ.barrier()
#circ.crz(pi/4,2,0)
#circ.barrier()
#circ.h(1)
#circ.barrier()
#circ.crz(pi/2,2,1)
#circ.barrier()
#circ.h(2)
#circ.barrier()
#circ.swap(0,2)
#circ.barrier()

def Cmaj_arpeggio(n):
    if (n+1)%3==0:
        return (4*n-1)+60
    return (4*n)+60
time_list = [[200,40]]*7+[[960,0]]

make_music_midi(circ, "qft_midi", rhythm, 0.0, 0, [[57]], Cmaj_arpeggio)