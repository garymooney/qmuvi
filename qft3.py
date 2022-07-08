import quantum_music
from quantum_music import make_music_video, get_instruments
import qiskit
from qiskit import QuantumCircuit
from numpy import pi

circ = QuantumCircuit(3)
circ.x(0)
circ.x(1)
circ.x(2)
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
circ.swap(0,2)
circ.barrier()

def Cmaj_arpeggio(n):
    if (n+1)%3==0:
        return (4*n-1)+60
    return (4*n)+60
time_list = [[200,40]]*7+[[960,0]]

make_music_video(circ, "qft", time_list, 0.05, 0.1, [[57], get_instruments('windband')], Cmaj_arpeggio)