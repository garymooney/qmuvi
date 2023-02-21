import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
from qmuvi.musical_processing import note_map_chromatic_middle_c
import qiskit
from qiskit import QuantumCircuit

repeats = 2

c = QuantumCircuit(5)

for i in range(repeats):
    if i==0:
        c.x(2)
        c.x(3)
    c.barrier()
    c.barrier()
    c.x(0)
    c.x(3)
    c.h(3)
    c.cx(3,0)
    c.cx(3,1)
    c.barrier()
    c.cx(3,1)
    c.barrier()
    c.cx(3,4)
    c.cx(4,3)
    c.cx(4,2)
    c.cx(4,0)
    c.barrier()
    c.x(0)
    c.x(2)
    c.cx(4,2)
    c.barrier()
    c.cx(4,3)
    c.cx(4,2)
    c.cx(3,4)
    c.barrier()
    c.barrier()
    c.cx(2,1)
    c.barrier()
    c.cx(2,1)
    c.barrier()
    c.cx(3,4)
    c.cx(2,1)
    c.cx(2,0)
    c.cx(4,3)
    c.cx(1,2)
    c.barrier()
    c.x(0)
    c.x(2)
    c.cx(4,2)
    c.cx(1,0)
    c.cx(4,1)
    c.barrier()
    c.cx(4,0)
    c.cx(4,2)
    c.cx(4,3)
    c.cx(3,4)
    c.barrier()
    c.barrier()
    c.cx(3,4)
    c.cx(3,2)
    c.barrier()
    c.cx(4,0)
    c.cx(4,2)
    c.cx(4,3)
    c.barrier()
    c.x(2)
    c.x(3)
    c.cx(4,3)
    c.barrier()
    c.x(0)
    c.x(1)
    c.cx(4,1)
    c.barrier()
    c.cx(4,3)
    c.cx(4,2)
    c.cx(4,1)
    c.cx(2,4)
    c.barrier()
    c.x(0)
    c.cx(2,4)
    c.cx(2,3)
    c.cx(2,0)
    c.barrier()
    c.barrier()
    c.x(0)
    c.x(1)
    c.x(2)
    c.cx(0,2)
    c.barrier()
    c.cx(0,2)
    c.barrier()
    c.x(2)
    c.x(3)
    c.cx(4,3)
    c.cx(2,1)
    c.cx(1,2)
    c.barrier()
    c.x(0)
    c.x(2)
    c.cx(4,2)
    c.cx(4,1)
    c.cx(4,0)
    c.barrier()
    if i+1 != repeats:
        c.cx(4,2)
        c.h(4)
        c.x(3)
        c.x(0)


time_list = [[80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
             [80,0],[40,0],[120,0],[120,0],[120,0],[120,0],[120,0],
             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],]*repeats

time_list[len(time_list)-1] = [360,0]
# slow down by factor s
s = 2
time_list = [[s*t for t in tt] for tt in time_list]

qmuvi.generate_qmuvi(c, 
                     "happy_bday_bassline", 
                     noise_model = get_simple_noise_model(0.025, 0.045), 
                     rhythm = time_list, 
                     instruments = [[75], qmuvi.get_instrument_collection("pipe"), qmuvi.get_instrument_collection("pipe"), qmuvi.get_instrument_collection("reed"), qmuvi.get_instrument_collection("brass"), qmuvi.get_instrument_collection("organ")], 
                     note_map = note_map_chromatic_middle_c,
                     invert_colours = False, 
                     fps = 60, 
                     smooth_transitions = True
                     )