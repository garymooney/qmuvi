import qmuvi
from qmuvi.quantum_simulation import get_simple_noise_model
import qiskit
from qiskit import QuantumCircuit

circ = QuantumCircuit(4)

circ.barrier()
circ.barrier()
circ.x(1)
circ.barrier()
circ.x(1)
circ.barrier()
circ.x(0)
circ.x(2)
circ.barrier()
circ.x(0)
circ.barrier()
circ.x(2)
circ.barrier()
circ.barrier()
circ.x(1)
circ.barrier()
circ.x(1)
circ.barrier()
circ.x(0)
circ.x(1)
circ.x(2)
circ.barrier()
circ.x(1)
circ.barrier()
circ.x(0)
circ.x(2)
circ.barrier()
circ.barrier()
circ.x(2)
circ.x(3)
circ.barrier()
circ.x(2)
circ.x(0)
circ.barrier()
circ.x(3)
circ.x(2)
circ.barrier()
circ.x(0)
circ.barrier()
circ.x(1)
circ.x(2)
circ.barrier()
circ.x(3)
circ.barrier()
circ.barrier()
circ.x(0)
circ.x(1)
circ.barrier()
circ.x(2)
circ.x(3)
circ.barrier()
circ.x(1)
circ.barrier()
circ.x(1)
circ.barrier()

time_list = [[80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],
             [80,0],[40,0],[120,0],[120,0],[120,0],[120,0],[120,0],
             [80,0],[40,0],[120,0],[120,0],[120,0],[240,0],]

qmuvi.generate_qmuvi(circ, 
                     "unhappy_bday", 
                     noise_model = get_simple_noise_model(0.2, 0.4), 
                     rhythm = time_list, 
                     phase_instruments = [get_instruments("pipe"), get_instruments("reed"), get_instruments("brass"), get_instruments("organ")], 
                     invert_colours = False, 
                     fps = 24, 
                     smooth_transitions = True
                     )