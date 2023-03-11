import qiskit
from qiskit import QuantumCircuit
from math import pi

# Add the qmuvi path so that we can import qmuvi (if you have installed qmuvi, you can skip this step)
import sys
sys.path.append(r"../..")
import qmuvi


circ = QuantumCircuit(3)

circ.x(0)
circ.x(1)
circ.barrier() # qMuVi will play the state as a sound when it encounters a barrier gate.
circ.h(0)
circ.barrier()
circ.crz(pi/2, 1, 0)
circ.barrier()
circ.crz(pi/4, 2, 0)
circ.barrier()
circ.h(1)
circ.barrier()
circ.crz(pi/2, 2, 1)
circ.barrier()
circ.h(2)
circ.barrier()
circ.barrier()

qmuvi.generate_qmuvi(circ, "simple_qft3")