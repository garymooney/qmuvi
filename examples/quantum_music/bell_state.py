import qmuvi
from qiskit import QuantumCircuit

circ = QuantumCircuit(2)
circ.barrier()
circ.h(0)
circ.barrier()
circ.cx(0, 1)
circ.barrier()

qmuvi.generate_qmuvi(circ, "bell_state")