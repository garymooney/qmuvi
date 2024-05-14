import qiskit.quantum_info
from qiskit import QuantumCircuit
import math
from typing import List


from qmuvi.quantum_simulation import get_simple_noise_model, sample_circuit_barriers

def test_get_simple_noise_model():

    noise_model = get_simple_noise_model(0.01, 0.1)
    assert noise_model is not None


def test_sample_circuit_barriers():

    circ = QuantumCircuit(2)
    circ.barrier()
    circ.h(0)
    circ.barrier()
    circ.cx(0, 1)
    circ.barrier()

    density_matrices_pure: List[qiskit.quantum_info.DensityMatrix] = sample_circuit_barriers(circ)
    assert len(density_matrices_pure) == 3
    assert density_matrices_pure[0].data.shape == (4, 4)
    assert math.isclose(density_matrices_pure[0].data[0][0].real, 1)
    assert math.isclose(density_matrices_pure[2].data[0][0].real, 0.5) and math.isclose(density_matrices_pure[2].data[3][3].real, 0.5)

