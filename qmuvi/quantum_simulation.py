"""This module contains functions relating to the simulation and sampliing of Qiskit circuits for qMuVi."""

from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit_aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import NoiseModel, depolarizing_error


def get_simple_noise_model(gate_error_rate_1q: float = 0.0, gate_error_rate_cnot: float = 0.0) -> NoiseModel:
    """Generates a simple noise model with depolarising errors on single qubit gates and CNOT gates.

    Parameters
    ----------
        gate_error_rate_1q
            The depolarising error to be applied to single qubit gates.
        gate_error_rate_cnot
            The depolarising error to be applied to CNOT gates.

    Returns
    -------
        The noise model to be applied to the simulation.
    """
    noise_model = NoiseModel()

    gate_error_1q_depolarising = depolarizing_error(gate_error_rate_1q, 1)
    noise_model.add_all_qubit_quantum_error(gate_error_1q_depolarising, ["u1", "u2", "u3"])

    gate_error_cnot_depolarising = depolarizing_error(gate_error_rate_cnot, 2)
    noise_model.add_all_qubit_quantum_error(gate_error_cnot_depolarising, ["cx"])

    return noise_model


def sample_circuit_barriers(quantum_circuit: QuantumCircuit, noise_model: Optional[NoiseModel] = None) -> List[np.ndarray]:
    """Saves the state of the quantum circuit at each barrier in the simulation as a density matrix.

    Parameters
    ----------
        quantum_circuit
            The quantum circuit to be simulated.
        noise_model
            The noise model to be applied to the simulation.

    Returns
    -------
        density_matrices: A list of the sampled density matrices as 2d complex numpy arrays.
    """

    if noise_model is None:
        simulator = AerSimulator()
    else:
        simulator = AerSimulator(noise_model=noise_model)

    dag = circuit_to_dag(quantum_circuit)
    qubit_count = len(dag.qubits)
    new_quantum_circuit = QuantumCircuit(qubit_count)

    barrier_iter = 0
    for node in dag.topological_op_nodes():
        if node.name == "barrier":
            new_quantum_circuit.save_density_matrix(label=f"rho{barrier_iter}")  # type: ignore
            barrier_iter += 1
        if node.name != "measure":
            new_quantum_circuit.append(node.op, node.qargs, node.cargs)
    barrier_count = barrier_iter

    transpiled_quantum_circuit = transpile(new_quantum_circuit, simulator)

    result = simulator.run(transpiled_quantum_circuit).result()

    density_matrices = []
    for i in range(barrier_count):
        density_matrices.append(result.data(0)[f"rho{i}"])

    # dag = circuit_to_dag(quantum_circuit)
    # qubit_count = len(dag.qubits)
    # new_quantum_circuit = QuantumCircuit(qubit_count)
    # quantum_circuits = []

    # barrier_iter = 0
    # for node in dag.topological_op_nodes():
    #     if node.name == "barrier":
    #         quantum_circuits.append(new_quantum_circuit.copy())
    #         #new_quantum_circuit.save_density_matrix(label=f"rho{barrier_iter}")
    #         barrier_iter += 1
    #     if node.name != "measure":
    #         new_quantum_circuit.append(node.op, node.qargs, node.cargs)
    # barrier_count = barrier_iter

    # density_matrices = []
    # for qc in quantum_circuits:
    #     transpiled_quantum_circuit = transpile(qc, simulator)

    #     result : qiskit.Result = simulator.run(transpiled_quantum_circuit).result()
    #     density_matrices.append(result.data(0)["density_matrix"])

    # density_matrices = []
    # for i in range(barrier_count):
    #     density_matrices.append(result.data(0)[f"rho{i}"])

    return density_matrices
