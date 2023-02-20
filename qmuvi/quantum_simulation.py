# Methods relating to the simulation and sampliing of QISKIT circuits for QMUVI

import qiskit
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Kraus, SuperOp
from qiskit import Aer

from typing import List
import numpy as np

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
)

from qiskit.converters import circuit_to_dag

def get_simple_noise_model(single_qubit_gater_error: float, cnot_gate_error: float) -> NoiseModel:
    ''' Generates a simple noise model with depolarising errors on single qubit gates and CNOT gates.
    
    Parameters
    ----------
        single_qubit_gater_error
            The depolarising error to be applied to single qubit gates.
        cnot_gate_error
            The depolarising error to be applied to CNOT gates.

    Returns
    -------
        The noise model to be applied to the simulation.
    '''
    
    noise_model = NoiseModel()
    single_qubit_dep_error = depolarizing_error(single_qubit_gater_error, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_dep_error, ['u1', 'u2', 'u3'])
    cnot_gate_dep_error = depolarizing_error(cnot_gate_error, 2)
    noise_model.add_all_qubit_quantum_error(cnot_gate_dep_error, ['cx'])
    return noise_model

def sample_circuit_barriers(quantum_circuit: QuantumCircuit, noise_model: NoiseModel = None) -> List[np.ndarray]:
    '''Saves the state of the quantum circuit at each barrier in the simulation as a density matrix.
    
    Parameters
    ----------
        quantum_circuit
            The quantum circuit to be simulated.
        noise_model
            The noise model to be applied to the simulation.

    Returns
    -------
        density_matrices: A list of the sampled density matrices as 2d complex numpy arrays.
    '''

    if noise_model == None:
        simulator = AerSimulator()
    else:
        simulator = AerSimulator(noise_model = noise_model)

    dag = circuit_to_dag(quantum_circuit)
    qubit_count = len(dag.qubits)
    new_quantum_circuit = QuantumCircuit(qubit_count)

    barrier_iter = 0
    for node in dag.topological_op_nodes():
        if node.name == "barrier":
            new_quantum_circuit.save_density_matrix(label=f'rho{barrier_iter}')
            barrier_iter += 1
        if node.name != "measure":
            new_quantum_circuit.append(node.op, node.qargs, node.cargs)
    barrier_count = barrier_iter

    transpiled_quantum_circuit = transpile(new_quantum_circuit, simulator)
    
    result = simulator.run(transpiled_quantum_circuit).result()

    density_matrices = []
    for i in range(barrier_count):
        density_matrices.append(result.data(0)[f'rho{i}'])
        
    return density_matrices