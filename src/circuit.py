import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, QFT
from gates import single_q_gates

def create_fsim_gate(theta=np.pi / 2, phi=np.pi / 6):
    mat = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -1j * np.sin(theta), 0],
        [0, -1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, np.exp(-1j * phi)]
    ])
    return UnitaryGate(mat, label="fSim")

def apply_random_circuit(num_qubits, layers):
    qc = QuantumCircuit(num_qubits)
    fsim_gate = create_fsim_gate()
    prev_gates = [-1] * num_qubits

    for layer_idx in range(layers):
        for i in range(num_qubits):
            options = list(range(len(single_q_gates)))
            if prev_gates[i] != -1:
                options.remove(prev_gates[i])
            choice = random.choice(options)
            qc.append(single_q_gates[choice], [i])
            prev_gates[i] = choice

        start = 0 if layer_idx % 2 == 0 else 1
        for i in range(start, num_qubits - 1, 2):
            qc.append(fsim_gate, [i, i + 1])
    return qc

def apply_random_circuit_with_qft(num_qubits, layers):
    qc = apply_random_circuit(num_qubits, layers)
    qc.append(QFT(num_qubits), range(num_qubits))
    return qc