import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, QFT
from gates import single_q_gates


def create_fsim_gate(theta=np.pi / 2, phi=np.pi / 6):
    """
    Create the two-qubit fSim gate used to couple nearest-neighbor qubits.
    """
    mat = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -1j * np.sin(theta), 0],
        [0, -1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, np.exp(-1j * phi)]
    ], dtype=complex)
    return UnitaryGate(mat, label="fSim")


def apply_random_circuit(num_qubits, layers, seed=42):
    """
    Base RQC without QFT.

    The local seed makes layer comparisons prefix-consistent:
    apply_random_circuit(n, 1, seed) is the first layer of
    apply_random_circuit(n, 2, seed).
    """
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits)
    fsim_gate = create_fsim_gate()
    prev_gates = [-1] * num_qubits

    for layer_idx in range(layers):
        for q in range(num_qubits):
            options = list(range(len(single_q_gates)))
            if prev_gates[q] != -1:
                options.remove(prev_gates[q])
            choice = rng.choice(options)
            qc.append(single_q_gates[choice], [q])
            prev_gates[q] = choice

        start = 0 if layer_idx % 2 == 0 else 1
        for q in range(start, num_qubits - 1, 2):
            qc.append(fsim_gate, [q, q + 1])

    return qc


def apply_random_circuit_with_qft(num_qubits, layers, seed=42, do_swaps=True):
    """
    Same RQC, then QFT. Use only in the QFT condition.
    """
    qc = apply_random_circuit(num_qubits=num_qubits, layers=layers, seed=seed)
    qc.append(QFT(num_qubits, do_swaps=do_swaps), range(num_qubits))
    return qc
