import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from gates import single_q_gates



def create_fsim_gate(theta=np.pi / 2, phi=np.pi / 6):
    """
    Crea una compuerta fSim de dos qubits.

    Esta compuerta puede generar interacción entre qubits vecinos.
    """

    mat = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -1j * np.sin(theta), 0],
        [0, -1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, np.exp(-1j * phi)]
    ])

    return UnitaryGate(mat, label="fSim")

def apply_random_circuit(num_qubits, layers):
    """
    Construye un circuito cuántico aleatorio completo.

    En cada capa:
        1. Aplica una compuerta aleatoria de un qubit a cada qubit.
        2. Aplica compuertas fSim entre qubits vecinos.

    Esta función devuelve solo el circuito final.
    """

    qc = QuantumCircuit(num_qubits)

    fsim_gate = create_fsim_gate()
    prev_gates = [-1] * num_qubits

    for layer_idx in range(layers):

        # Compuertas aleatorias de un qubit
        for i in range(num_qubits):
            options = list(range(len(single_q_gates)))

            if prev_gates[i] != -1:
                options.remove(prev_gates[i])

            choice = random.choice(options)

            qc.append(single_q_gates[choice], [i])
            prev_gates[i] = choice

        # Compuertas fSim entre qubits vecinos
        start = 0 if layer_idx % 2 == 0 else 1

        for i in range(start, num_qubits - 1, 2):
            qc.append(fsim_gate, [i, i + 1])

    return qc