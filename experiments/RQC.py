import numpy as np
import random
import csv
import time

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator


# ============================================================
# 1. DEFINICIÓN DE COMPUERTAS
# ============================================================

def create_root_x_gate():
    """
    Análogo a RootXGate en C++.

    Matriz:
        1/sqrt(2) [[1, -i],
                   [-i, 1]]
    """

    mat = np.array([
        [1 / np.sqrt(2), -1j / np.sqrt(2)],
        [-1j / np.sqrt(2), 1 / np.sqrt(2)]
    ], dtype=complex)

    return UnitaryGate(mat, label="√X")


def create_root_y_gate():
    """
    Análogo a RootYGate en C++.

    Matriz:
        [[cos(pi/4), -sin(pi/4)],
         [sin(pi/4),  cos(pi/4)]]
    """

    c = 1 / np.sqrt(2)
    s = 1 / np.sqrt(2)

    mat = np.array([
        [c, -s],
        [s,  c]
    ], dtype=complex)

    return UnitaryGate(mat, label="√Y")


def create_root_w_gate():
    """
    Análogo a RootWGate en C++.

    En el C++:
        sqrt_half = 1 / sqrt(2)
        minus_sqrt_i = -(1+i)/sqrt(2)
        sqrt_minus_i = (1-i)/sqrt(2)

    Luego divide los términos fuera de la diagonal entre sqrt(2),
    por eso quedan:
        -(1+i)/2
         (1-i)/2
    """

    mat = np.array([
        [1 / np.sqrt(2), -(1 + 1j) / 2],
        [(1 - 1j) / 2,  1 / np.sqrt(2)]
    ], dtype=complex)

    return UnitaryGate(mat, label="√W")


def create_fsim_gate(theta=np.pi / 2, phi=np.pi / 6):
    """
    Análogo a fSimGate en C++.

    Base computacional:
        |00>, |01>, |10>, |11>

    Matriz:
        |00> -> |00>
        |01> -> cos(theta)|01> - i sin(theta)|10>
        |10> -> -i sin(theta)|01> + cos(theta)|10>
        |11> -> exp(-i phi)|11>
    """

    mat = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -1j * np.sin(theta), 0],
        [0, -1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, np.exp(-1j * phi)]
    ], dtype=complex)

    return UnitaryGate(mat, label="fSim")


# Gates globales, como en C++ donde se reutilizan funciones
GATE_X = create_root_x_gate()
GATE_Y = create_root_y_gate()
GATE_W = create_root_w_gate()

SINGLE_QUBIT_GATES = [GATE_X, GATE_Y, GATE_W]


# ============================================================
# 2. CAPA DE COMPUERTAS DE UN QUBIT
# ============================================================

def choose_single_qubit_gate(prev_gate=-1):
    """
    Análogo a singleQubitGate en C++.

    Escoge aleatoriamente entre:
        0 -> √X
        1 -> √Y
        2 -> √W

    Regla importante:
        No repetir la misma gate que se usó en la capa anterior
        sobre el mismo qubit.
    """

    options = [0, 1, 2]

    if prev_gate != -1:
        options.remove(prev_gate)

    choice = random.choice(options)

    return SINGLE_QUBIT_GATES[choice], choice


def layer1_qiskit(qc, num_qubits, prev_gates=None):
    """
    Análogo a layer1Cpp.

    Aplica una gate aleatoria de un qubit a cada qubit.
    Devuelve la lista current_gates para recordar qué gate se usó.
    """

    if prev_gates is None:
        prev_gates = [-1] * num_qubits

    current_gates = [-1] * num_qubits

    for q in range(num_qubits):
        gate, choice = choose_single_qubit_gate(prev_gates[q])

        qc.append(gate, [q])
        current_gates[q] = choice

    return qc, current_gates


# ============================================================
# 3. CAPA DE COMPUERTAS FSIM ENTRE VECINOS
# ============================================================

def layer2_qiskit(qc, num_qubits, start):
    """
    Análogo a layer2Cpp.

    Aplica fSim entre pares vecinos.

    En C++ los índices empiezan en 1:
        start = 1 -> pares (1,2), (3,4), ...
        start = 2 -> pares (2,3), (4,5), ...

    En Python/Qiskit los índices empiezan en 0:
        start = 0 -> pares (0,1), (2,3), ...
        start = 1 -> pares (1,2), (3,4), ...
    """

    fsim_gate = create_fsim_gate()

    for q in range(start, num_qubits - 1, 2):
        qc.append(fsim_gate, [q, q + 1])

    return qc


# ============================================================
# 4. RANDOM QUANTUM CIRCUIT COMPLETO
# ============================================================

def apply_random_circuit(num_qubits, layers):
    """
    Análogo a applyRandomCircuit en C++.

    Por cada layer:
        1. Aplica gates aleatorias de un qubit.
        2. Aplica fSim en patrón brickwork alternado.

    Patrón:
        layer par:   (0,1), (2,3), (4,5), ...
        layer impar: (1,2), (3,4), (5,6), ...
    """

    qc = QuantumCircuit(num_qubits)

    prev_gates = [-1] * num_qubits

    for layer_idx in range(layers):
        qc, prev_gates = layer1_qiskit(qc, num_qubits, prev_gates)

        start = 0 if layer_idx % 2 == 0 else 1

        qc = layer2_qiskit(qc, num_qubits, start)

    return qc


# ============================================================
# 5. SIMULACIÓN / BENCHMARK
# ============================================================

def run_rqc_once(num_qubits, layers, method="matrix_product_state", shots=1):
    """
    Corre una vez el Random Quantum Circuit y mide el tiempo.

    Usamos AerSimulator(method='matrix_product_state') porque es lo más
    parecido al espíritu del C++ con ITensor/MPS.

    Importante:
        Para muchos qubits, NO uses statevector.
        Statevector escala como 2^N y explota rápido.
    """

    qc = apply_random_circuit(num_qubits, layers)

    # Medición final para obligar al simulador a ejecutar el circuito.
    # El C++ no mide, solo evoluciona el MPS, pero en Qiskit esto ayuda
    # a tener una ejecución concreta y resultados verificables.
    qc.measure_all()

    simulator = AerSimulator(method=method)

    tqc = transpile(qc, simulator)

    start_time = time.perf_counter()

    result = simulator.run(tqc, shots=shots).result()

    end_time = time.perf_counter()

    elapsed = end_time - start_time

    return elapsed, result


def benchmark_qubits(
    min_qubits=20,
    max_qubits=60,
    layers=10,
    runs=1,
    output_file="60Q_10L_qiskit.csv",
    method="matrix_product_state"
):
    """
    Análogo al main activo del C++:

        for numQubits = 20 to 60:
            correr RQC
            medir tiempo
            guardar CSV

    CSV:
        num_qubits, avg_time_seconds, run_1, run_2, ...
    """

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        header = ["num_qubits", "avg_time_seconds"]

        for i in range(1, runs + 1):
            header.append(f"run_{i}")

        writer.writerow(header)

        for num_qubits in range(min_qubits, max_qubits + 1):
            times = []

            for run in range(runs):
                elapsed, _ = run_rqc_once(
                    num_qubits=num_qubits,
                    layers=layers,
                    method=method,
                    shots=1
                )

                times.append(elapsed)

                print(
                    f"Tiempo de evolución "
                    f"(N={num_qubits}, run={run + 1}): {elapsed:.6f} s"
                )

            avg_time = sum(times) / len(times)

            row = [num_qubits, avg_time] + times
            writer.writerow(row)

            print(
                f">>> Qubits: {num_qubits} "
                f"| Tiempo promedio: {avg_time:.6f} s"
            )

    print(f"Datos guardados en '{output_file}'")


def benchmark_layers(
    num_qubits=21,
    min_layers=1,
    max_layers=15,
    runs=1,
    output_file="15L_21Q_qiskit.csv",
    method="matrix_product_state"
):
    """
    Análogo al main comentado de LAYERS en el C++.

    Varía el número de capas dejando fijo el número de qubits.
    """

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        header = ["layers", "num_qubits", "avg_time_seconds"]

        for i in range(1, runs + 1):
            header.append(f"run_{i}")

        writer.writerow(header)

        for layers in range(min_layers, max_layers + 1):
            times = []

            for run in range(runs):
                elapsed, _ = run_rqc_once(
                    num_qubits=num_qubits,
                    layers=layers,
                    method=method,
                    shots=1
                )

                times.append(elapsed)

                print(
                    f"Tiempo de evolución "
                    f"(L={layers}, N={num_qubits}, run={run + 1}): "
                    f"{elapsed:.6f} s"
                )

            avg_time = sum(times) / len(times)

            row = [layers, num_qubits, avg_time] + times
            writer.writerow(row)

            print(
                f">>> Layers: {layers} "
                f"| Qubits: {num_qubits} "
                f"| Tiempo promedio: {avg_time:.6f} s"
            )

    print(f"Datos guardados en '{output_file}'")


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # Versión análoga al main activo del C++:
    # N = 20 hasta 60, layers = 10.
    #
    # OJO:
    # En Qiskit puede demorar bastante para N grande.
    # Primero prueba con max_qubits=25 o 30.

    benchmark_qubits(
        min_qubits=20,
        max_qubits=30,
        layers=10,
        runs=1,
        output_file="rqc_qiskit_20Q_30Q_10L.csv",
        method="matrix_product_state"
    )

    # Si quieres replicar más fielmente el rango del C++:
    #
    # benchmark_qubits(
    #     min_qubits=20,
    #     max_qubits=60,
    #     layers=10,
    #     runs=1,
    #     output_file="60Q_10L_qiskit.csv",
    #     method="matrix_product_state"
    # )

    # Si quieres el benchmark por capas:
    #
    # benchmark_layers(
    #     num_qubits=21,
    #     min_layers=1,
    #     max_layers=15,
    #     runs=1,
    #     output_file="15L_21Q_qiskit.csv",
    #     method="matrix_product_state"
    # )