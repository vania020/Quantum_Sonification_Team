import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from gates import single_q_gates
from circuit import create_fsim_gate



# ============================================================
# 4. EXTRAER STATEVECTOR DEL CIRCUITO
# ============================================================
def get_statevector_from_circuit(qc, use_mps=False):
    """
    Simula el circuito y devuelve el statevector completo.
    """

    qc_temp = qc.copy()
    qc_temp.save_statevector()

    # Descompone compuertas personalizadas para que Aer las entienda mejor.
    qc_temp = qc_temp.decompose()

    if use_mps:
        simulator = AerSimulator(method="matrix_product_state")
    else:
        simulator = AerSimulator(method="statevector")

    result = simulator.run(qc_temp).result()
    state = np.asarray(result.get_statevector(qc_temp))

    return state

# ============================================================
# 5. CALCULAR THETA, PHI, r Y 1-r POR QUBIT
# ============================================================

def get_bloch_data_from_statevector(state, num_qubits, eps=1e-12):
    """
    Calcula el vector de Bloch local de cada qubit.

    Para cada qubit devuelve:
        x, y, z       : componentes del vector de Bloch
        theta         : ángulo polar
        phi           : ángulo azimutal
        r             : longitud del vector de Bloch
        mixedness     : 1 - r
    """

    dim = len(state)
    indices = np.arange(dim)

    x_values = []
    y_values = []
    z_values = []

    theta_values = []
    phi_values = []
    r_values = []
    mixedness_values = []

    for qubit in range(num_qubits):

        # En Qiskit, el qubit 0 corresponde al bit menos significativo.
        mask = 1 << qubit

        # Índices donde el qubit elegido está en |0>
        indices_0 = indices[(indices & mask) == 0]

        # Índices correspondientes donde ese mismo qubit está en |1>
        indices_1 = indices_0 | mask

        amp_0 = state[indices_0]
        amp_1 = state[indices_1]

        # Probabilidades locales
        p0 = np.sum(np.abs(amp_0) ** 2)
        p1 = np.sum(np.abs(amp_1) ** 2)

        # Coherencia local entre |0> y |1>
        coherence = np.vdot(amp_0, amp_1)

        # Componentes del vector de Bloch
        x = 2 * np.real(coherence)
        y = 2 * np.imag(coherence)
        z = p0 - p1

        # Longitud del vector de Bloch
        r = np.sqrt(x**2 + y**2 + z**2)

        # Evita errores numéricos tipo 1.00000000001
        r = np.clip(r, 0.0, 1.0)

        if r > eps:
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))
            phi = np.arctan2(y, x)

            if phi < 0:
                phi += 2 * np.pi
        else:
            # Si r ≈ 0, el vector está en el centro de Bloch.
            # Entonces theta y phi no tienen dirección física clara.
            theta = 0.0
            phi = 0.0

        mixedness = 1.0 - r

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

        theta_values.append(theta)
        phi_values.append(phi)
        r_values.append(r)
        mixedness_values.append(mixedness)

    bloch_data = {
        "x": np.array(x_values),
        "y": np.array(y_values),
        "z": np.array(z_values),
        "theta": np.array(theta_values),
        "phi": np.array(phi_values),
        "r": np.array(r_values),
        "mixedness": np.array(mixedness_values)
    }

    return bloch_data


def get_bloch_data_from_circuit(qc, use_mps=False):
    """
    Simula un circuito y devuelve theta, phi, r y 1-r para cada qubit.
    """

    state = get_statevector_from_circuit(qc, use_mps=use_mps)
    num_qubits = qc.num_qubits

    bloch_data = get_bloch_data_from_statevector(state, num_qubits)

    return bloch_data


# ============================================================
# 6. GENERAR DATOS DE BLOCH POR CAPA
# ============================================================

def generate_bloch_audio_data_by_layer(num_qubits, layers, use_mps=False):
    """
    Construye el circuito capa por capa.

    Después de cada capa calcula:
        theta
        phi
        r
        mixedness = 1 - r

    para cada qubit.
    """

    qc = QuantumCircuit(num_qubits)

    fsim_gate = create_fsim_gate()
    prev_gates = [-1] * num_qubits

    all_layers_data = []

    # Capa 0: estado inicial |000...0>
    bloch_data = get_bloch_data_from_circuit(qc, use_mps=use_mps)
    all_layers_data.append((0, bloch_data))

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

        # Datos de Bloch después de la capa completa
        bloch_data = get_bloch_data_from_circuit(qc, use_mps=use_mps)
        all_layers_data.append((layer_idx + 1, bloch_data))

    return all_layers_data