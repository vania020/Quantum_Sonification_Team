import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
import scipy.io.wavfile as wavfile


# --- 1. DEFINICIÓN DE COMPUERTAS ---
sqrt_X_mat = np.array([
    [1/np.sqrt(2), -1j/np.sqrt(2)],
    [-1j/np.sqrt(2), 1/np.sqrt(2)]
])

sqrt_Y_mat = np.array([
    [1/np.sqrt(2), -1/np.sqrt(2)],
    [1/np.sqrt(2), 1/np.sqrt(2)]
])

sqrt_W_mat = np.array([
    [1/np.sqrt(2), -(1+1j)/2],
    [(1-1j)/2, 1/np.sqrt(2)]
])

gate_X = UnitaryGate(sqrt_X_mat, label="√X")
gate_Y = UnitaryGate(sqrt_Y_mat, label="√Y")
gate_W = UnitaryGate(sqrt_W_mat, label="√W")

single_q_gates = [gate_X, gate_Y, gate_W]


def create_fsim_gate(theta=np.pi/2, phi=np.pi/6):
    mat = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -1j*np.sin(theta), 0],
        [0, -1j*np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, np.exp(-1j*phi)]
    ])

    return UnitaryGate(mat, label="fSim")


# --- 2. CONSTRUCTOR DEL CIRCUITO ALEATORIO ---
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


# --- 3. EXTRAER PROBABILIDADES Y FASES DE UN CIRCUITO ---
def get_probabilities_phases_from_circuit(qc, use_mps=False):
    qc_temp = qc.copy()
    qc_temp.save_statevector()

    if use_mps:
        simulator = AerSimulator(method="matrix_product_state")
    else:
        simulator = AerSimulator(method="statevector")

    result = simulator.run(qc_temp).result()
    state = np.asarray(result.get_statevector())

    probabilities = np.abs(state) ** 2
    phases = np.angle(state)

    return probabilities, phases


# --- 4. GENERADOR DE DATOS PARA SONIFICACIÓN POR CAPAS ---
def generate_quantum_audio_data_by_layer(num_qubits, layers, use_mps=False):
    qc = QuantumCircuit(num_qubits)

    fsim_gate = create_fsim_gate()
    prev_gates = [-1] * num_qubits

    all_layers_data = []

    # Capa 0: estado inicial |000...0>
    probabilities, phases = get_probabilities_phases_from_circuit(qc, use_mps)
    all_layers_data.append((0, probabilities, phases))

    for layer_idx in range(layers):

        # Gates aleatorias de un qubit
        for i in range(num_qubits):
            options = list(range(len(single_q_gates)))

            if prev_gates[i] != -1:
                options.remove(prev_gates[i])

            choice = random.choice(options)

            qc.append(single_q_gates[choice], [i])
            prev_gates[i] = choice

        # Gates fSim de dos qubits
        start = 0 if layer_idx % 2 == 0 else 1

        for i in range(start, num_qubits - 1, 2):
            qc.append(fsim_gate, [i, i + 1])

        # Estado después de la capa completa: gates 1-qubit + fSim
        probabilities, phases = get_probabilities_phases_from_circuit(qc, use_mps)
        all_layers_data.append((layer_idx + 1, probabilities, phases))

    return all_layers_data


# --- 5. EXPORTAR A WAV ---
def export_to_wav(probabilities, phases, filename="quantum_texture.wav", sample_rate=44100, duration_sec=3):
    total_samples = int(sample_rate * duration_sec)

    x_original = np.linspace(0, 1, len(probabilities))
    x_audio = np.linspace(0, 1, total_samples)

    # Sampleo interpolado: convierte las probabilidades discretas en una señal continua
    prob_interp = np.interp(x_audio, x_original, probabilities)

    # Interpolación de la fase
    phases_unwrapped = np.unwrap(phases)
    phase_interp = np.interp(x_audio, x_original, phases_unwrapped)

    # La probabilidad controla la amplitud audible
    if np.max(prob_interp) > 0:
        prob_norm = prob_interp / np.max(prob_interp)
    else:
        prob_norm = prob_interp

    carrier_freq = 220
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    left = prob_norm * np.sin(2 * np.pi * carrier_freq * t + phase_interp)
    right = prob_norm * np.sin(2 * np.pi * carrier_freq * 1.5 * t + phase_interp)

    fade_samples = int(0.05 * sample_rate)

    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    envelope = np.ones(total_samples)
    envelope[:fade_samples] = fade_in
    envelope[-fade_samples:] = fade_out

    left *= envelope
    right *= envelope

    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))

    if max_val > 0:
        left = left / max_val
        right = right / max_val

    stereo_audio = np.column_stack((left, right))
    stereo_audio_int16 = (stereo_audio * 32767).astype(np.int16)

    wavfile.write(filename, sample_rate, stereo_audio_int16)

    print(f"Archivo guardado: {filename}")


# --- 6. BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    num_qubits = 15
    layers = 10

    print(f"Generando circuito RQC de {num_qubits} qubits y {layers} capas...")
    print(f"Se generarán {layers + 1} archivos WAV.")
    print("Cada archivo sonifica las probabilidades de una capa del circuito.")

    layer_data = generate_quantum_audio_data_by_layer(
        num_qubits,
        layers,
        use_mps=False
    )

    for layer_number, probs, phas in layer_data:

        if layer_number == 0:
            filename = "rqc_layer_00_initial.wav"
        else:
            filename = f"rqc_layer_{layer_number:02d}.wav"

        export_to_wav(probs, phas, filename)