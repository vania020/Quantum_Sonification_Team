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


# --- 3. GENERADOR DE DATOS PARA SONIFICACIÓN ---
def generate_quantum_audio_data(num_qubits, layers, use_mps=False):
    qc = apply_random_circuit(num_qubits, layers)
    qc.save_statevector()

    if use_mps:
        simulator = AerSimulator(method="matrix_product_state")
    else:
        simulator = AerSimulator(method="statevector")

    result = simulator.run(qc).result()
    state = np.asarray(result.get_statevector())

    magnitudes = np.abs(state)
    phases = np.angle(state)

    return magnitudes, phases


def export_to_wav(magnitudes, phases, filename="quantum_texture.wav", sample_rate=44100, duration_sec=3):
    """
    Convierte magnitudes y fases del statevector en una textura sonora reproducible.

    No cambia la lógica cuántica:
    - magnitudes siguen viniendo del statevector
    - phases siguen viniendo del statevector

    Pero ahora:
    - se interpola a una duración fija
    - se centra la señal alrededor de 0
    - se suaviza la fase
    - se evita clipping
    - se agrega fade in/out
    """

    total_samples = int(sample_rate * duration_sec)

    x_original = np.linspace(0, 1, len(magnitudes))
    x_audio = np.linspace(0, 1, total_samples)

    # Interpolar magnitudes y fases para obtener duración fija
    mag_interp = np.interp(x_audio, x_original, magnitudes)

    # Desenrollar fase para evitar saltos bruscos entre -pi y pi
    phases_unwrapped = np.unwrap(phases)
    phase_interp = np.interp(x_audio, x_original, phases_unwrapped)

    # Normalizar magnitudes
    if np.max(mag_interp) > 0:
        mag_norm = mag_interp / np.max(mag_interp)
    else:
        mag_norm = mag_interp

    # Crear una señal audible usando fase como argumento oscilatorio
    carrier_freq = 220  # Hz, frecuencia base
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    left = mag_norm * np.sin(2 * np.pi * carrier_freq * t + phase_interp)
    right = mag_norm * np.sin(2 * np.pi * carrier_freq * 1.5 * t + phase_interp)

    # Fade in / fade out para evitar clicks
    fade_samples = int(0.05 * sample_rate)

    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    envelope = np.ones(total_samples)
    envelope[:fade_samples] = fade_in
    envelope[-fade_samples:] = fade_out

    left *= envelope
    right *= envelope

    # Normalizar para evitar saturación
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))

    if max_val > 0:
        left = left / max_val
        right = right / max_val

    stereo_audio = np.column_stack((left, right))

    # Convertir a int16 para WAV
    stereo_audio_int16 = (stereo_audio * 32767).astype(np.int16)

    wavfile.write(filename, sample_rate, stereo_audio_int16)

    print(f"Archivo guardado exitosamente: {filename}")
    print(f"Duración: {duration_sec} segundos")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Muestras originales del statevector: {len(magnitudes)}")
    print(f"Muestras finales del audio: {total_samples}")


# --- 4. BLOQUE PRINCIPAL (EJECUCIÓN) ---
if __name__ == "__main__":
    # Semilla para reproducibilidad
    random.seed(0)
    np.random.seed(0)

    # N=15 generará 32768 muestras
    num_qubits = 15
    layers = 10

    print(f"Generando circuito RQC de {num_qubits} qubits y {layers} capas...")

    mags, phas = generate_quantum_audio_data(
        num_qubits,
        layers,
        use_mps=False
    )

    export_to_wav(mags, phas, "rqc_layer10.wav")
