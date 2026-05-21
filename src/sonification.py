import numpy as np
import scipy.io.wavfile as wavfile



# ============================================================
# 7. EXPORTAR DATOS DE BLOCH A WAV
# ============================================================

def export_bloch_to_wav(
    bloch_data,
    filename="bloch_texture.wav",
    sample_rate=44100,
    duration_sec=3,
    base_freq=220,
    freq_range=660
):
    """
    Convierte datos de Bloch local en audio estéreo.

    Mapeo:
        theta      -> frecuencia
        phi        -> paneo estéreo
        r          -> amplitud
        mixedness  -> textura/modulación
    """

    theta = bloch_data["theta"]
    phi = bloch_data["phi"]
    r = bloch_data["r"]
    mixedness = bloch_data["mixedness"]

    num_qubits = len(theta)

    total_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    left = np.zeros(total_samples)
    right = np.zeros(total_samples)

    for q in range(num_qubits):

        # theta controla frecuencia
        freq = base_freq + (theta[q] / np.pi) * freq_range

        # phi controla paneo estéreo
        pan = phi[q] / (2 * np.pi)

        # r controla amplitud
        amp = r[q] / num_qubits

        # 1-r controla textura
        texture_depth = mixedness[q]

        modulation = 1.0 + 0.15 * texture_depth * np.sin(2 * np.pi * 8 * t)

        signal = amp * modulation * np.sin(2 * np.pi * freq * t)

        # Paneo estéreo
        left_gain = np.cos(pan * np.pi / 2)
        right_gain = np.sin(pan * np.pi / 2)

        left += left_gain * signal
        right += right_gain * signal

    # Fade in / fade out
    fade_samples = int(0.05 * sample_rate)

    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    envelope = np.ones(total_samples)
    envelope[:fade_samples] = fade_in
    envelope[-fade_samples:] = fade_out

    left *= envelope
    right *= envelope

    # Normalización final
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))

    if max_val > 0:
        left = left / max_val
        right = right / max_val

    stereo_audio = np.column_stack((left, right))
    stereo_audio_int16 = (stereo_audio * 32767).astype(np.int16)

    wavfile.write(filename, sample_rate, stereo_audio_int16)

    print(f"Archivo guardado: {filename}")