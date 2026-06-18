# ============================================================
# sonification.py (NUEVO: Sintetizador de Espectro de Schmidt)
# ============================================================
import numpy as np
import scipy.io.wavfile as wavfile

def export_schmidt_spectrum_to_wav(eigenvalues, filename, sample_rate=44100, duration_sec=5.0):
    """
    Mapeo Espectral Físico:
    Convierte la distribución de autovalores de entrelazamiento
    en la firma timbral (espectro inarmónico) del sonido.
    """
    total_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    # Señal mono que luego duplicaremos
    audio_signal = np.zeros(total_samples)
    
    # Frecuencia fundamental del sistema no entrelazado
    base_freq = 110.0 # La2 (A2)
    
    # Reconstrucción de la Onda a partir del Entrelazamiento
    for i, val in enumerate(eigenvalues):
        # Mapeo Vanguardista Inarmónico:
        # Los modos de entrelazamiento no son múltiplos enteros (no es música clásica).
        # Usamos la raíz cuadrada para crear un espectro denso, metálico y complejo (campana).
        freq = base_freq * np.sqrt(i + 1)
        
        if freq < 20000: # Límite de Nyquist / Audición
            # La amplitud del armónico es exactamente el peso del autovalor
            audio_signal += val * np.sin(2 * np.pi * freq * t)

    # Envolvente natural (evita clicks)
    attack = int(0.1 * sample_rate)
    decay = int(0.5 * sample_rate)
    env = np.ones(total_samples)
    env[:attack] = np.linspace(0, 1, attack)
    env[-decay:] = np.linspace(1, 0, decay)
    audio_signal *= env
    
    # Normalización matemática preservando las proporciones físicas
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        audio_signal = (audio_signal / max_val) * 0.9 
        
    # Salida estéreo idéntica (el fenómeno es de Timbre, no de Espacio)
    stereo_audio = np.vstack((audio_signal, audio_signal)).T
    
    wavfile.write(filename, sample_rate, stereo_audio.astype(np.float32))