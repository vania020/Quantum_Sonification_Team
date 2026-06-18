# ============================================================
# sonification.py (Rama: Sintetizador de Espectro de Schmidt)
# ============================================================
import numpy as np
import scipy.io.wavfile as wavfile

def export_schmidt_spectrum_to_wav(eigenvalues, filename, sample_rate=44100, duration_sec=5.0):
    total_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    audio_signal = np.zeros(total_samples)
    
    # Frecuencia fundamental del sistema no entrelazado
    base_freq = 110.0 # La2 (A2)
    
    for i, val in enumerate(eigenvalues):
        # Mapeo Vanguardista Inarmónico guiado por la Ley de Weyl
        freq = base_freq * np.sqrt(i + 1)
        
        if freq < 20000: # Límite de audición
            audio_signal += val * np.sin(2 * np.pi * freq * t)

    # Envolvente natural (evita clicks)
    attack = int(0.1 * sample_rate)
    decay = int(0.5 * sample_rate)
    env = np.ones(total_samples)
    env[:attack] = np.linspace(0, 1, attack)
    env[-decay:] = np.linspace(1, 0, decay)
    audio_signal *= env
    
    # Normalización preservando las proporciones de los autovalores
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        audio_signal = (audio_signal / max_val) * 0.9 
        
    # Salida estéreo idéntica (el fenómeno es de Timbre inarmónico)
    stereo_audio = np.vstack((audio_signal, audio_signal)).T
    wavfile.write(filename, sample_rate, stereo_audio.astype(np.float32))