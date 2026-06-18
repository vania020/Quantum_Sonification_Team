# ============================================================
# sonification.py (Rama: Drone Sci-Fi por Espectro de Schmidt)
# ============================================================
import numpy as np
import scipy.io.wavfile as wavfile

def export_schmidt_sci_fi_to_wav(eigenvalues, filename, sample_rate=44100, duration_sec=6.0):
    """
    Sintetizador Sci-Fi Aditivo.
    Cada autovalor genera una capa de sonido inarmónico. Su peso dicta el volumen,
    y modula osciladores FM y LFOs para darle un efecto de "respiración alienígena".
    """
    total_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    # Frecuencia base oscura (Do2 - C2)
    base_freq = 65.41 
    audio_mix = np.zeros(total_samples)
    
    # Proporción Áurea para generar tensión inarmónica metálica
    golden_ratio = 1.61803398 
    
    for i, val in enumerate(eigenvalues):
        # Frecuencias espaciadas irracionalmente
        freq = base_freq * (1.0 + (i * golden_ratio))
        
        if freq > 18000:
            continue
            
        # LFO (Oscilador de Baja Frecuencia) que depende del autovalor
        # Autovalores grandes respiran rápido, los pequeños son lentos
        lfo_rate = 0.5 + (val * 3.0) 
        lfo = 0.5 * (1 + np.sin(2 * np.pi * lfo_rate * t))
        
        # Síntesis FM (Modulación de Frecuencia para textura metálica)
        mod_index = val * 8.0
        modulator = np.sin(2 * np.pi * (freq * 0.5) * t) * mod_index
        
        voice = np.sin(2 * np.pi * freq * t + modulator)
        
        # La probabilidad física dicta la amplitud
        voice *= val 
        voice *= lfo
        
        audio_mix += voice

    # Envolvente lenta (Attack y Release masivos para estilo Pad Sci-Fi)
    attack = int(0.3 * sample_rate)
    decay = int(0.5 * sample_rate)
    env = np.ones(total_samples)
    env[:attack] = np.linspace(0, 1, attack)
    env[-decay:] = np.linspace(1, 0, decay)
    
    audio_mix *= env
    
    # Efecto "Haas" estéreo para ampliar la imagen acústica
    left = audio_mix * 0.8
    # Retrasamos el canal derecho 20 milisegundos para estéreo inmenso
    right = np.roll(audio_mix, int(sample_rate * 0.02)) * 0.8 
    
    final_audio = np.vstack((left, right)).T
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val
        
    wavfile.write(filename, sample_rate, final_audio.astype(np.float32))