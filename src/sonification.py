# ============================================================
# sonification.py (Actualización: Binaural Beats y Ducking)
# ============================================================
import numpy as np
import scipy.io.wavfile as wavfile

def export_wigner_shadows_to_wav(layer_pairs_data, filename, sample_rate=44100, duration_sec=6.0):
    """
    Sintetizador Psicoacústico de Sombras (Versión Neuro-Acústica).
    La negatividad cuántica (Sombras) genera latidos binaurales en el cerebro
    y silencia (ducking) las frecuencias reales de la matriz de densidad.
    """
    total_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    audio_left = np.zeros(total_samples)
    audio_right = np.zeros(total_samples)
    
    base_freq = 130.81 # Do3 (C3) - Frecuencia media/aguda cristalina para la "Realidad"
    shadow_carrier = 55.0 # La1 (A1) - Base ronca para la "Sombra"
    
    for data in layer_pairs_data:
        pair_idx = data["pair"][0]
        pos_evals = data["positive"]
        neg_evals = data["negative"]
        
        # Frecuencia de la partícula (Realidad)
        real_freq = base_freq * (pair_idx + 1) * 0.5
        
        # 1. Medir la masa total de la Sombra (Suma de negatividad)
        pair_negativity = sum(abs(n) for n in neg_evals)
        
        # 2. SÍNTESIS DE LA REALIDAD (El orden clásico)
        real_signal = np.zeros(total_samples)
        for i, pos_val in enumerate(pos_evals):
            harmonic_freq = real_freq * (i + 1)
            if harmonic_freq < 12000:
                real_signal += (pos_val * 0.8) * np.sin(2 * np.pi * harmonic_freq * t)
                
        # 3. EL VACÍO DEVORA LA LUZ (Ducking)
        # Si la negatividad crece, el volumen de la realidad colapsa hacia cero
        ducking_factor = max(0.0, 1.0 - (pair_negativity * 15.0))
        real_signal *= ducking_factor
        
        audio_left += real_signal
        audio_right += real_signal
        
        # 4. SÍNTESIS DE LA SOMBRA (Latidos Binaurales)
        if pair_negativity > 0.0001:
            # Amplificamos la sombra con raíz cuadrada (controlado pero presente)
            amp_shadow = (pair_negativity ** 0.5) * 3.0
            
            # La física de la ilusión:
            # El latido variará desde 2 Hz (lento/hipnótico) hasta 20 Hz (ansiedad/presión)
            # dependiendo de la cantidad de negatividad cuántica.
            beat_freq = np.clip(pair_negativity * 200.0, 2.0, 20.0)
            
            # Canal Izquierdo: Frecuencia portadora plana
            shadow_left = amp_shadow * np.sin(2 * np.pi * shadow_carrier * t)
            # Canal Derecho: Portadora + Desfase (crea la ilusión en el cerebro)
            shadow_right = amp_shadow * np.sin(2 * np.pi * (shadow_carrier + beat_freq) * t)
            
            audio_left += shadow_left
            audio_right += shadow_right
                
    # Envolvente general atmosférica
    attack = int(0.15 * sample_rate)
    decay = int(0.4 * sample_rate)
    env = np.ones(total_samples)
    env[:attack] = np.linspace(0, 1, attack)
    env[-decay:] = np.linspace(1, 0, decay)
    
    audio_left *= env
    audio_right *= env
    
    # Normalización matemática estricta para máxima claridad
    final_audio = np.vstack((audio_left, audio_right)).T
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = (final_audio / max_val) * 0.85 
        
    wavfile.write(filename, sample_rate, final_audio.astype(np.float32))