import numpy as np
import scipy.io.wavfile as wavfile
import random

def export_granular_entropy_to_wav(entropy_val, filename, sample_rate=44100, duration_sec=5.0):
    max_S = 7.0 
    norm_S = np.clip(entropy_val / max_S, 0.0, 1.0)
    
    total_samples = int(sample_rate * duration_sec)
    audio_left = np.zeros(total_samples)
    audio_right = np.zeros(total_samples)
    
    base_freq = 110.0 # La2 (A2)
    
    if norm_S < 0.02:
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * base_freq * t)
        audio_left = signal
        audio_right = signal
        
    else:
        grain_min_ms = 10 + (1.0 - norm_S) * 90 
        grain_max_ms = 20 + (1.0 - norm_S) * 180 
        num_grains = int(10 + (norm_S * 1500)) 
        
        for _ in range(num_grains):
            grain_dur_ms = random.uniform(grain_min_ms, grain_max_ms)
            grain_samples = int((grain_dur_ms / 1000.0) * sample_rate)
            
            pitch_shift = random.uniform(1.0 - (norm_S * 0.7), 1.0 + (norm_S * 0.7))
            grain_freq = base_freq * pitch_shift
            
            t_grain = np.linspace(0, grain_dur_ms / 1000.0, grain_samples, endpoint=False)
            grain_wave = np.sin(2 * np.pi * grain_freq * t_grain)
            
            window = np.hanning(grain_samples)
            grain_wave *= window
            
            start_sample = random.randint(0, total_samples - grain_samples - 1)
            
            pan = random.uniform(0.5 - (norm_S * 0.5), 0.5 + (norm_S * 0.5))
            left_gain = np.cos(pan * np.pi / 2)
            right_gain = np.sin(pan * np.pi / 2)
            
            audio_left[start_sample:start_sample + grain_samples] += grain_wave * left_gain * 0.15
            audio_right[start_sample:start_sample + grain_samples] += grain_wave * right_gain * 0.15

    attack = int(0.15 * sample_rate)
    decay = int(0.4 * sample_rate)
    env = np.ones(total_samples)
    env[:attack] = np.linspace(0, 1, attack)
    env[-decay:] = np.linspace(1, 0, decay)
    
    audio_left *= env
    audio_right *= env
    
    final_audio = np.vstack((audio_left, audio_right)).T
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val
        
    wavfile.write(filename, sample_rate, final_audio.astype(np.float32))