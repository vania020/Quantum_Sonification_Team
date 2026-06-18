import numpy as np
import scipy.io.wavfile as wavfile
import networkx as nx

def export_trinity_architecture_to_wav(graph, statevector, filename, sample_rate=44100):
    probs = np.abs(statevector)**2
    top_k = 32  
    peak_indices = np.argsort(probs)[-top_k:]
    
    base_freq_bfs = 55.0  
    base_freq_dfs = 220.0 
    pentatonic_ratios = [1.0, 1.122, 1.25, 1.5, 1.666]
    
    num_nodes = len(graph.nodes)
    center_node = num_nodes // 2

    def synthesize_note(node_id, base_f, duration_sec, is_percussive=False):
        total_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)
        
        octave = node_id // 5
        ratio = pentatonic_ratios[node_id % 5]
        fund = base_f * ratio * (2 ** octave)
        
        snd = np.zeros(total_samples)
        for idx in peak_indices:
            p = probs[idx]
            if p < 1e-4: continue
            harmonic = (idx % 16) + 1 
            freq = fund * harmonic
            if freq < 12000:
                snd += (p * 150) * np.sin(2 * np.pi * freq * t)
                
        if np.max(np.abs(snd)) == 0:
            snd = 0.5 * np.sin(2 * np.pi * fund * t)
        
        if is_percussive: 
            attack = int(0.05 * total_samples)
        else:             
            attack = int(0.3 * total_samples)
            
        decay = total_samples - attack
        env = np.concatenate([np.linspace(0, 1, attack), np.linspace(1, 0, decay)])
        snd *= env
        
        pan = node_id / (num_nodes - 1)
        left = snd * np.cos(pan * np.pi / 2)
        right = snd * np.sin(pan * np.pi / 2)
        return np.vstack((left, right)).T

    lengths = nx.single_source_shortest_path_length(graph, center_node)
    max_dist = max(lengths.values()) if lengths else 0
    
    bfs_sequence = []
    for dist in range(max_dist + 1):
        nodes_in_layer = [n for n, d in lengths.items() if d == dist]
        
        if dist == 0: avg_weight = 0.1
        else:
            weights = [graph.get_edge_data(nx.shortest_path(graph, center_node, n)[-2], n)['weight'] for n in nodes_in_layer]
            avg_weight = np.mean(weights)
            
        dur = np.clip(1.5 / (avg_weight + 1e-5), 0.5, 3.0)
        layer_audio = np.zeros((int(sample_rate * dur), 2))
        
        for n in nodes_in_layer:
            layer_audio += synthesize_note(n, base_freq_bfs, dur, is_percussive=False) / len(nodes_in_layer)
        bfs_sequence.append(layer_audio)
        
    bfs_full = np.concatenate(bfs_sequence) if bfs_sequence else np.zeros((10, 2))

    dfs_edges = list(nx.dfs_edges(graph, source=center_node))
    dfs_sequence = []
    
    if not dfs_edges:
        dfs_sequence.append(synthesize_note(center_node, base_freq_dfs, 2.0, is_percussive=True))
    else:
        for u, v in dfs_edges:
            weight = graph.get_edge_data(u, v)['weight']
            dur = np.clip(0.4 / (weight + 1e-5), 0.08, 0.4)
            dfs_sequence.append(synthesize_note(v, base_freq_dfs, dur, is_percussive=True))
            
    dfs_full = np.concatenate(dfs_sequence) if dfs_sequence else np.zeros((10, 2))

    len_bfs = len(bfs_full)
    len_dfs = len(dfs_full)
    
    if len_dfs > 0 and len_bfs > 0:
        if len_dfs < len_bfs:
            repeats = (len_bfs // len_dfs) + 1
            dfs_looped = np.tile(dfs_full, (repeats, 1))
            dfs_padded = dfs_looped[:len_bfs]
            bfs_padded = bfs_full
        else:
            repeats = (len_dfs // len_bfs) + 1
            bfs_looped = np.tile(bfs_full, (repeats, 1))
            bfs_padded = bfs_looped[:len_dfs]
            dfs_padded = dfs_full
    else:
        bfs_padded = bfs_full
        dfs_padded = dfs_full
    
    final_audio = (bfs_padded * 0.6) + (dfs_padded * 1.0)
    
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val
        
    wavfile.write(filename, sample_rate, final_audio.astype(np.float32))