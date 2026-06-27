import numpy as np
import scipy.io.wavfile as wavfile
import networkx as nx


DEFAULT_CONFIG = {
    "sample_rate": 44100,
    "base_freq_bfs": 55.0,
    "base_freq_dfs": 220.0,
    "pentatonic_ratios": [1.0, 1.122, 1.25, 1.5, 1.666],
    "bfs_duration_scale": 1.5,
    "bfs_min_duration": 0.5,
    "bfs_max_duration": 3.0,
    "dfs_duration_scale": 0.4,
    "dfs_min_duration": 0.08,
    "dfs_max_duration": 0.4,
    "bfs_attack_fraction": 0.30,
    "dfs_attack_fraction": 0.05,
    "bfs_gain": 0.6,
    "dfs_gain": 1.0,
    "bloch_microtune_hz": 5.0,
    "bloch_fm_index_scale": 2.0,
    "bloch_texture_rate_hz": 8.0,
    "bloch_texture_depth": 0.18,
    "bloch_mixedness_gain_reduction": 0.25,
    "top_k": 32,
    "spectral_min_probability": 1e-4,
    "spectral_max_frequency": 12000.0,
    "spectral_harmonic_modulo": 16,
}


def merge_config(config):
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)
    return merged


def make_envelope(total_samples, attack_fraction):
    """Asymmetric envelope 0 -> 1 -> 0. It does not change event duration."""
    if total_samples <= 1:
        return np.ones(max(total_samples, 1))

    attack = int(np.clip(attack_fraction, 0.0, 1.0) * total_samples)
    attack = max(1, min(attack, total_samples - 1))
    decay = total_samples - attack

    return np.concatenate([
        np.linspace(0.0, 1.0, attack, endpoint=False),
        np.linspace(1.0, 0.0, decay, endpoint=True),
    ])


def node_pitch(node_id, base_f, config):
    ratios = config["pentatonic_ratios"]
    octave = node_id // len(ratios)
    ratio = ratios[node_id % len(ratios)]
    return base_f * ratio * (2 ** octave)


def stereo_pan(signal, node_id, num_nodes):
    pan = 0.5 if num_nodes <= 1 else node_id / (num_nodes - 1)
    left = signal * np.cos(pan * np.pi / 2.0)
    right = signal * np.sin(pan * np.pi / 2.0)
    return np.vstack((left, right)).T


def synthesize_bloch_note(node_id, base_f, duration_sec, bloch_data, num_nodes, is_percussive, config):
    """
    Deterministic Bloch synthesis. No stochastic noise.

    theta_q -> microtuning
    phi_q   -> FM index
    1-r_q   -> deterministic amplitude texture and mild gain reduction
    node q  -> pitch scaffold, octave group, stereo position
    """
    sample_rate = config["sample_rate"]
    total_samples = max(1, int(sample_rate * duration_sec))
    t = np.linspace(0.0, duration_sec, total_samples, endpoint=False)

    theta = float(bloch_data["theta"][node_id])
    phi = float(bloch_data["phi"][node_id])
    r = float(bloch_data["r"][node_id])
    mix = float(bloch_data["mixedness"][node_id])

    fund = node_pitch(node_id, base_f, config) + config["bloch_microtune_hz"] * theta

    phi_norm = phi / (2.0 * np.pi)
    fm_index = config["bloch_fm_index_scale"] * phi_norm
    modulator = fm_index * np.sin(2.0 * np.pi * (2.0 * fund) * t)
    signal = np.sin(2.0 * np.pi * fund * t + modulator)

    texture = 1.0 + config["bloch_texture_depth"] * mix * np.sin(
        2.0 * np.pi * config["bloch_texture_rate_hz"] * t
    )
    gain = r * (1.0 - config["bloch_mixedness_gain_reduction"] * mix)
    signal = gain * texture * signal

    attack_fraction = config["dfs_attack_fraction"] if is_percussive else config["bfs_attack_fraction"]
    signal *= make_envelope(total_samples, attack_fraction)

    return stereo_pan(signal, node_id, num_nodes)


def spectral_peak_indices(statevector, config):
    probs = np.abs(statevector) ** 2
    probs = probs / np.sum(probs)
    top_k = min(int(config["top_k"]), len(probs))
    indices = np.argsort(probs)[-top_k:]
    indices = indices[np.argsort(probs[indices])[::-1]]
    return probs, indices


def synthesize_spectral_note(node_id, base_f, duration_sec, statevector, num_nodes, is_percussive, config):
    """
    Deterministic spectral synthesis.

    The same function is used for the base state and the QFT-transformed state.
    This is the fair QFT toggle.
    """
    sample_rate = config["sample_rate"]
    total_samples = max(1, int(sample_rate * duration_sec))
    t = np.linspace(0.0, duration_sec, total_samples, endpoint=False)

    fund = node_pitch(node_id, base_f, config)
    probs, peak_indices = spectral_peak_indices(statevector, config)
    selected_probs = probs[peak_indices]

    valid = selected_probs >= config["spectral_min_probability"]
    peak_indices = peak_indices[valid]
    selected_probs = selected_probs[valid]

    signal = np.zeros(total_samples)

    if len(peak_indices) == 0 or np.sum(selected_probs) <= 0:
        signal = 0.5 * np.sin(2.0 * np.pi * fund * t)
    else:
        weights = selected_probs / np.sum(selected_probs)
        for idx, weight in zip(peak_indices, weights):
            harmonic = (int(idx) % int(config["spectral_harmonic_modulo"])) + 1
            freq = fund * harmonic
            if freq <= config["spectral_max_frequency"]:
                signal += weight * np.sin(2.0 * np.pi * freq * t)

    max_abs = np.max(np.abs(signal))
    if max_abs > 0:
        signal = signal / max_abs

    attack_fraction = config["dfs_attack_fraction"] if is_percussive else config["bfs_attack_fraction"]
    signal *= make_envelope(total_samples, attack_fraction)

    return stereo_pan(signal, node_id, num_nodes)


def bfs_frontier_average_weight(graph, center_node, nodes_in_layer, dist):
    """Average only the frontier edges entering the current BFS layer."""
    if dist == 0:
        return 0.1

    weights = []
    for node in nodes_in_layer:
        path = nx.shortest_path(graph, center_node, node)
        previous_node = path[-2]
        weights.append(graph.get_edge_data(previous_node, node)["weight"])

    return float(np.mean(weights)) if weights else 0.1


def render_with_bfs_dfs(graph, note_synth, config):
    """Common BFS/DFS renderer used by all comparison modes."""
    sample_rate = config["sample_rate"]
    num_nodes = len(graph.nodes)
    center_node = num_nodes // 2

    lengths = nx.single_source_shortest_path_length(graph, center_node)
    max_dist = max(lengths.values()) if lengths else 0

    bfs_sequence = []
    for dist in range(max_dist + 1):
        nodes_in_layer = [node for node, d in lengths.items() if d == dist]
        avg_weight = bfs_frontier_average_weight(graph, center_node, nodes_in_layer, dist)
        dur = np.clip(
            config["bfs_duration_scale"] / (avg_weight + 1e-5),
            config["bfs_min_duration"],
            config["bfs_max_duration"],
        )
        layer_audio = np.zeros((int(sample_rate * dur), 2))
        for node in nodes_in_layer:
            layer_audio += note_synth(node, config["base_freq_bfs"], dur, False) / max(1, len(nodes_in_layer))
        bfs_sequence.append(layer_audio)
    bfs_full = np.concatenate(bfs_sequence) if bfs_sequence else np.zeros((10, 2))

    dfs_edges = list(nx.dfs_edges(graph, source=center_node))
    dfs_sequence = []
    if not dfs_edges:
        dfs_sequence.append(note_synth(center_node, config["base_freq_dfs"], 2.0, True))
    else:
        for u, v in dfs_edges:
            weight = graph.get_edge_data(u, v)["weight"]
            dur = np.clip(
                config["dfs_duration_scale"] / (weight + 1e-5),
                config["dfs_min_duration"],
                config["dfs_max_duration"],
            )
            dfs_sequence.append(note_synth(v, config["base_freq_dfs"], dur, True))
    dfs_full = np.concatenate(dfs_sequence) if dfs_sequence else np.zeros((10, 2))

    len_bfs = len(bfs_full)
    len_dfs = len(dfs_full)

    if len_bfs == 0 or len_dfs == 0:
        final_audio = bfs_full if len_bfs >= len_dfs else dfs_full
    elif len_dfs < len_bfs:
        repeats = (len_bfs // len_dfs) + 1
        dfs_padded = np.tile(dfs_full, (repeats, 1))[:len_bfs]
        final_audio = config["bfs_gain"] * bfs_full + config["dfs_gain"] * dfs_padded
    else:
        repeats = (len_dfs // len_bfs) + 1
        bfs_padded = np.tile(bfs_full, (repeats, 1))[:len_dfs]
        final_audio = config["bfs_gain"] * bfs_padded + config["dfs_gain"] * dfs_full

    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val

    return final_audio.astype(np.float32)


def export_layer_audio(graph, layer_payload, filename, mode, config=None):
    """
    Export one layer to WAV.

    mode:
        bloch         -> deterministic Bloch descriptors, no stochastic noise
        spectral_base -> spectral engine using |psi_l>
        spectral_qft  -> same spectral engine using QFT|psi_l>
    """
    config = merge_config(config)
    num_nodes = len(graph.nodes)

    if mode == "bloch":
        bloch_data = layer_payload["bloch"]
        def note_synth(node_id, base_f, duration_sec, is_percussive):
            return synthesize_bloch_note(node_id, base_f, duration_sec, bloch_data, num_nodes, is_percussive, config)
    elif mode == "spectral_base":
        statevector = layer_payload["state_base"]
        def note_synth(node_id, base_f, duration_sec, is_percussive):
            return synthesize_spectral_note(node_id, base_f, duration_sec, statevector, num_nodes, is_percussive, config)
    elif mode == "spectral_qft":
        statevector = layer_payload["state_qft"]
        def note_synth(node_id, base_f, duration_sec, is_percussive):
            return synthesize_spectral_note(node_id, base_f, duration_sec, statevector, num_nodes, is_percussive, config)
    else:
        raise ValueError("mode must be 'bloch', 'spectral_base', or 'spectral_qft'")

    audio = render_with_bfs_dfs(graph, note_synth, config)
    wavfile.write(filename, config["sample_rate"], audio)
