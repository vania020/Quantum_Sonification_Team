import numpy as np
import networkx as nx
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace, entropy
from circuit import apply_random_circuit, apply_random_circuit_with_qft


def simulate_statevector(qc, simulator=None):
    """Return the statevector of a circuit using Aer."""
    if simulator is None:
        simulator = AerSimulator(method="statevector")

    qc_run = qc.copy()
    qc_run.save_statevector()
    qc_comp = transpile(qc_run, backend=simulator)
    result = simulator.run(qc_comp).result()
    return np.asarray(result.get_statevector(qc_comp), dtype=complex)


def get_bloch_data_from_statevector(state, num_qubits, eps=1e-12):
    """
    Extract local Bloch descriptors for every qubit from the global statevector.

    Returns theta, phi, r, mixedness=1-r, and x,y,z.
    No stochastic noise is introduced here.
    """
    dim = len(state)
    expected_dim = 2 ** num_qubits
    if dim != expected_dim:
        raise ValueError(f"Statevector dimension {dim} does not match 2^{num_qubits} = {expected_dim}.")

    indices = np.arange(dim)
    bloch_data = {
        "theta": np.zeros(num_qubits),
        "phi": np.zeros(num_qubits),
        "r": np.zeros(num_qubits),
        "mixedness": np.zeros(num_qubits),
        "x": np.zeros(num_qubits),
        "y": np.zeros(num_qubits),
        "z": np.zeros(num_qubits),
    }

    for q in range(num_qubits):
        mask = 1 << q
        indices_0 = indices[(indices & mask) == 0]
        indices_1 = indices_0 | mask

        amp_0 = state[indices_0]
        amp_1 = state[indices_1]

        p0 = np.sum(np.abs(amp_0) ** 2)
        p1 = np.sum(np.abs(amp_1) ** 2)
        coherence = np.vdot(amp_0, amp_1)

        x = 2.0 * np.real(coherence)
        y = 2.0 * np.imag(coherence)
        z = p0 - p1

        r = float(np.clip(np.sqrt(x * x + y * y + z * z), 0.0, 1.0))

        if r > eps:
            theta = float(np.arccos(np.clip(z / r, -1.0, 1.0)))
            phi = float(np.arctan2(y, x))
            if phi < 0:
                phi += 2.0 * np.pi
        else:
            theta = 0.0
            phi = 0.0

        bloch_data["theta"][q] = theta
        bloch_data["phi"][q] = phi
        bloch_data["r"][q] = r
        bloch_data["mixedness"][q] = 1.0 - r
        bloch_data["x"][q] = x
        bloch_data["y"][q] = y
        bloch_data["z"][q] = z

    return bloch_data


def build_mutual_information_graph(state, num_qubits):
    """
    Build the weighted nearest-neighbor graph.

    Edge weight:
        I(q_i : q_{i+1}) = S(q_i) + S(q_{i+1}) - S(q_i q_{i+1})

    The graph is computed from the base RQC state. That keeps BFS/DFS timing
    identical across Bloch, spectral_base, and spectral_qft modes.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_qubits))

    for i in range(num_qubits - 1):
        keep = [i, i + 1]
        trace_over = [j for j in range(num_qubits) if j not in keep]

        rho_ab = partial_trace(state, trace_over)
        s_a = entropy(partial_trace(rho_ab, [1]), base=2)
        s_b = entropy(partial_trace(rho_ab, [0]), base=2)
        s_ab = entropy(rho_ab, base=2)

        mut_info = max(0.0, float(np.real(s_a + s_b - s_ab)))
        G.add_edge(i, i + 1, weight=mut_info)

    return G


def probability_metrics(state, top_k=32, eps=1e-15):
    """Concentration metrics for checking whether QFT changes usable peaks."""
    probs = np.abs(state) ** 2
    probs = probs / np.sum(probs)

    sorted_probs = np.sort(probs)[::-1]
    k = min(top_k, len(sorted_probs))

    return {
        "top_k_mass": float(np.sum(sorted_probs[:k])),
        "shannon_entropy": float(-np.sum(probs * np.log2(probs + eps))),
        "participation_ratio": float(1.0 / np.sum(probs ** 2)),
    }


def generate_fair_comparison_data(num_qubits, layers, seed=42, qft_do_swaps=True, top_k=32):
    """
    Generate layer data for a controlled comparison.

    For every layer:
      - base state |psi_l>
      - graph from base state
      - Bloch descriptors from base state
      - QFT state QFT|psi_l>
      - probability concentration metrics for base and QFT states
    """
    simulator_sv = AerSimulator(method="statevector")
    all_layers = []

    for layer_idx in range(layers + 1):
        qc_base = apply_random_circuit(num_qubits, layer_idx, seed=seed)
        state_base = simulate_statevector(qc_base, simulator=simulator_sv)

        graph = build_mutual_information_graph(state_base, num_qubits)
        bloch_data = get_bloch_data_from_statevector(state_base, num_qubits)

        qc_qft = apply_random_circuit_with_qft(
            num_qubits, layer_idx, seed=seed, do_swaps=qft_do_swaps
        )
        state_qft = simulate_statevector(qc_qft, simulator=simulator_sv)

        all_layers.append({
            "layer": layer_idx,
            "graph": graph,
            "bloch": bloch_data,
            "state_base": state_base,
            "state_qft": state_qft,
            "metrics_base": probability_metrics(state_base, top_k=top_k),
            "metrics_qft": probability_metrics(state_qft, top_k=top_k),
        })

    return all_layers
