# ============================================================
# simulations.py (NUEVO: Grafo BFS/DFS + Datos de Bloch Locales)
# ============================================================
import numpy as np
import networkx as nx
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace, entropy
from circuit import apply_random_circuit

def get_bloch_data_from_statevector(state, num_qubits, eps=1e-12):
    """ Extrae theta, phi, r y mixedness de cada qubit localmente. """
    dim = len(state)
    indices = np.arange(dim)
    
    bloch_data = {'theta': [], 'phi': [], 'r': [], 'mixedness': []}

    for qubit in range(num_qubits):
        mask = 1 << qubit
        indices_0 = indices[(indices & mask) == 0]
        indices_1 = indices_0 | mask

        amp_0 = state[indices_0]
        amp_1 = state[indices_1]

        p0 = np.sum(np.abs(amp_0) ** 2)
        p1 = np.sum(np.abs(amp_1) ** 2)
        coherence = np.vdot(amp_0, amp_1)

        x = 2 * np.real(coherence)
        y = 2 * np.imag(coherence)
        z = p0 - p1

        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.clip(r, 0.0, 1.0)

        if r > eps:
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))
            phi = np.arctan2(y, x)
            if phi < 0: phi += 2 * np.pi
        else:
            theta, phi = 0.0, 0.0

        bloch_data['theta'].append(theta)
        bloch_data['phi'].append(phi)
        bloch_data['r'].append(r)
        bloch_data['mixedness'].append(1.0 - r)

    return bloch_data

def generate_topology_and_bloch_data(num_qubits, layers):
    """ Extrae el Grafo Bipartito (MPS) y la Esfera de Bloch (Local). """
    simulator_mps = AerSimulator(method="matrix_product_state")
    all_layers_data = []

    for layer_idx in range(layers + 1):
        qc = apply_random_circuit(num_qubits, layer_idx)
        qc.save_statevector()
        
        qc_comp = transpile(qc, backend=simulator_mps)
        result = simulator_mps.run(qc_comp).result()
        state = np.asarray(result.get_statevector(qc_comp))
        
        # 1. Extracción de Bloch Local
        bloch_data = get_bloch_data_from_statevector(state, num_qubits)
        
        # 2. Extracción de Topología Global (Grafo)
        G = nx.Graph()
        G.add_nodes_from(range(num_qubits))
        
        for i in range(num_qubits - 1):
            keep = [i, i + 1]
            trace_over = [j for j in range(num_qubits) if j not in keep]
            rho_ab = partial_trace(state, trace_over)
            
            s_a = entropy(partial_trace(rho_ab, [1]), base=2)
            s_b = entropy(partial_trace(rho_ab, [0]), base=2)
            s_ab = entropy(rho_ab, base=2)
            
            mut_info = max(0.0, s_a + s_b - s_ab)
            G.add_edge(i, i + 1, weight=mut_info)
            
        all_layers_data.append((layer_idx, G, bloch_data))
        
    return all_layers_data