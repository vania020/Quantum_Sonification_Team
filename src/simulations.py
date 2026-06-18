import numpy as np
import networkx as nx
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace, entropy
from circuit import apply_random_circuit, apply_random_circuit_with_qft

def generate_unified_quantum_data(num_qubits, layers):
    simulator_mps = AerSimulator(method="matrix_product_state")
    simulator_sv = AerSimulator(method="statevector")
    
    all_layers_data = []

    for layer_idx in range(layers + 1):
        # 1. Extracción Topológica (Grafo)
        qc_base = apply_random_circuit(num_qubits, layer_idx)
        qc_mps = qc_base.copy()
        qc_mps.save_statevector()
        
        qc_mps_comp = transpile(qc_mps, backend=simulator_mps)
        result_mps = simulator_mps.run(qc_mps_comp).result()
        state_mps = result_mps.get_statevector(qc_mps_comp)
        
        G = nx.Graph()
        G.add_nodes_from(range(num_qubits))
        
        for i in range(num_qubits - 1):
            keep = [i, i + 1]
            trace_over = [j for j in range(num_qubits) if j not in keep]
            rho_ab = partial_trace(state_mps, trace_over)
            
            s_a = entropy(partial_trace(rho_ab, [1]), base=2)
            s_b = entropy(partial_trace(rho_ab, [0]), base=2)
            s_ab = entropy(rho_ab, base=2)
            
            mut_info = max(0.0, s_a + s_b - s_ab)
            G.add_edge(i, i + 1, weight=mut_info)
            
        # 2. Extracción Espectral (QFT)
        qc_qft = apply_random_circuit_with_qft(num_qubits, layer_idx)
        qc_qft.save_statevector()
        
        qc_qft_comp = transpile(qc_qft, backend=simulator_sv)
        result_qft = simulator_sv.run(qc_qft_comp).result()
        state_qft = np.asarray(result_qft.get_statevector(qc_qft_comp))
        
        all_layers_data.append((layer_idx, G, state_qft))
        
    return all_layers_data