import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace, entropy
from circuit import apply_random_circuit

def generate_macroscopic_entropy_data(num_qubits, layers):
    simulator = AerSimulator(method="statevector")
    entropy_data = []

    # Corte físico: Si tenemos 15 qubits, trazamos la mitad derecha
    trace_over_B = list(range(num_qubits // 2, num_qubits))
    
    for layer_idx in range(layers + 1):
        qc = apply_random_circuit(num_qubits, layer_idx)
        qc.save_statevector()
        
        qc_comp = transpile(qc, backend=simulator)
        result = simulator.run(qc_comp).result()
        state = result.get_statevector(qc_comp)
        
        # 1. Matriz de Densidad Reducida del subsistema A
        rho_A = partial_trace(state, trace_over_B)
        
        # 2. Calcular la Entropía S = -Tr(rho * log2(rho))
        S_A = abs(entropy(rho_A, base=2))
        
        entropy_data.append((layer_idx, S_A))
        
    return entropy_data