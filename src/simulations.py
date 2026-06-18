# ============================================================
# Añadir a simulations.py (Negatividad y Criterio de Peres)
# ============================================================
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace
from circuit import apply_random_circuit

def get_partial_transpose_2q(rho_4x4):
    """
    Realiza una transposición parcial sobre el primer qubit (Subsistema A)
    de una matriz de densidad de 2 qubits (4x4) de forma matemática estricta
    sin depender de versiones de Qiskit.
    """
    # Remodelamos la matriz 4x4 a un tensor 2x2x2x2
    tensor = rho_4x4.reshape((2, 2, 2, 2))
    # Transponemos los índices del primer qubit (0 y 1 -> 1 y 0)
    tensor_pt = tensor.transpose((1, 0, 2, 3))
    return tensor_pt.reshape((4, 4))

def generate_wigner_shadows_data(num_qubits, layers):
    """
    Extrae los autovalores reales y "fantasmas" (negativos) de la 
    transposición parcial para cada par de qubits adyacentes.
    """
    simulator = AerSimulator(method="statevector")
    all_layers_eigenvalues = []

    for layer_idx in range(layers + 1):
        qc = apply_random_circuit(num_qubits, layer_idx)
        qc.save_statevector()
        
        qc_comp = transpile(qc, backend=simulator)
        result = simulator.run(qc_comp).result()
        state = result.get_statevector(qc_comp)
        
        layer_pairs = []
        
        for i in range(num_qubits - 1):
            keep = [i, i + 1]
            trace_over = [j for j in range(num_qubits) if j not in keep]
            
            # Matriz de densidad 4x4 del par
            rho_ab = partial_trace(state, trace_over).data
            
            # Aplicamos la Transposición Parcial
            rho_ab_pt = get_partial_transpose_2q(rho_ab)
            
            # Calculamos los autovalores
            eigenvalues = np.linalg.eigvals(rho_ab_pt)
            
            # Los autovalores de la transposición parcial son siempre reales
            eigenvalues = np.real(eigenvalues)
            
            # Separamos la realidad (positivos) de la sombra (negativos)
            pos_evals = [ev for ev in eigenvalues if ev > 1e-10]
            neg_evals = [ev for ev in eigenvalues if ev < -1e-10]
            
            layer_pairs.append({
                "pair": (i, i+1),
                "positive": pos_evals,
                "negative": neg_evals
            })
            
        all_layers_eigenvalues.append((layer_idx, layer_pairs))
        
    return all_layers_eigenvalues