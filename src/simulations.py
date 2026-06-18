# ============================================================
# simulations.py (Rama: Espectro de Schmidt / Autovalores)
# ============================================================
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace
from circuit import apply_random_circuit

def generate_schmidt_spectrum_data(num_qubits, layers):
    """
    Realiza el corte bipartito y extrae los autovalores exactos 
    (Espectro de Schmidt al cuadrado) de la matriz de densidad reducida.
    """
    simulator = AerSimulator(method="statevector")
    spectrum_data = []

    # Corte bipartito (Mitad derecha trazada)
    trace_over_B = list(range(num_qubits // 2, num_qubits))
    
    for layer_idx in range(layers + 1):
        qc = apply_random_circuit(num_qubits, layer_idx)
        qc.save_statevector()
        
        qc_comp = transpile(qc, backend=simulator)
        result = simulator.run(qc_comp).result()
        state = result.get_statevector(qc_comp)
        
        rho_A = partial_trace(state, trace_over_B)
        
        # 1. Extraer los autovalores de la matriz rho_A
        eigenvalues = np.real(np.linalg.eigvals(rho_A.data))
        
        # 2. Ordenar de mayor a menor y filtrar el "polvo" numérico cuántico
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-6]
        
        spectrum_data.append((layer_idx, eigenvalues))
        
    return spectrum_data