# ============================================================
# simulations.py (NUEVO: Extracción del Espectro de Schmidt)
# ============================================================
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace
from circuit import apply_random_circuit

def generate_schmidt_spectrum_data(num_qubits, layers):
    """
    Calcula el Espectro de Entrelazamiento exacto.
    Divide el sistema a la mitad y extrae todos los autovalores 
    (coeficientes de Schmidt al cuadrado) de la matriz de densidad reducida.
    """
    simulator = AerSimulator(method="statevector")
    all_layers_spectra = []
    
    # Corte bipartito exacto: Trazamos la mitad derecha del procesador
    trace_over_B = list(range(num_qubits // 2, num_qubits))

    for layer_idx in range(layers + 1):
        qc = apply_random_circuit(num_qubits, layer_idx)
        qc.save_statevector()
        
        qc_comp = transpile(qc, backend=simulator)
        result = simulator.run(qc_comp).result()
        state = result.get_statevector(qc_comp)
        
        # 1. Matriz de Densidad Reducida de la mitad izquierda
        rho_A = partial_trace(state, trace_over_B)
        
        # 2. Diagonalización rigurosa para obtener los autovalores de Schmidt
        evals = np.real(np.linalg.eigvals(rho_A.data))
        
        # 3. Ordenamos de mayor a menor y filtramos el ruido de precisión de máquina
        evals = np.sort(evals)[::-1]
        evals = evals[evals > 1e-8]
        
        all_layers_spectra.append((layer_idx, evals))
        
    return all_layers_spectra