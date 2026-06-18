import numpy as np
import random
from pathlib import Path
from simulations import generate_schmidt_spectrum_data
from sonification import export_schmidt_spectrum_to_wav

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    num_qubits = 15
    layers = 10

    print(f"Iniciando Análisis de Espectro de Schmidt para {num_qubits} qubits...")
    print("Rigor Físico: Distribución exacta de Autovalores de Entrelazamiento Bipartito")
    print("Rigor Acústico: Síntesis Espectral Inarmónica (Timbre Cuántico)")
    
    layer_data = generate_schmidt_spectrum_data(
        num_qubits=num_qubits,
        layers=layers
    )

    output_dir = Path("../outputs_schmidt_spectrum")
    output_dir.mkdir(exist_ok=True, parents=True)

    for layer_number, evals in layer_data:
        filename = output_dir / f"schmidt_layer_{layer_number:02d}.wav"
        export_schmidt_spectrum_to_wav(evals, filename)
        
        # Interpretación Teórica Impresa en Terminal
        print(f"Capa {layer_number:02d} | Rango de Entrelazamiento (Modos activos): {len(evals)}")

    print("Transmutación física completada exitosamente.")