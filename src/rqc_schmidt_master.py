import numpy as np
import random
from pathlib import Path
from simulations import generate_schmidt_spectrum_data
from sonification import export_schmidt_sci_fi_to_wav

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    num_qubits = 15
    layers = 10

    print(f"Iniciando Extracción del Espectro de Schmidt para {num_qubits} qubits...")
    
    spectrum_data = generate_schmidt_spectrum_data(
        num_qubits=num_qubits,
        layers=layers
    )

    output_dir = Path("../outputs_schmidt_scifi")
    output_dir.mkdir(exist_ok=True, parents=True)

    for layer_number, eigenvalues in spectrum_data:
        filename = output_dir / f"schmidt_sci_fi_layer_{layer_number:02d}.wav"
        export_schmidt_sci_fi_to_wav(eigenvalues, filename)
        num_states = len(eigenvalues)
        print(f"Capa {layer_number:02d} | Estados activos en la mezcla: {num_states} -> Renderizada.")

    print("Atmósfera Sci-Fi Cuántica completada.")