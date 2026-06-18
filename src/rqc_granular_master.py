import numpy as np
import random
from pathlib import Path
from simulations import generate_macroscopic_entropy_data
from sonification import export_granular_entropy_to_wav

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    num_qubits = 15
    layers = 10

    print(f"Iniciando Análisis Termodinámico para {num_qubits} qubits...")
    print("Mapeo Granular: Entropía Bipartita S(rho_A) -> Densidad y Jitter Acústico")
    
    entropy_data = generate_macroscopic_entropy_data(
        num_qubits=num_qubits,
        layers=layers
    )

    output_dir = Path("../outputs_granular_entropy")
    output_dir.mkdir(exist_ok=True, parents=True)

    for layer_number, S_val in entropy_data:
        filename = output_dir / f"granular_layer_{layer_number:02d}.wav"
        export_granular_entropy_to_wav(S_val, filename)
        print(f"Capa {layer_number:02d} | Entropía: {S_val:.4f} bits -> Renderizada.")

    print("Colapso Termodinámico completado.")