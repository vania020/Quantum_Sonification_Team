import numpy as np
import random
from pathlib import Path
from simulations import generate_unified_quantum_data
from sonification import export_trinity_architecture_to_wav

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    num_qubits = 15
    layers = 10

    print(f"Iniciando Transmutación Trinitaria (BFS + DFS + QFT) para {num_qubits} qubits...")
    
    layer_data = generate_unified_quantum_data(
        num_qubits=num_qubits,
        layers=layers
    )

    output_dir = Path("../outputs_trinity_master")
    output_dir.mkdir(exist_ok=True, parents=True)

    for layer_number, graph, statevector in layer_data:
        filename = output_dir / f"trinity_layer_{layer_number:02d}.wav"
        export_trinity_architecture_to_wav(graph, statevector, filename)
        print(f"Capa {layer_number:02d} renderizada.")

    print("Obra de Vanguardia Cuántica completada.")