import numpy as np
import random
from pathlib import Path
from simulations import generate_topology_and_bloch_data
from sonification import export_bloch_topology_to_wav

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    num_qubits = 15
    layers = 10

    print(f"Iniciando Análisis Comparativo (BFS + DFS + BLOCH) para {num_qubits} qubits...")
    
    layer_data = generate_topology_and_bloch_data(
        num_qubits=num_qubits,
        layers=layers
    )

    output_dir = Path("../outputs_bloch_topology")
    output_dir.mkdir(exist_ok=True, parents=True)

    for layer_number, graph, bloch_data in layer_data:
        filename = output_dir / f"bloch_topo_layer_{layer_number:02d}.wav"
        export_bloch_topology_to_wav(graph, bloch_data, filename)
        print(f"Capa {layer_number:02d} renderizada.")

    print("Sonificación Local (Bloch) completada para contraste.")