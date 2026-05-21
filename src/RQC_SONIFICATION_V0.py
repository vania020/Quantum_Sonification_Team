import numpy as np
import random
from simulations import generate_bloch_audio_data_by_layer
from sonification import export_bloch_to_wav
from pathlib import Path


# ============================================================
# 8. BLOQUE PRINCIPAL
# ============================================================

if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)

    num_qubits = 15
    layers = 10

    print(f"Generando circuito RQC de {num_qubits} qubits y {layers} capas...")
    print(f"Se generarán {layers + 1} archivos WAV.")
    print("Mapeo usado:")
    print("theta -> frecuencia")
    print("phi -> paneo estéreo")
    print("r -> amplitud")
    print("1-r -> textura/modulación")

    layer_data = generate_bloch_audio_data_by_layer(
        num_qubits=num_qubits,
        layers=layers,
        use_mps=False
    )

    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)

    for layer_number, bloch_data in layer_data:

        if layer_number == 0:
            filename = output_dir / "bloch_layer_00_initial.wav"
        else:
            filename = output_dir / f"bloch_layer_{layer_number:02d}.wav"

        export_bloch_to_wav(bloch_data, filename)

    print("Proceso terminado correctamente.")
