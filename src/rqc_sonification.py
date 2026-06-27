import csv
import random
from pathlib import Path

import numpy as np

from simulations import generate_fair_comparison_data
from sonification import export_layer_audio, DEFAULT_CONFIG


NUM_QUBITS = 15
LAYERS = 10
SEED = 42

# Render all three to separate folders.
# For the strict QFT experiment, compare spectral_base vs spectral_qft.
# For representation comparison, compare bloch vs spectral_qft.
MODES_TO_RENDER = ["bloch", "spectral_base", "spectral_qft"]

OUTPUT_DIR = Path("../outputs_fair_qft_comparison")

CONFIG = DEFAULT_CONFIG.copy()
CONFIG.update({
    "top_k": 32,
    "spectral_min_probability": 1e-4,
    "bfs_gain": 0.6,
    "dfs_gain": 1.0,
    "bfs_attack_fraction": 0.30,
    "dfs_attack_fraction": 0.05,
})


def write_metrics_csv(layer_data, output_dir):
    csv_path = output_dir / "qft_comparison_metrics.csv"

    fieldnames = [
        "layer",
        "base_top_k_mass",
        "qft_top_k_mass",
        "delta_top_k_mass",
        "base_shannon_entropy",
        "qft_shannon_entropy",
        "delta_shannon_entropy",
        "base_participation_ratio",
        "qft_participation_ratio",
        "delta_participation_ratio",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in layer_data:
            mb = item["metrics_base"]
            mq = item["metrics_qft"]
            writer.writerow({
                "layer": item["layer"],
                "base_top_k_mass": mb["top_k_mass"],
                "qft_top_k_mass": mq["top_k_mass"],
                "delta_top_k_mass": mq["top_k_mass"] - mb["top_k_mass"],
                "base_shannon_entropy": mb["shannon_entropy"],
                "qft_shannon_entropy": mq["shannon_entropy"],
                "delta_shannon_entropy": mq["shannon_entropy"] - mb["shannon_entropy"],
                "base_participation_ratio": mb["participation_ratio"],
                "qft_participation_ratio": mq["participation_ratio"],
                "delta_participation_ratio": mq["participation_ratio"] - mb["participation_ratio"],
            })

    return csv_path


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("============================================================")
    print("RQC FAIR COMPARISON: Bloch vs Spectral base vs Spectral QFT")
    print("============================================================")
    print(f"num_qubits = {NUM_QUBITS}")
    print(f"layers     = {LAYERS}")
    print(f"seed       = {SEED}")
    print(f"modes      = {MODES_TO_RENDER}")
    print("No stochastic noise is injected in any mode.")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("Generating quantum data...")
    layer_data = generate_fair_comparison_data(
        num_qubits=NUM_QUBITS,
        layers=LAYERS,
        seed=SEED,
        qft_do_swaps=True,
        top_k=CONFIG["top_k"],
    )

    metrics_path = write_metrics_csv(layer_data, OUTPUT_DIR)
    print(f"Metrics written to: {metrics_path}")
    print()

    for mode in MODES_TO_RENDER:
        mode_dir = OUTPUT_DIR / mode
        mode_dir.mkdir(exist_ok=True, parents=True)
        print(f"Rendering mode: {mode}")

        for item in layer_data:
            layer = item["layer"]
            filename = mode_dir / f"{mode}_layer_{layer:02d}.wav"
            export_layer_audio(
                graph=item["graph"],
                layer_payload=item,
                filename=filename,
                mode=mode,
                config=CONFIG,
            )
            print(f"  layer {layer:02d} -> {filename}")
        print()

    print("Done.")
    print("Strict QFT test: spectral_base vs spectral_qft")
    print("Representation comparison: bloch vs spectral_qft")


if __name__ == "__main__":
    main()
