"""Command-line entry point for the paper-ready Bloch + MI graph pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from simulations import generate_bloch_graph_data
from sonification import (
    DEFAULT_CONFIG,
    concatenate_with_gaps,
    render_layer_audio,
    write_wav,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=15)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=int, default=None)
    parser.add_argument("--edge-tolerance", type=float, default=1e-10)
    parser.add_argument("--output-dir", type=Path, default=Path("../outputs/paper_bloch_graph"))
    return parser.parse_args()


def write_layer_metrics(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "num_qubits", "layer", *records[0]["metrics"].keys()]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "seed": record["seed"],
                    "num_qubits": record["num_qubits"],
                    "layer": record["layer"],
                    **record["metrics"],
                }
            )


def write_segments(boundaries, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["layer", "start_sec", "end_sec"])
        writer.writeheader()
        for layer, (start, end) in enumerate(boundaries):
            writer.writerow({"layer": layer, "start_sec": start, "end_sec": end})


def main() -> None:
    args = parse_args()
    preferred_root = args.num_qubits // 2 if args.root is None else args.root
    if not 0 <= preferred_root < args.num_qubits:
        raise ValueError("root must be a valid qubit index")

    run_dir = args.output_dir / f"seed_{args.seed:06d}"
    audio_root = run_dir / "audio"
    matrix_root = run_dir / "mutual_information_matrices"
    matrix_root.mkdir(parents=True, exist_ok=True)

    layer_data = generate_bloch_graph_data(
        num_qubits=args.num_qubits,
        layers=args.layers,
        seed=args.seed,
        edge_tolerance=args.edge_tolerance,
    )

    rendered_by_mode = {"bfs": [], "dfs": [], "combined": []}
    metric_records = []

    for item in layer_data:
        layer = int(item["layer"])
        audio = render_layer_audio(
            item["sonification_backbone"],
            item["bloch"],
            preferred_root=preferred_root,
            config=DEFAULT_CONFIG,
        )

        for mode, signal in audio.items():
            write_wav(
                audio_root / mode / f"layer_{layer:02d}.wav",
                signal,
                int(DEFAULT_CONFIG["sample_rate"]),
            )
            rendered_by_mode[mode].append(signal)

        np.save(
            matrix_root / f"mi_layer_{layer:02d}.npy",
            item["mutual_information_matrix"],
        )
        metric_records.append(
            {
                "seed": args.seed,
                "num_qubits": args.num_qubits,
                "layer": layer,
                "metrics": item["metrics"],
            }
        )

    write_layer_metrics(metric_records, run_dir / "layer_metrics.csv")

    for mode, chunks in rendered_by_mode.items():
        continuous, boundaries = concatenate_with_gaps(
            chunks,
            gap_sec=float(DEFAULT_CONFIG["layer_gap_sec"]),
            sample_rate=int(DEFAULT_CONFIG["sample_rate"]),
        )
        write_wav(
            audio_root / f"{mode}_continuous.wav",
            continuous,
            int(DEFAULT_CONFIG["sample_rate"]),
        )
        write_segments(boundaries, audio_root / f"{mode}_segments.csv")

    metadata = {
        "scientific_scope": "Bloch descriptors plus all-pairs mutual-information graph",
        "num_qubits": args.num_qubits,
        "layers": args.layers,
        "seed": args.seed,
        "preferred_root": preferred_root,
        "edge_tolerance": args.edge_tolerance,
        "audio_config": DEFAULT_CONFIG,
        "mapping": {
            "theta": "log-frequency",
            "phi": "equal-power stereo pan",
            "bloch_radius": "amplitude with fixed floor",
            "normalized_linear_entropy": "deterministic amplitude-modulation depth",
            "mutual_information": "event duration using the fixed 0-to-2-bit range",
            "graph": "all-pairs MI for analysis; maximum-spanning forest for traversal",
        },
        "qft_enabled": False,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Completed paper pipeline: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
