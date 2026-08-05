# Quantum Sonification Team — Paper Branch

This branch isolates the scientifically testable core of the project:

1. local one-qubit geometry through Bloch descriptors;
2. all-pairs quantum mutual information as a correlation graph;
3. deterministic BFS/DFS sonification over a maximum-spanning forest;
4. explicit metadata and STFT-ready audio outputs.

**QFT and spectral-state sonification are intentionally excluded from this branch.** They remain a separate future experiment.

## Scientific mapping

| Quantum descriptor | Acoustic parameter |
|---|---|
| Bloch polar angle `theta` | logarithmic frequency |
| Bloch azimuth `phi` | equal-power stereo pan |
| Bloch radius `r` | amplitude with a fixed floor |
| normalized linear entropy `1-r^2` | deterministic AM depth |
| pairwise mutual information `I(i:j)` | event duration on a fixed 0–2 bit scale |
| maximum-spanning forest | traversal backbone only |

The full all-pairs mutual-information matrix is saved for quantitative analysis. The spanning forest is not treated as the scientific data; it is only a deterministic and connected-as-possible sonification backbone.

## Key methodological safeguards

- one seed produces one prefix-consistent circuit trajectory;
- no stochastic noise is injected into the audio;
- BFS and DFS are exported separately;
- the combined audio is not loop-padded and is only a listening aid;
- WAV files are not normalized independently, preserving amplitude comparability;
- zero-information edges are not added;
- layer metrics, MI matrices, audio configuration and segment boundaries are saved.

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
cd src
python rqc_sonification.py --num-qubits 15 --layers 10 --seed 42
```

For a dataset, repeat across independent seeds:

```bash
python rqc_sonification.py --num-qubits 15 --layers 10 --seed 1
python rqc_sonification.py --num-qubits 15 --layers 10 --seed 2
python rqc_sonification.py --num-qubits 15 --layers 10 --seed 3
```

## Tests

From the repository root:

```bash
pytest -q
```

## STFT features

```bash
python analysis/extract_stft_features.py outputs/paper_bloch_graph/seed_000042/audio \
  --output outputs/paper_bloch_graph/seed_000042/audio_features.csv
```

## Scope of claims

The code measures total pairwise correlation through quantum mutual information. It does **not** claim that every graph edge is entanglement, and it does not claim that BFS/DFS are quantum observables. They are deterministic navigation rules applied after the quantum descriptors have been computed.
