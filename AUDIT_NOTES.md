# Audit notes for the uploaded branches

## Recommended lineage

Create `paper/bloch-mi-graph` from the QFT-comparison branch, then remove QFT and spectral modes. That branch contains the best reproducibility work: local seeded circuit construction, deterministic Bloch synthesis, dimension checks and a shared renderer.

## Critical findings

1. `feature-grafo-bloch-compare/src/rqc_sonification.py` imports functions that do not exist in that branch. The actual working entry point is `rqc_bloch_master.py`, while the README still tells users to execute `rqc_sonification.py`.
2. `feature-grafo-bloch-compare` and `feature-grafo-mps` rebuild every layer with the global RNG. Consecutive layer indices therefore do not represent prefixes of one evolving circuit.
3. The graph branches only compute nearest-neighbour mutual information. The resulting graph is a chain, so BFS/DFS topology is almost fixed by qubit index rather than learned from the quantum state.
4. Zero-weight edges are retained, including at the initial product state. This creates a connected traversal even when no correlations exist.
5. In `feature-grafo-bloch-compare`, `phi` controls FM index but stereo panning is determined by node ID, contradicting the README mapping.
6. The same branch calls `1-r` “local entropy” and injects unseeded Gaussian noise. Both weaken physical correctness and reproducibility.
7. Per-file peak normalization destroys between-layer amplitude information even though Bloch radius is mapped to amplitude.
8. BFS and DFS are loop-tiled to equal length before mixing. The repeated material is an audio artifact, not quantum information.
9. The MPS branch requests a full statevector from an MPS simulation, largely forfeiting the memory advantage of retaining the state in MPS form.
10. No tests, environment lock, structured metadata, MI matrices or analysis-ready segment boundaries are present.

## Changes implemented in this clean branch

- QFT and spectral code removed;
- prefix-consistent circuit seeds;
- all-pairs MI matrix;
- positive-edge graph plus maximum-spanning forest for traversal;
- normalized linear entropy and von Neumann entropy;
- direct, monotonic Bloch-to-audio mapping;
- deterministic audio;
- separate BFS/DFS outputs;
- zero-padding rather than loop-padding;
- fixed gains rather than per-file normalization;
- metrics CSV, MI matrices, metadata JSON and segment CSV;
- STFT feature extractor;
- basic physics and determinism tests.
