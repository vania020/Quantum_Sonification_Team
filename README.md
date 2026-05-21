# Quantum Sonification Team

A research-driven project for the **sonification of Random Quantum Circuits (RQC)**.

This repository explores how quantum information can be transformed into sound. The current model extracts local Bloch-sphere data from a multiqubit random quantum circuit and maps it into stereo audio.

---

## Overview

The goal is to create an audio representation of quantum circuit evolution, including:

- local qubit behavior
- phase evolution
- superposition dynamics
- mixedness caused by multiqubit interaction
- layer-by-layer changes in an RQC

---

## Pipeline

```text
Random Quantum Circuit
        ↓
Statevector Simulation
        ↓
Local Bloch Data Extraction
        ↓
Quantum-to-Audio Mapping
        ↓
Layer-by-Layer WAV Export
````
For each circuit layer, the code extracts local Bloch information from every qubit and generates an audio file.


---

## Sonification Model

```text
theta      → frequency
phi        → stereo panning
r          → amplitude
mixedness  → texture/modulation
```

Where:

* `theta` is the polar angle.
* `phi` is the azimuthal angle.
* `r` is the local Bloch vector length.
* `mixedness = 1 - r` measures how far the local qubit state is from pure.

Where:

* theta is the polar angle of the local Bloch vector.
* phi is the azimuthal angle.
* r is the length of the local Bloch vector.
* mixedness = 1 - r indicates how far the local qubit state is from a pure state.

---

## Repository Structure

```text
Quantum_Sonification_Team/
│
├── README.md
├── notebooks/
├── outputs/
├── experiments/
│   ├── rqc_dayana_prob.py
│   ├── rqc_prototipo.py
│   ├── rqc_rocio.py
│   └── rqc_valentino_ampli.py
│
└── src/
    ├── gates.py
    ├── circuit.py
    ├── simulations.py
    ├── sonification.py
    └── rqc_sonification.py
```

---

## Main Modules

* `gates.py`: defines the √X, √Y, √W, and fSim gates.
* `circuit.py`: builds the Random Quantum Circuit layer by layer.
* `simulations.py`: simulates the circuit and extracts local Bloch data.
* `sonification.py`: maps Bloch data into stereo audio.
* `rqc_sonification.py`: main script that generates one WAV file per layer.

---

## Experiments

The `experiments/` folder contains previous or alternative versions of the project, including probability-based, amplitude-based, and prototype sonification models.

---

## Technologies

* Python
* Qiskit
* Qiskit Aer
* NumPy
* SciPy
* WAV audio synthesis

---

## Installation

```bash
pip install numpy scipy qiskit qiskit-aer
```

---

## How to Run

From the repository root:

```bash
cd src
python rqc_sonification.py
```

Generated audio files are saved in:

```text
outputs/
```

---

## Research Direction

This repository investigates how quantum circuit dynamics can be translated into sound in a computationally grounded and perceptually interpretable way.

Future directions include:

* animated Bloch-sphere visualization synchronized with sound
* interactive exploration of quantum circuits
* interface to listen to and visualize quantum evolution

---

## Contributors

Quantum Sonification Team


