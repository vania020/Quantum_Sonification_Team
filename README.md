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
