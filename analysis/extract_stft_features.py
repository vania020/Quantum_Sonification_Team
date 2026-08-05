"""Extract transparent STFT-based features from generated WAV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


def safe_weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int = 0) -> np.ndarray:
    denominator = np.sum(weights, axis=axis)
    numerator = np.sum(values * weights, axis=axis)
    return np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def extract_features(path: Path, nperseg: int = 2048, noverlap: int = 1536) -> dict[str, float]:
    sample_rate, audio = wavfile.read(path)
    audio = np.asarray(audio, dtype=float)
    if audio.ndim == 1:
        left = right = audio
    else:
        left, right = audio[:, 0], audio[:, 1]
    mono = 0.5 * (left + right)

    frequencies, _, spectrum = stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=min(nperseg, max(8, len(mono))),
        noverlap=min(noverlap, max(0, min(nperseg, len(mono)) - 1)),
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum) ** 2
    frequency_grid = frequencies[:, None]
    centroid = safe_weighted_mean(frequency_grid, power, axis=0)
    bandwidth = np.sqrt(
        safe_weighted_mean((frequency_grid - centroid[None, :]) ** 2, power, axis=0)
    )

    normalized_power = np.divide(
        power,
        np.sum(power, axis=0, keepdims=True),
        out=np.zeros_like(power),
        where=np.sum(power, axis=0, keepdims=True) > 0,
    )
    spectral_entropy = -np.sum(
        normalized_power * np.log2(normalized_power + 1e-15), axis=0
    )
    spectral_flux = np.sqrt(np.sum(np.diff(normalized_power, axis=1) ** 2, axis=0))

    rms = float(np.sqrt(np.mean(mono**2)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(mono))))) if len(mono) > 1 else 0.0
    rms_left = float(np.sqrt(np.mean(left**2)))
    rms_right = float(np.sqrt(np.mean(right**2)))
    stereo_balance = (rms_right - rms_left) / (rms_right + rms_left + 1e-15)
    stereo_correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else 0.0

    return {
        "duration_sec": float(len(mono) / sample_rate),
        "rms": rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid_mean_hz": float(np.mean(centroid)),
        "spectral_centroid_std_hz": float(np.std(centroid)),
        "spectral_bandwidth_mean_hz": float(np.mean(bandwidth)),
        "spectral_bandwidth_std_hz": float(np.std(bandwidth)),
        "spectral_entropy_mean": float(np.mean(spectral_entropy)),
        "spectral_entropy_std": float(np.std(spectral_entropy)),
        "spectral_flux_mean": float(np.mean(spectral_flux)) if len(spectral_flux) else 0.0,
        "stereo_balance": float(stereo_balance),
        "stereo_correlation": stereo_correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("audio_features.csv"))
    args = parser.parse_args()

    files = sorted(args.audio_dir.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found under {args.audio_dir}")

    rows = [{"path": str(path), **extract_features(path)} for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
