# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import welch


def calculate_fft(
        signal,
        fs,
        calibration_db=0.0,
        nperseg=None):

    signal = np.asarray(signal, dtype=float)

    if signal.dtype.kind in ["i", "u"]:
        signal = signal / np.iinfo(signal.dtype).max

    if nperseg is None:
        nperseg = min(
            len(signal),
            fs
        )

    freq, pxx = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        scaling="spectrum",
        detrend=False
    )

    level = (
        10*np.log10(
            pxx + 1e-30
        )
        +
        calibration_db
    )

    return freq, level


def decimate_for_plot(
        freq,
        level,
        max_points=8000):

    if len(freq) <= max_points:
        return freq, level

    step = max(
        1,
        int(len(freq)/max_points)
    )

    return (
        freq[::step],
        level[::step]
    )


def get_dominant_frequency(
        freq,
        level,
        fmin=20,
        fmax=20000):

    mask = (
        (freq >= fmin)
        &
        (freq <= fmax)
    )

    freq2 = freq[mask]
    level2 = level[mask]

    if len(freq2) == 0:
        return None

    idx = np.argmax(level2)

    return float(freq2[idx])


def calculate_raw_fft(
        signal,
        fs,
        calibration_db=0.0):

    signal = np.asarray(signal, dtype=float)

    if signal.dtype.kind in ["i", "u"]:
        signal = signal / np.iinfo(signal.dtype).max

    
    n = min(
        len(signal),
        262144
    )

    signal = signal[:n]

    window = np.hanning(n)

    fft_values = np.fft.rfft(
        signal * window
    )

    freq = np.fft.rfftfreq(
        n,
        d=1/fs
    )

    magnitude = (
        np.abs(fft_values)
        / np.sum(window)
    )

    level = (
        20 * np.log10(
            magnitude + 1e-30
        )
        + calibration_db
    )

    return freq, level