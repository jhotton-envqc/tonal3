# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.signal import welch


THIRD_OCTAVE_BANDS = np.array([
    6.3,
    8.0,
    10.0,
    12.5,
    16.0,
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0
])


def third_octave_limits(fc):
    """
    Limites d'une bande de tiers d'octave.
    """
    factor = 2 ** (1 / 6)

    return (
        fc / factor,
        fc * factor
    )


def wav_to_third_octaves(
        signal,
        fs,
        calibration_db=0.0):
    """
    Calcule les bandes de tiers d'octave
    à partir du signal WAV.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if signal.dtype.kind in ["i", "u"]:
        signal = signal / np.iinfo(
            signal.dtype
        ).max

    freq, pxx = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(
            fs,
            len(signal)
        ),
        scaling="spectrum"
    )

    rows = []

    for fc in THIRD_OCTAVE_BANDS:

        fl, fu = third_octave_limits(fc)

        idx = np.where(
            (freq >= fl) &
            (freq < fu)
        )[0]

        if len(idx) == 0:
            continue

        energy = np.sum(
            pxx[idx]
        )

        level = (
            10 * np.log10(
                energy + 1e-30
            )
            + calibration_db
        )

        rows.append(
            {
                "frequency": fc,
                "band_level": level
            }
        )

    return pd.DataFrame(rows)


def estimate_calibration_factor(
        signal,
        fs,
        measured_octaves):
    """
    Estime le facteur de calibration
    du WAV à partir des bandes mesurées.

    Parameters
    ----------
    signal : ndarray

    fs : int

    measured_octaves : DataFrame
        Colonnes:
        - frequency
        - band_level

    Returns
    -------
    calibration_db
    comparison_df
    """

    wav_octaves = wav_to_third_octaves(
        signal,
        fs,
        calibration_db=0.0
    )

    merged = wav_octaves.merge(
        measured_octaves,
        on="frequency",
        suffixes=(
            "_wav",
            "_measured"
        )
    )

    if merged.empty:
        raise ValueError(
            "Aucune fréquence commune trouvée."
        )

    merged["delta_db"] = (
        merged["band_level_measured"]
        -
        merged["band_level_wav"]
    )

    calibration_bands = merged[
        (merged["frequency"] >= 200)
        &
        (merged["frequency"] <= 6300)
    ].copy()

    calibration_db = float(
        calibration_bands["delta_db"].median()
    )

    calibration_bands["residual_db"] = (
        calibration_bands["delta_db"]
        -
        calibration_db
    )

    stats = {
        "n_bands":
            int(len(calibration_bands)),

        "median_delta":
            float(
                calibration_bands["delta_db"].median()
            ),

        "mean_delta":
            float(
                calibration_bands["delta_db"].mean()
            ),

        "std_delta":
            float(
                calibration_bands["delta_db"].std()
            ),

        "max_residual":
            float(
                np.abs(
                    calibration_bands["residual_db"]
                ).max()
            )
    }

    return (
        calibration_db,
        calibration_bands,
        stats
    )
