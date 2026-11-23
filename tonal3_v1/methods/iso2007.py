# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.signal import welch


def critical_band(fc):
    """
    ISO 1996-2:2007 Annexe C

    Bande critique :

    <= 500 Hz : 100 Hz
    > 500 Hz : 20 % de fc
    """

    if fc <= 500:
        bandwidth = 100.0
    else:
        bandwidth = 0.20 * fc

    lower = fc - bandwidth / 2
    upper = fc + bandwidth / 2

    return lower, upper, bandwidth


def tonal_audibility(
        Lpt,
        Lpn,
        fc):
    """
    Eq. C.3
    """

    return (
        Lpt
        - Lpn
        + 2.0
        + np.log10(
            1.0 + (fc / 502.0) ** 2.5
        )
    )


def tonal_adjustment(delta_Lta):
    """
    Eq. C.4 à C.6
    """

    if delta_Lta > 10.0:
        return 6.0

    if delta_Lta < 4.0:
        return 0.0

    return delta_Lta - 4.0


def analyze_iso2007(
        signal,
        fs,
        calibration_db=0.0):
    """
    ISO 1996-2:2007 simplifiée

    Retour :
        result_df
        summary
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if signal.dtype.kind in ["i", "u"]:
        signal = (
            signal
            / np.iinfo(signal.dtype).max
        )

    nperseg = min(
        len(signal),
        fs
    )

    freq, pxx = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        window="hann",
        scaling="spectrum",
        detrend=False
    )

    level = (
        10 * np.log10(
            pxx + 1e-30
        )
        + calibration_db
    )

    rows = []

    for i in range(2, len(freq) - 2):

        fc = float(freq[i])

        if fc < 50:
            continue

        if fc > 10000:
            continue

        current = level[i]

        local_noise = np.mean([
            level[i - 2],
            level[i - 1],
            level[i + 1],
            level[i + 2]
        ])

        # Le pic doit dépasser le bruit local d'au moins 3 dB
        if current - local_noise < 3:
            continue

        cb_low, cb_high, cb_width = (
            critical_band(fc)
        )

        mask = (
            (freq >= cb_low)
            &
            (freq <= cb_high)
            &
            (np.abs(freq - fc) > 10)
        )


        if np.sum(mask) < 5:
            continue

# ==========test==========
#
        if 240 <= fc <= 260:

            print(
                f"fc={fc:.1f}"
            )

            print(
                f"Nb bins={len(level[mask])}"
            )

            print(
                f"min={np.min(level[mask]):.1f}"
            )

            print(
                f"max={np.max(level[mask]):.1f}"
            )

            print(
                f"p90={np.percentile(level[mask],90):.1f}"
            )
#
#=========================

        # Bruit de masquage
        noise_levels = level[mask]



        # niveau représentatif du bruit de fond
        Lpn = np.percentile(
            noise_levels,
            90
        )

        tone_mask = (
            (freq >= fc - 5)
            &
            (freq <= fc + 5)
        )

        Lpt = (
            10 * np.log10(
                np.sum(
                    10 ** (
                        level[tone_mask] / 10
                    )
                )
            )
        )

        delta_Lta = tonal_audibility(
            Lpt,
            Lpn,
            fc
        )

        kt = tonal_adjustment(
            delta_Lta
        )

        rows.append(
            {
                "frequency": fc,
                "Lpt": Lpt,
                "Lpn": Lpn,
                "delta_Lta": delta_Lta,
                "critical_band": cb_width,
                "kt": kt
            }
        )

    # =====================================================
    # AUCUN CANDIDAT
    # =====================================================

    if len(rows) == 0:

        summary = {
            "frequency": np.nan,
            "Lpt": np.nan,
            "Lpn": np.nan,
            "delta_Lta": 0.0,
            "kt": 0.0,
            "n_candidates": 0
        }

        return (
            pd.DataFrame(),
            summary
        )

    # =====================================================
    # MEILLEUR CANDIDAT
    # =====================================================

    result_df = pd.DataFrame(rows)

    # Trier du plus audible au moins audible
    result_df = result_df.sort_values(
        "delta_Lta",
        ascending=False
    )

    # Éliminer les doublons fréquentiels proches
    filtered_rows = []

    for _, row in result_df.iterrows():

        keep = True

        for kept in filtered_rows:
            if abs(
                row["frequency"]
                - kept["frequency"]
            ) < 20:
                keep = False
                break

        if keep:
            filtered_rows.append(
                row.to_dict()
            )

    result_df = pd.DataFrame(filtered_rows)

    result_df = result_df.sort_values(
        "delta_Lta",
        ascending=False
    )
        
    best = result_df.loc[
        result_df["delta_Lta"].idxmax()
    ]

    summary = {
        "frequency":
            float(best["frequency"]),

        "Lpt":
            float(best["Lpt"]),

        "Lpn":
            float(best["Lpn"]),

        "delta_Lta":
            float(best["delta_Lta"]),

        "kt":
            float(best["kt"]),

        "n_candidates":
            len(rows)
    }

    return (
        result_df,
        summary
    )