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

    factor = 2 ** (1/6)

    return (
        fc/factor,
        fc*factor
    )


def a_weighting(f):

    f = np.maximum(f, 1e-10)

    ra = (
        (12194**2 * f**4)
        /
        (
            (f**2 + 20.6**2)
            * np.sqrt(f**2 + 107.7**2)
            * np.sqrt(f**2 + 737.9**2)
            * (f**2 + 12194**2)
        )
    )

    return 20*np.log10(ra) + 2.0


def threshold(fc):

    if fc <= 125:
        return 15

    if fc <= 400:
        return 8

    return 5


def calculate_global_A(
        signal,
        fs,
        calibration_db):

    freq, pxx = welch(
        signal,
        fs=fs,
        nperseg=min(
            16384,
            len(signal)
        )
    )

    levels = (
        10*np.log10(
            pxx + 1e-30
        )
        +
        calibration_db
    )

    levels_A = (
        levels
        +
        a_weighting(freq)
    )

    return float(
        10*np.log10(
            np.sum(
                10**(levels_A/10)
            )
        )
    )


def third_octave_analysis(
        signal,
        fs,
        calibration_db):

    freq, pxx = welch(
        signal,
        fs=fs,
        nperseg=min(
            16384,
            len(signal)
        ),
        scaling="spectrum"
    )

    rows = []

    for fc in THIRD_OCTAVE_BANDS:

        fl, fu = third_octave_limits(fc)

        idx = np.where(
            (freq >= fl)
            &
            (freq < fu)
        )[0]

        if len(idx) == 0:
            continue

        band_energy = np.sum(
            pxx[idx]
        )

        level = (
            10*np.log10(
                band_energy + 1e-30
            )
            +
            calibration_db
        )
        rows.append({
            "frequency": fc,
            "band_level": level
        })

    return pd.DataFrame(rows)


def analyze_ni9801(
        signal,
        fs,
        calibration_db=0.0):

    bands = third_octave_analysis(
        signal,
        fs,
        calibration_db
    )

    LA_global = calculate_global_A(
        signal,
        fs,
        calibration_db
    )

    results = []

    for i in range(
        1,
        len(bands)-1
    ):

        fc = bands.iloc[i]["frequency"]

        center = bands.iloc[i]["band_level"]

        left = bands.iloc[i-1]["band_level"]

        right = bands.iloc[i+1]["band_level"]

        adjacent = max(
            left,
            right
        )

        emergence = (
            center - adjacent
        )

        crit1 = (
            emergence
            >=
            threshold(fc)
        )

        crit2 = (
            center >
            LA_global - 15
        )

        kt = (
            5
            if (
                crit1
                and crit2
            )
            else 0
        )

        results.append({

            "frequency": fc,

            "band_level": center,

            "adjacent_level": adjacent,

            "emergence": emergence,

            "threshold": threshold(fc),

            "criterion_emergence": crit1,

            "criterion_15dBA": crit2,

            "kt": kt

        })

    df = pd.DataFrame(results)

    best = df.loc[
        df["emergence"].idxmax()
    ]

    summary = {

        "frequency":
            float(best["frequency"]),

        "emergence":
            float(best["emergence"]),

        "kt":
            float(best["kt"]),

        "LA_global":
            float(LA_global)

    }


    return df, summary


def build_report(
        df,
        summary):

    lines = []

    lines.append("Méthode NI 98-01")
    lines.append("")

    lines.append(
        f"Bande : {summary['frequency']:.1f} Hz"
    )

    lines.append(
        f"Émergence : {summary['emergence']:.1f} dB"
    )

    lines.append(
        f"LA global : {summary['LA_global']:.1f} dBA"
    )

    lines.append(
        f"Kt : {summary['kt']:.1f} dB"
    )

    return "\n".join(lines)

# =====================================================
# ANALYSE À PARTIR DE BANDES COLLÉES
# =====================================================

import re

def parse_octave_text(text):

    rows = []
    LA_global = None

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("Project"):
            continue

        if "Frequency" in line:
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        freq_str = parts[0].strip()
        level_str = parts[1].strip()

        if freq_str == "A":

            try:
                LA_global = float(
                    level_str.replace(",", ".")
                )
            except:
                pass

            continue

        if freq_str == "C":
            continue

        try:

            if "kHz" in freq_str:

                freq = (
                    freq_str
                    .replace("kHz", "")
                    .replace(" ", "")
                    .replace(",", ".")
                )

                freq = float(freq) * 1000

            else:

                freq = (
                    freq_str
                    .replace("Hz", "")
                    .replace(" ", "")
                    .replace(",", ".")
                )

                freq = float(freq)

            level = float(
                level_str.replace(",", ".")
            )

            rows.append(
                {
                    "frequency": freq,
                    "band_level": level
                }
            )

        except Exception:
            pass

    df = pd.DataFrame(rows)

    if not df.empty:

        df = (
            df.sort_values("frequency")
              .reset_index(drop=True)
        )

    return df, LA_global

def analyze_ni9801_from_octaves(
        df,
        LA_global):
    """
    Analyse NI98-01 sur bandes déjà calculées.
    """

    if len(df) < 3:
        raise ValueError(
            "Au moins 3 bandes sont requises."
        )

    results = []

    for i in range(len(df)):

        fc = df.iloc[i]["frequency"]

        center = df.iloc[i]["band_level"]

        if i == 0:

            adjacent = df.iloc[i+1]["band_level"]

            emergence = np.nan

        elif i == len(df)-1:

            adjacent = df.iloc[i-1]["band_level"]

            emergence = np.nan

        else:

            left = df.iloc[i-1]["band_level"]

            right = df.iloc[i+1]["band_level"]

            adjacent = max(
                left,
                right
            )

            emergence = (
                center - adjacent
            )

        crit1 = (

            not np.isnan(emergence)
            and emergence >= threshold(fc)
        )

        crit2 = (
            center > (LA_global - 15)
        )

        kt = (
            5
            if (
                crit1
                and crit2
            )
            else 0
        )

        results.append({

            "frequency": fc,

            "band_level": center,

            "adjacent_level": adjacent,

            "emergence": emergence,

            "threshold": threshold(fc),

            "criterion_emergence": crit1,

            "criterion_15dBA": crit2,

            "kt": kt

        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise ValueError(
            "Impossible de calculer l'émergence."
        )
    
    best = result_df.loc[
        result_df["emergence"].idxmax()
    ]

    summary = {

        "frequency":
            float(best["frequency"]),

        "emergence":
            float(best["emergence"]),

        "kt":
            float(best["kt"]),

        "LA_global":
            float(LA_global)
    }

    return (
        result_df,
        summary
    )