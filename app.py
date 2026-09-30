# -*- coding: utf-8 -*-

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

from scipy.io import wavfile

from utils.fft import (
    calculate_fft,
    calculate_raw_fft,
    decimate_for_plot,
    get_dominant_frequency
)

from methods.ni98_01 import (
    analyze_ni9801,
    analyze_ni9801_from_octaves,
    parse_octave_text,
    build_report
)

from utils.calibration import (
    estimate_calibration_factor
)

from methods.iso2007 import (
analyze_iso2007
)


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Comparateur de tonalités",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# FONCTIONS
# ==========================================================

def load_wav(file_bytes):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(file_bytes)
        tmp_path = tmp.name

    fs, data = wavfile.read(tmp_path)

    os.remove(tmp_path)

    if data.ndim > 1:
        data = data[:, 0]

    return fs, data

def create_playback_audio(
        signal,
        gain_db=0,
        normalize=True):

    audio = signal.astype(float)

    if normalize:

        peak = np.max(
            np.abs(audio)
        )

        if peak > 0:

            audio = (
                audio / peak
            ) * 0.95

    gain = 10 ** (
        gain_db / 20
    )

    audio = audio * gain

    audio = np.clip(
        audio,
        -32767,
        32767
    )

    return audio.astype(
        np.int16
    )


def get_peak_dbfs(signal):

    peak = np.max(
        np.abs(signal)
    )

    if peak <= 0:
        return -np.inf

    if signal.dtype.kind in ["i", "u"]:

        reference = np.iinfo(
            signal.dtype
        ).max

    else:

        reference = 1.0

    return (
        20
        * np.log10(
            peak / reference
        )
    )


def create_fft_plot(
        freq,
        level,
        dominant_freq=None,
        max_freq=5000):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=freq,
            y=level,
            mode="lines",
            name="Spectre FFT",
            line=dict(
                color="navy",
                width=1
            ),
            hovertemplate=
            "Fréquence : %{x:.1f} Hz<br>"
            "Niveau : %{y:.1f} dB"
            "<extra></extra>"
        )
    )

    if dominant_freq is not None:

        fig.add_vline(
            x=dominant_freq,
            line_color="red",
            line_width=2,
            line_dash="dash"
        )

        fig.add_annotation(
            x=dominant_freq,
            y=np.max(level),
            text=f"{dominant_freq:.1f} Hz",
            showarrow=True,
            arrowcolor="red",
            font=dict(
                color="red",
                size=12
            )
        )

    fig.update_layout(
        title="Spectre FFT",
        template="plotly_white",
        hovermode="closest",
        height=650
    )

    fig.update_xaxes(
        title="Fréquence (Hz)",
        range=[0, max_freq]
    )

    fig.update_yaxes(
        title="Niveau (dB)"
    )

    return fig


def create_ni98_plot(df, summary, la_limit=None):

    colors = []

    for _, row in df.iterrows():

        if row["kt"] > 0:
            colors.append("red")

        elif row["emergence"] > 0:
            colors.append("orange")

        else:
            colors.append("steelblue")

    fig = go.Figure()

    fig.add_trace(

        go.Bar(

            x=df["frequency"].astype(str),

            y=df["band_level"],

            marker_color=colors,

            customdata=df["emergence"],

            hovertemplate=
            "<b>%{x} Hz</b><br>"
            "Niveau = %{y:.1f} dB<br>"
            "Émergence = %{customdata:.1f} dB"
            "<extra></extra>"

        )

    )

    try:

        candidate_freq = summary["frequency"]

        candidate_level = float(

            df.loc[
                df["frequency"] == candidate_freq,
                "band_level"
            ].iloc[0]

        )

        fig.add_trace(

            go.Scatter(

                x=[str(candidate_freq)],

                y=[candidate_level],

                mode="markers+text",

                text=["Bande candidate"],

                textposition="top center",

                marker=dict(
                    color="red",
                    size=14,
                    symbol="diamond"
                ),

                showlegend=False

            )

        )

    except Exception:
        pass

    if (
        la_limit is not None
        and pd.notna(la_limit)
    ):

        fig.add_hline(
            y=la_limit,
            line_color="green",
            line_dash="dash",
            line_width=2,
            annotation_text=
            f"LA global - 15 dB ({la_limit:.1f} dB)",
            annotation_position="top left"
        )

    fig.update_layout(

        title="Analyse NI 98-01",

        template="plotly_white",

        height=600,

        xaxis_title="Fréquence centrale (Hz)",

        yaxis_title="Niveau de bande (dB)"

    )

    return fig


# ==========================================================
# TITRE
# ==========================================================

st.title(
    "Analyse comparative des tonalités"
)

st.markdown("""
### Méthodes disponibles

- NI 98-01
- ISO 1996-2:2007
- ISO 1996-2:2017 / ISO-TS 20065:2022
""")

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("Données d'entrée")

    uploaded_file = st.file_uploader(
        "Fichier WAV",
        type=["wav"]
    )

    octave_text = st.text_area(
        "Bandes de tiers d'octave (optionnel)",
        height=350,
        placeholder="""Frequency\tLZeq [dB]
6,3 Hz\t46,13
8 Hz\t46,52
10 Hz\t50,50
12,5 Hz\t53,12
...
A\t63,99
C\t73,79"""
    )

    st.subheader("Source NI 98-01")

    ni98_source = st.radio(
        "Calcul NI 98-01 à partir de :",
        [
            "Bandes de tiers d'octave",
            "Fichier WAV"
        ]
    )

    st.subheader("Calibration")

    calibration_db = st.number_input(
        "Calibration acoustique (dB)",
        min_value=0.0,
        max_value=150.0,
        value=0.0,
        step=0.1
    )

    auto_calibration = st.checkbox(
        "Calibration automatique depuis les tiers d'octave",
        value=True
    )

    st.caption(
        "La calibration automatique remplace la valeur manuelle lorsqu'un WAV et des bandes de tiers d'octave sont disponibles."
    )

    st.subheader("Affichage FFT")

    fft_display_mode = st.radio(
        "Type de spectre",
        [
            "FFT brute",
            "Welch"
        ],
        index=1      # Welch par défaut
    )

    fft_max_freq = st.selectbox(
        "Fréquence maximale affichée",
        [5000, 10000, 20000],
        index=0
    )

    run_analysis = st.button(
        "Analyser",
        type="primary",
        use_container_width=True
    )

    st.subheader(
        "Écoute audio"
    )

    normalize_audio = st.checkbox(
        "Normalisation automatique",
        value=True
    )

    playback_gain_db = st.slider(
        "Gain d'écoute (dB)",
        min_value=-20,
        max_value=40,
        value=0,
        step=1
    )

# ==========================================================
# DISPONIBILITÉ DES DONNÉES
# ==========================================================

wav_available = uploaded_file is not None

octave_available = (
    octave_text is not None
    and octave_text.strip() != ""
)
# ==========================================================
# ONGLETS
# ==========================================================

(
    tab_info,
    tab_fft,
    tab_ni,
    tab_iso2007,
    tab_iso2017,
    tab_compare
) = st.tabs([
    "Info",
    "FFT",
    "NI 98-01",
    "ISO 1996-2:2007",
    "ISO 1996-2:2017",
    "Comparaison"
])

# ==========================================================
# AUCUNE DONNÉE
# ==========================================================

if not wav_available and not octave_available:

    with tab_info:

        st.info(
            "Importer un fichier WAV ou coller des bandes de tiers d'octave."
        )

    st.stop()


# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================

if wav_available:

    file_bytes = uploaded_file.read()

    fs, signal = load_wav(
        file_bytes
    )

    playback_signal = (
        create_playback_audio(
            signal,
            gain_db=playback_gain_db,
            normalize=normalize_audio
        )
    )

    peak_dbfs = get_peak_dbfs(
        signal
    )

if octave_available:

    octave_df, LA_global = parse_octave_text(
        octave_text
    )

    if octave_df.empty:

        st.error(
            "Aucune bande valide détectée."
        )

        st.stop()


if wav_available and octave_available:

    auto_calibration_db, calibration_df, calibration_stats = (
        estimate_calibration_factor(
            signal,
            fs,
            octave_df
        )
    )

if (
    auto_calibration
    and wav_available
    and octave_available
):
    calibration_db = auto_calibration_db


# ==========================================================
# INFO
# ==========================================================

with tab_info:

    col1, col2 = st.columns(2)

    # ------------------------------------------------------
    # WAV
    # ------------------------------------------------------

    with col1:

        st.subheader("Fichier WAV")

        if wav_available:

            st.success("WAV chargé")

            st.write(
                f"Nom : {uploaded_file.name}"
            )

            st.write(
                f"Échantillonnage : {fs:,} Hz"
            )

            st.write(
                f"Durée : {len(signal)/fs:.1f} s"
            )

            st.write(
                f"Niveau crête : {peak_dbfs:.1f} dBFS"
            )

            if peak_dbfs < -40:

                st.warning(
                    "Enregistrement très faible."
                )

            elif peak_dbfs < -20:

                st.info(
                    "Enregistrement modéré."
                )

            elif peak_dbfs < -3:

                st.success(
                    "Niveau d'enregistrement adéquat."
                )

            else:

                st.warning(
                    "Signal proche de la saturation."
                )

            st.audio(
                playback_signal,
                sample_rate=fs
            )

        else:

            st.info(
                "Aucun fichier WAV."
            )

    # ------------------------------------------------------
    # TIERS D'OCTAVE
    # ------------------------------------------------------

    with col2:

        st.subheader(
            "Bandes de tiers d'octave"
        )

        if octave_available:

            st.success(
                f"{len(octave_df)} bandes détectées"
            )

            st.dataframe(
                octave_df,
                use_container_width=True,
                height=300
            )

        else:

            st.info(
                "Aucune bande collée."
            )

    # ------------------------------------------------------
    # CALIBRATION AUTOMATIQUE
    # ------------------------------------------------------

    if wav_available and octave_available:

        st.divider()

        st.subheader(
            "Calibration automatique"
        )

        std = calibration_stats["std_delta"]

        if std < 3:
            quality = "Excellente"
        elif std < 6:
            quality = "Bonne"
        elif std < 10:
            quality = "Acceptable"
        else:
            quality = "Faible"

        
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Facteur estimé",
            f"{auto_calibration_db:.1f} dB"
        )

        c2.metric(
            "Bandes comparées",
            calibration_stats["n_bands"]
        )

        c3.metric(
            "Écart-type",
            f"{calibration_stats['std_delta']:.1f} dB"
        )

        c4.metric(
            "Qualité",
            quality
        )
        st.dataframe(
            calibration_df.round(1),
            use_container_width=True
        )
        calibration_df["band_level_wav_cal"] = (
            calibration_df["band_level_wav"]
            + auto_calibration_db
        )

        
        ### test temporaire
        fig_cal = go.Figure()

        fig_cal.add_trace(
            go.Scatter(
                x=calibration_df["frequency"],
                y=calibration_df["band_level_wav_cal"],
                mode="lines+markers",
                name="WAV calibré",
                line=dict(color="green")
            )
        )

        fig_cal.add_trace(
            go.Scatter(
                x=calibration_df["frequency"],
                y=calibration_df["band_level_measured"],
                mode="lines+markers",
                name="Sonomètre"
            )
        )

        fig_cal.update_layout(
            title="Comparaison des bandes de tiers d'octave",
            xaxis_title="Fréquence (Hz)",
            yaxis_title="Niveau (dB)",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(
            fig_cal,
            use_container_width=True
        )

        ###################


        st.metric(
            "Calibration utilisée",
            f"{calibration_db:.1f} dB"
        )

# ==========================================================
# ANALYSE
# ==========================================================


# ----------------------------------
# PRIORITÉ AUX TIERS D'OCTAVE
# ----------------------------------

if run_analysis:

# ==============================================
# FFT (seulement si WAV disponible)
# ==============================================

    if wav_available:

        if fft_display_mode == "FFT brute":

            freq_fft, level_fft = (
                calculate_raw_fft(
                    signal,
                    fs,
                    calibration_db
                )
            )

        else:

            freq_fft, level_fft = (
                calculate_fft(
                    signal,
                    fs,
                    calibration_db
                )
            )

        if fft_display_mode == "FFT brute":

            freq_plot, level_plot = decimate_for_plot(
                freq_fft,
                level_fft,
                max_points=50000
            )

        else:

            freq_plot, level_plot = (
                decimate_for_plot(
                freq_fft,
                level_fft,
                max_points=8000
                )
            )

        dominant_freq = (
            get_dominant_frequency(
                freq_fft,
                level_fft
            )
        )

        max_level = float(
            level_fft.max()
        )

    # ==============================================
    # NI98-01
    # ==============================================

    if (
        ni98_source
        == "Bandes de tiers d'octave"
    ):

        if not octave_available:

            st.error(
                "Aucune bande de tiers d'octave disponible."
            )

            st.stop()

        ni_df, ni_summary = (
            analyze_ni9801_from_octaves(
                octave_df,
                LA_global
            )
        )

    else:

        if not wav_available:

            st.error(
                "Aucun fichier WAV disponible."
            )

            st.stop()

        ni_df, ni_summary = (
            analyze_ni9801(
                signal,
                fs,
                calibration_db
            )
        )

    report = build_report(
        ni_df,
        ni_summary
    )

# ============================================================
# iso 1996-2:2007
# ============================================================
    if wav_available:

        iso2007_df, iso2007_summary = (
            analyze_iso2007(
                signal,
                fs,
                calibration_db
            )
        )


    # ======================================================
    # FFT
    # ======================================================

    with tab_fft:

        st.info(
            f"Mode d'affichage : {fft_display_mode}"
        )

        mask = freq_plot <= fft_max_freq

        freq_display = freq_plot[mask]
        level_display = level_plot[mask]

        if not wav_available:

            st.info(
                "Aucune FFT disponible sans fichier WAV."
            )
            
        else:

            st.subheader(
                "Spectre FFT interactif"
            )

            st.plotly_chart(
                create_fft_plot(
                    freq_display,
                    level_display,
                    dominant_freq=dominant_freq,
                    max_freq=fft_max_freq
                ),
                use_container_width=True
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Fréquence dominante",
                f"{dominant_freq:.1f} Hz"
            )
            st.caption(
                f"Spectre affiché de 0 à {fft_max_freq:,} Hz"
            )
            st.caption(
            f"{len(freq_display):,} points affichés"
            )
            c2.metric(
                "Niveau maximal",
                f"{max_level:.1f} dB"
            )

    # ======================================================
    # NI 98-01
    # ======================================================

    with tab_ni:

        st.subheader(
            "Résultats NI 98-01"
        )

        st.info(
            f"Source utilisée : {ni98_source}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Bande",
            f"{ni_summary['frequency']:.1f} Hz"
        )

        c2.metric(
            "Émergence",
            f"{ni_summary['emergence']:.1f} dB"
        )

        c3.metric(
            "Kt",
            f"{ni_summary['kt']:.1f} dB"
        )

        c4.metric(
            "LA global",
            f"{ni_summary['LA_global']:.1f} dBA"
        )

        la_limit = None

        if (
            "LA_global" in ni_summary
            and pd.notna(
                ni_summary["LA_global"]
            )
        ):

            la_limit = (
                ni_summary["LA_global"] - 15
            )


        st.plotly_chart(
            create_ni98_plot(
                ni_df,
                ni_summary,
                la_limit
            ),
            use_container_width=True
        )

        # ======================================================
        # BANDES ÉMERGENTES
        # ======================================================

        st.subheader(
            "Bandes émergentes"
        )

        emergent_df = ni_df[
            ni_df["emergence"] > 0
        ].copy()

        if not emergent_df.empty:

            st.dataframe(
                emergent_df[
                    [
                        "frequency",
                        "band_level",
                        "adjacent_level",
                        "emergence",
                        "threshold",
                        "criterion_emergence",
                        "criterion_15dBA",
                        "kt"
                    ]
                ].round(1),
                use_container_width=True
            )

        else:

            st.info(
                "Aucune bande émergente détectée."
            )

        st.subheader(
            "Tableau détaillé"
        )

        display_df = ni_df.copy()

        display_df["band_level"] = (
            display_df["band_level"].round(1)
        )

        display_df["adjacent_level"] = (
            display_df["adjacent_level"].round(1)
        )

        display_df["emergence"] = (
            display_df["emergence"].round(1)
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )

        st.subheader("Rapport")

        st.text_area(
            "",
            report,
            height=200
        )

        st.download_button(
            "Télécharger le rapport",
            report,
            file_name="NI98_01.txt",
            mime="text/plain"
        )


    # ======================================================
    # ISO 2007
    # ======================================================

    with tab_iso2007:

        st.subheader(
            "ISO 1996-2:2007"
        )

        if iso2007_df.empty:

            st.info(
                "Aucune tonalité détectée."
            )

        else:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Fréquence",
                f"{iso2007_summary['frequency']:.1f} Hz"
            )

            c2.metric(
                "ΔLta",
                f"{iso2007_summary['delta_Lta']:.1f} dB"
            )

            c3.metric(
                "Kt",
                f"{iso2007_summary['kt']:.1f} dB"
            )

            c4.metric(
                "Lpt",
                f"{iso2007_summary['Lpt']:.1f} dB"
            )

            st.dataframe(
                iso2007_df.round(1),
                use_container_width=True
            )

            st.subheader(
                "Tous les candidats"
            )

            st.dataframe(
                iso2007_df.round(1),
                use_container_width=True
            )

    # ======================================================
    # ISO 2017
    # ======================================================

    with tab_iso2017:

        st.info(
            "Module ISO 1996-2:2017 / ISO-TS 20065:2022 à intégrer."
        )

    # ======================================================
    # COMPARAISON
    # ======================================================

    with tab_compare:

        comparison = pd.DataFrame([
            {
                "Méthode": "NI 98-01",
                "Kt (dB)": ni_summary["kt"]
            }
        ])

        st.dataframe(
            comparison,
            use_container_width=True
        )

        fig_compare = px.bar(
            comparison,
            x="Méthode",
            y="Kt (dB)",
            text="Kt (dB)"
        )

        fig_compare.update_layout(
            template="plotly_white",
            height=500
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True
        )

        st.success(
            f"Pénalité maximale : {comparison['Kt (dB)'].max():.1f} dB"
        )