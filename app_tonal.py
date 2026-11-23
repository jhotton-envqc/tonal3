import streamlit as st
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import welch
from scipy.fft import fft
import matplotlib.pyplot as plt
import io

# Config de la page Streamlit
st.set_page_config(
    page_title="Comparaison des Méthodes d'Analyse des Tonalités",
    layout="wide"
)

st.title("🔊 Comparaison des Méthodes d'Analyse des Tonalités Acoustiques")

# ==============================================================================
# 1. MÉTHODE 1 : TIERS D'OCTAVE (ANNEXE IV - QC/MELCCFP)
# ==============================================================================
def analyze_third_octave(signal, fs):
    """
    Analyse par tiers d'octave simplifiée selon les critères de l'Annexe IV.
    Fréquences centrales normalisées (16 Hz à 20 000 Hz).
    """
    center_freqs = np.array([
        16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ])
    
    # Calcul des niveaux par bande via Spectres FFT regroupés
    N = len(signal)
    fft_vals = np.abs(fft(signal))[:N//2]
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    
    levels = []
    valid_freqs = []
    
    for fc in center_freqs:
        f_low = fc / (2**(1/6))
        f_high = fc * (2**(1/6))
        idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
        if len(idx) > 0:
            # Niveau RMS en dB dans la bande
            power_sum = np.sum(fft_vals[idx]**2)
            level_db = 10 * np.log10(power_sum + 1e-12)
            levels.append(level_db)
            valid_freqs.append(fc)
            
    levels = np.array(levels)
    valid_freqs = np.array(valid_freqs)
    
    # Analyse du dépassement des bandes adjacentes
    tonal_detected = False
    max_exceedance = 0
    tonal_freq = None
    kt = 0
    
    for i in range(1, len(levels) - 1):
        fc = valid_freqs[i]
        L_current = levels[i]
        L_prev = levels[i-1]
        L_next = levels[i+1]
        
        # Moyenne ou max des bandes adjacentes
        L_adj_max = max(L_prev, L_next)
        diff = L_current - L_adj_max
        
        # Critères du Tableau 4
        if fc <= 140:
            threshold = 15.0
        elif 141 <= fc <= 440:
            threshold = 8.0
        else: # >= 440 Hz
            threshold = 5.0
            
        if diff >= threshold:
            tonal_detected = True
            if diff > max_exceedance:
                max_exceedance = diff
                tonal_freq = fc
                kt = 5.0  # Pénalité fixe de 5 dB selon l'Annexe IV
                
    return valid_freqs, levels, tonal_detected, tonal_freq, max_exceedance, kt


# ==============================================================================
# 2. MÉTHODE 2 : FFT SELON ISO 1996-2:2007 (ANNEXE C)
# ==============================================================================
def analyze_iso_1996_2_2007(signal, fs):
    """
    Analyse tonale FFT simplifiée basée sur l'ISO 1996-2:2007 (Bandes critiques & Régression).
    """
    nperseg = min(len(signal), fs)  # Résolution de 1 Hz
    f, p = welch(signal, fs=fs, nperseg=nperseg, window='hann', scaling='spectrum')
    levels = 10 * np.log10(p + 1e-12)
    
    # Recherche de pics émergents de plus de 6 dB
    peaks_idx = []
    for i in range(2, len(levels) - 2):
        if levels[i] - levels[i-1] >= 6.0 and levels[i] - levels[i+1] >= 6.0:
            peaks_idx.append(i)
            
    if not peaks_idx:
        return f, levels, False, None, 0.0, 0.0
        
    # Analyse de la tonalité la plus forte
    max_delta_Lta = -999.0
    best_freq = None
    best_kt = 0.0
    
    for idx in peaks_idx:
        fc = f[idx]
        if fc < 50:
            continue
            
        # Largeur de bande critique (B_crit)
        b_crit = 100.0 if fc <= 500.0 else 0.20 * fc
        
        # Estimation du niveau du bruit masquant (L_pn) par moyenne locale hors pic
        f_low = fc - b_crit / 2
        f_high = fc + b_crit / 2
        band_mask = (f >= f_low) & (f <= f_high) & (np.abs(f - fc) > 2.0)
        
        if np.any(band_mask):
            l_pn = 10 * np.log10(np.mean(10**(levels[band_mask] / 10)))
        else:
            l_pn = levels[idx] - 10
            
        l_pt = levels[idx] # Niveau du ton
        
        # Audibilité tonale Delta L_ta (Eq. C.3)
        delta_Lta = l_pt - l_pn + 2.0 + np.log10(1.0 + (fc / 502.0)**2.5)
        
        # Ajustement Kt (Eq. C.4 - C.6)
        if delta_Lta > 10.0:
            kt = 6.0
        elif delta_Lta < 4.0:
            kt = 0.0
        else:
            kt = delta_Lta - 4.0
            
        if delta_Lta > max_delta_Lta:
            max_delta_Lta = delta_Lta
            best_freq = fc
            best_kt = kt
            
    tonal_detected = max_delta_Lta > 0
    return f, levels, tonal_detected, best_freq, max(0.0, max_delta_Lta), max(0.0, best_kt)


# ==============================================================================
# 3. MÉTHODE 3 : AUDIBILITÉ DES TONALITÉS (ISO 1996-2:2017 / ISO/TS 20065:2022)
# ==============================================================================
def analyze_iso_20065_2022(signal, fs):
    """
    Analyse d'audibilité des tonalités selon ISO/TS 20065 / ISO 1996-2:2017.
    """
    nperseg = min(len(signal), fs)
    f, p = welch(signal, fs=fs, nperseg=nperseg, window='hann', scaling='spectrum')
    levels = 10 * np.log10(p + 1e-12)
    
    # Application de la pondération A
    f_safe = np.where(f == 0, 1e-6, f)
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194
    pond_a = 20 * np.log10((f4**2 * f_safe**4) / ((f_safe**2 + f1**2) * np.sqrt(f_safe**2 + f2**2) * np.sqrt(f_safe**2 + f3**2) * (f_safe**2 + f4**2)) + 1e-12) + 2
    levels_a = levels + pond_a
    
    # Détection simplifiée des tons selon ISO 20065
    peaks = []
    for i in range(2, len(levels_a)-2):
        if levels_a[i] > levels_a[i-1] and levels_a[i] > levels_a[i+1]:
            if levels_a[i] - min(levels_a[i-2], levels_a[i+2]) > 3.0:
                peaks.append(i)
                
    if not peaks:
        return f, levels_a, False, None, 0.0, 0.0
        
    max_delta_L = -999.0
    best_freq = None
    
    for idx in peaks:
        fc = f[idx]
        if fc < 50 or fc > 10000:
            continue
            
        # Largeur de bande critique delta_fc
        delta_fc = 25 + 75 * (1 + 1.4 * (fc / 1000)**2)**0.69
        
        # Niveau du masque Ls
        mask = (f >= fc - delta_fc/2) & (f <= fc + delta_fc/2) & (np.abs(f - fc) > 1.5)
        if np.any(mask):
            l_s = 10 * np.log10(np.mean(10**(levels_a[mask] / 10)))
        else:
            l_s = levels_a[idx] - 10
            
        l_t = levels_a[idx]
        l_g = l_s + 10 * np.log10(delta_fc / (f[1] - f[0]))
        a_v = -2 - np.log10(1 + (fc / 502)**2.5)
        
        delta_L = l_t - l_g - a_v
        if delta_L > max_delta_L:
            max_delta_L = delta_L
            best_freq = fc
            
    # Échelle de pénalité Kt selon ISO 1996-2:2017
    if max_delta_L <= 0:
        kt = 0
    elif max_delta_L <= 2:
        kt = 1
    elif max_delta_L <= 4:
        kt = 2
    elif max_delta_L <= 6:
        kt = 3
    elif max_delta_L <= 9:
        kt = 4
    elif max_delta_L <= 12:
        kt = 5
    else:
        kt = 6
        
    tonal_detected = max_delta_L > 0
    return f, levels_a, tonal_detected, best_freq, max(0.0, max_delta_L), kt


# ==============================================================================
# INTERFACE UTILISATEUR STREAMLIT
# ==============================================================================

st.sidebar.header("📁 Chargement de l'échantillon sonore")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier audio (.wav)", type=["wav"])

if uploaded_file is not None:
    # Lecture du fichier WAV
    fs, data = wavfile.read(io.BytesIO(uploaded_file.getvalue()))
    if data.ndim > 1:
        data = data[:, 0]  # Canal 1 uniquement
    data = data.astype(float)
    data /= np.max(np.abs(data)) + 1e-12  # Normalisation

    st.subheader("Signal temporel")
    st.audio(uploaded_file)
    
    # Calcul des 3 méthodes
    m1_freqs, m1_levels, m1_det, m1_f, m1_val, m1_kt = analyze_third_octave(data, fs)
    m2_freqs, m2_levels, m2_det, m2_f, m2_val, m2_kt = analyze_iso_1996_2_2007(data, fs)
    m3_freqs, m3_levels, m3_det, m3_f, m3_val, m3_kt = analyze_iso_20065_2022(data, fs)

    # --------------------------------------------------------------------------
    # Tableau comparatif des résultats
    # --------------------------------------------------------------------------
    st.subheader("📊 Tableau Comparatif des Pénalités")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Méthode 1 : Tiers d'octave (Annexe IV)",
            value=f"{m1_kt:.1f} dB",
            delta=f"Fréq: {m1_f:.0f} Hz" if m1_f else "Aucune tonalité"
        )
        st.write(f"**Dépassement max :** {m1_val:.2f} dB")

    with col2:
        st.metric(
            label="Méthode 2 : ISO 1996-2:2007 (FFT)",
            value=f"{m2_kt:.1f} dB",
            delta=f"Fréq: {m2_f:.0f} Hz" if m2_f else "Aucune tonalité"
        )
        st.write(f"**Audibilité $\Delta L_{{ta}}$ :** {m2_val:.2f} dB")

    with col3:
        st.metric(
            label="Méthode 3 : ISO 1996-2:2017 / TS 20065",
            value=f"{m3_kt:.1f} dB",
            delta=f"Fréq: {m3_f:.0f} Hz" if m3_f else "Aucune tonalité"
        )
        st.write(f"**Audibilité $\Delta L$ :** {m3_val:.2f} dB")

    # --------------------------------------------------------------------------
    # Visualisation Graphique
    # --------------------------------------------------------------------------
    st.subheader("📈 Visualisation des Spectres et Tonalités")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # G1: Tiers d'octave
    axes[0].bar([str(int(f)) for f in m1_freqs], m1_levels, color='skyblue', edgecolor='black')
    axes[0].set_title("Méthode 1 : Spectre par Tiers d'Octave")
    axes[0].set_ylabel("Niveau (dB)")
    axes[0].tick_params(axis='x', rotation=45)
    
    # G2: ISO 1996-2:2007
    axes[1].plot(m2_freqs, m2_levels, color='green', label='Spectre FFT')
    if m2_f:
        axes[1].axvline(m2_f, color='red', linestyle='--', label=f'Ton détecté ({m2_f:.0f} Hz)')
    axes[1].set_title("Méthode 2 : ISO 1996-2:2007 (FFT)")
    axes[1].set_xlim([0, 5000])
    axes[1].set_ylabel("Niveau (dB)")
    axes[1].legend()

    # G3: ISO/TS 20065:2022
    axes[2].plot(m3_freqs, m3_levels, color='purple', label='Spectre dBA FFT')
    if m3_f:
        axes[2].axvline(m3_f, color='red', linestyle='--', label=f'Ton détecté ({m3_f:.0f} Hz)')
    axes[2].set_title("Méthode 3 : ISO/TS 20065:2022 (dBA)")
    axes[2].set_xlim([0, 5000])
    axes[2].set_xlabel("Fréquence (Hz)")
    axes[2].set_ylabel("Niveau (dBA)")
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Veuillez charger un fichier audio au format `.wav` dans la barre latérale pour lancer l'analyse comparative.")