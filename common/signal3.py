"""Prétraitements du jalon 3 (copie fidèle de jalon3.ipynb) : normalisation par fenêtre, augmentation « patch »,
mesures de décalage entre appareils."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis

FS, N = 125, 1250
NOMS3 = ["sinusal", "FA", "ESV", "tachy.", "BAV2"]
T_S = np.arange(N) / FS
SOS_LP05 = butter(2, 0.5, btype="low", fs=FS, output="sos")
SOS_BP = butter(2, [5, 20], btype="band", fs=FS, output="sos")
SOS_HP30 = butter(4, 30, btype="high", fs=FS, output="sos")
TOUTES = ("gain", "derive", "bruit", "secteur")


def normaliser_fenetres(X):
    X = np.asarray(X).astype("float32")
    mu, sd = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    return ((X - mu) / (sd + 1e-6))[..., None]


def augmenter_patch(X, rng, actives=TOUTES, P=None):
    """Augmentation qui imite le patch (gain, dérive, bruit, secteur). P = PARAMS_AUG du notebook."""
    n = len(X); X = X.astype("float32").copy()
    if "gain" in actives:
        X *= rng.uniform(P["gain"][0], P["gain"][1], size=(n, 1)).astype("float32")
    if "derive" in actives:
        for _ in range(2):
            f = rng.uniform(0.05, 0.5, (n, 1)); a = rng.uniform(0, P["derive_max"], (n, 1)); ph = rng.uniform(0, 2 * np.pi, (n, 1))
            X += (a * np.sin(2 * np.pi * f * T_S[None] + ph)).astype("float32")
    if "bruit" in actives:
        X += (rng.standard_normal(X.shape) * rng.uniform(0, P["bruit_max"], (n, 1))).astype("float32")
    if "secteur" in actives:
        a = rng.uniform(0, P["secteur_max"], (n, 1)); f = 50 + rng.normal(0, 0.1, (n, 1)); ph = rng.uniform(0, 2 * np.pi, (n, 1))
        X += (a * np.sin(2 * np.pi * f * T_S[None] + ph)).astype("float32")
    return X


def mesures_fenetre(x):
    """Quelques mesures de `caracteriser` du notebook, pour UNE fenêtre brute (mV)."""
    xc = (x - x.mean())[None].astype("float32")
    hp = sosfiltfilt(SOS_HP30, xc, axis=1).std(1)[0]; bp = np.percentile(np.abs(sosfiltfilt(SOS_BP, xc, axis=1)), 99.5, axis=1)[0]
    F = np.abs(np.fft.rfft(xc, axis=1))[0] * 2 / N
    return {"crête (centile 99,5 de |x|, mV)": float(np.percentile(np.abs(xc), 99.5)),
            "dérive < 0,5 Hz (écart-type, mV)": float(sosfiltfilt(SOS_LP05, xc, axis=1).std()),
            "raie 50 Hz : amplitude (mV)": float(F[499:502].max()),
            "rapport crête QRS / bruit > 30 Hz": float(bp / (hp + 1e-9)),
            "aplatissement (kurtosis)": float(kurtosis(xc[0]))}
