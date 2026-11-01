"""Prétraitements du jalon 4 (copie fidèle de jalon4.ipynb) : fenêtre normalisée, suite de battements (RR, amplitude)."""
import numpy as np
from scipy.signal import find_peaks

FS, N, N_BATT = 125, 1250, 40
CLASSES4 = ["Sinusal", "FA", "Extrasyst. V", "Tachy sinus.", "Bloc AV 2"]


def normaliser_fenetres(X):
    X = np.asarray(X).astype("float32")
    mu, sd = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    return ((X - mu) / (sd + 1e-6))[..., None]


def detecter_qrs(x, fs=FS):
    """Version de REMPLACEMENT du notebook (détecteur type Pan-Tompkins simplifié) — celle utilisée dans jalon4.ipynb
    quand le notebook de démarrage n'est pas exécuté."""
    x = np.asarray(x, dtype="float32")
    z = x - np.median(x)
    d = np.diff(z, prepend=z[0])
    w = max(1, int(0.08 * fs))
    e = np.convolve(d ** 2, np.ones(w) / w, mode="same")
    p, _ = find_peaks(e, height=0.3 * np.percentile(e, 99), distance=int(0.25 * fs))
    r = int(0.06 * fs)
    pics = [max(0, i - r) + int(np.argmax(np.abs(z[max(0, i - r):i + r + 1]))) for i in p]
    return np.unique(pics).astype(int), e


def suite_battements(x):
    pics, _ = detecter_qrs(x)
    rr = np.diff(pics) / FS
    amp = x[pics[1:]].astype("float32")
    seq = np.zeros((N_BATT, 2), "float32")
    n = min(len(rr), N_BATT)
    seq[:n] = np.column_stack([rr, amp])[:n]
    return seq, pics


def norm_batt(B, mu_b, sd_b):
    B = np.asarray(B, "float32"); M = B[..., 0] > 0
    out = (B - np.asarray(mu_b, "float32")) / (np.asarray(sd_b, "float32") + 1e-6)
    out[~M] = 0.0
    return out.astype("float32")
