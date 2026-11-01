"""Prétraitements du jalon 2 (copie fidèle de jalon2.ipynb) : normalisation par fenêtre, bruit « patch »,
descripteurs du MLP comparateur, champ réceptif."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

FS, N = 125, 1250
NOMS2 = ["sinusal", "FA", "ESV", "tachy sin.", "bloc AV"]
TEMPS = np.arange(N) / FS
SOS_QRS = butter(2, [5, 20], btype="band", fs=FS, output="sos")
SOS_HP = butter(2, 0.5, btype="high", fs=FS, output="sos")
NOMS_DESC = ["n_batt", "rr_moy", "rr_sd", "rmssd", "pnn50", "rr_cv", "rr_min", "rr_max", "rr_max/med",
             "frac_courts", "frac_longs", "amp_cv", "amp_max/med"]


def normaliser_fenetres(X):
    X = np.asarray(X).astype("float32")
    mu, sd = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    return ((X - mu) / (sd + 1e-6))[..., None]


def normaliser_hp(X):
    return normaliser_fenetres(sosfiltfilt(SOS_HP, np.asarray(X).astype("float64"), axis=1))


def bruit_calibre(X, rng, p):
    """Bruit « de type patch » calibré (section 4 du notebook). p = params_bruit."""
    n = len(X); X = X.astype("float32")
    u = lambda lo_hi: rng.uniform(lo_hi[0], lo_hi[1], (n, 1))
    derive = u(p["derive_amp"]) * np.sin(2 * np.pi * u(p["derive_f"]) * TEMPS + rng.uniform(0, 2 * np.pi, (n, 1)))
    secteur = u(p["secteur_amp"]) * np.sin(2 * np.pi * 50 * TEMPS + rng.uniform(0, 2 * np.pi, (n, 1)))
    muscle = rng.normal(0.0, 1.0, X.shape) * u(p["muscle_sigma"])
    return (u(p["gain"]) * X + derive + secteur + muscle).astype("float32")


def detecter_R(x):
    y = sosfiltfilt(SOS_QRS, x.astype("float64"))
    env = np.convolve(y ** 2, np.ones(10) / 10, mode="same")
    pk, _ = find_peaks(env, height=0.3 * np.percentile(env, 99), distance=int(0.25 * FS))
    pk = np.array([max(range(max(p - 5, 0), min(p + 6, N)), key=lambda i: abs(y[i])) for p in pk], dtype=int)
    return pk, y


def descripteurs_fenetre(x):
    pk, y = detecter_R(x)
    if len(pk) < 3:
        return [len(pk)] + [np.nan] * 12
    rr = np.diff(pk) / FS; amp = np.abs(y[pk]); med = np.median(rr)
    return [len(pk), rr.mean(), rr.std(), np.sqrt(np.mean(np.diff(rr) ** 2)), np.mean(np.abs(np.diff(rr)) > 0.05),
            rr.std() / rr.mean(), rr.min(), rr.max(), rr.max() / med, np.mean(rr < 0.85 * med),
            np.mean(rr > 1.15 * med), amp.std() / amp.mean(), amp.max() / np.median(amp)]


def champ_receptif(couches):
    r, j = 1, 1
    for k, s, d in couches:
        r += (k - 1) * d * j
        j *= s
    return r, j


def couches_cnn(profondeur, k, dilations=None):
    dil = dilations or [1] * profondeur
    return [c for i in range(profondeur) for c in [(k, 1, dil[i]), (2, 2, 1)]]
