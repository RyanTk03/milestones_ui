"""Prétraitements du jalon 1 : copie fidèle des fonctions du notebook (jalon1.ipynb).

Détection des QRS (filtres Butterworth + énergie + seuil médiane + k×MAD) puis
32 descripteurs répartis en 4 groupes. Toute modification ici doit rester
identique au notebook, sinon les modèles entraînés ne reçoivent plus les mêmes
entrées qu'à l'entraînement.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 125
N_POINTS = 1250                     # 10 s à 125 Hz
NOMS = ["sinusal", "FA", "ESV", "tachy.", "BAV2"]
NOMS_LONGS = ["Rythme sinusal", "Fibrillation atriale", "Extrasystoles ventriculaires",
              "Tachycardie sinusale", "Bloc AV du 2e degré"]
GROUPES = ["base", "amplitude/largeur", "bandes", "irrégularité"]


def filtre_butter(sig, kind, coupure, ordre):
    nyq = FS / 2
    freq = [c / nyq for c in coupure] if kind == "band" else coupure / nyq
    b, a = butter(ordre, freq, btype=kind)
    return filtfilt(b, a, sig)


def detecter_qrs(x, k_mad=3.5):
    sans_derive = filtre_butter(x, "high", 0.5, 2)
    bande = filtre_butter(sans_derive, "band", (3, 25), 4)
    zc = (bande - bande.mean()) / (bande.std() + 1e-8)
    larg = int(0.08 * FS)
    e = np.convolve(zc ** 2, np.ones(larg) / larg, mode="same")
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-8
    seuil = med + k_mad * mad
    cand, _ = find_peaks(e, distance=int(0.3 * FS), height=seuil, prominence=0.3 * seuil)
    if len(cand) == 0:
        return cand, bande
    gardes = [cand[0]]
    for p_ in cand[1:]:
        if (p_ - gardes[-1]) / FS < 0.36 and e[p_] < 0.5 * e[gardes[-1]]:
            continue   # onde T probable
        gardes.append(p_)
    pics = [gardes[0]]
    for p_ in gardes[1:]:
        if (p_ - pics[-1]) / FS >= 0.3:
            pics.append(p_)
    return np.array(pics), bande


def fc_depuis_pics(pics):
    return 60.0 / (np.diff(pics) / FS).mean() if len(pics) >= 2 else 0.0


def sampen(r, m=2, k=0.2):
    if len(r) < m + 3:
        return 0.
    r = np.asarray(r)
    tol = k * np.std(r) + 1e-9

    def cnt(m):
        t = np.array([r[i:i + m] for i in range(len(r) - m)])
        d = np.max(np.abs(t[:, None] - t[None]), axis=2)
        return (np.sum(d <= tol) - len(t)) / 2
    a_, b_ = cnt(m + 1), cnt(m)
    return float(-np.log((a_ + 1e-9) / (b_ + 1e-9))) if b_ > 0 else 0.


def descripteurs_fenetre(x, age, sexe, k_mad):
    """Renvoie (base, ampl, bandes, irr) : 4 dictionnaires, ordre des clés = ordre du notebook."""
    pics, f = detecter_qrs(x, k_mad)
    rr = np.diff(pics) / FS if len(pics) > 2 else np.array([1., 1.])
    med = np.median(rr)
    d = np.diff(rr) if len(rr) > 1 else np.array([0.])
    base = dict(nb=len(pics), mrr=rr.mean(), srr=rr.std(), rmssd=np.sqrt(np.mean(d ** 2)), cv=rr.std() / rr.mean(),
                minrr=rr.min(), maxrr=rr.max(), medrr=med, energie=float(np.mean(x ** 2)), age=age, sexe=sexe)
    demi = int(0.08 * FS)
    amp = np.array([np.abs(f[max(0, i - demi):i + demi + 1]).max() for i in pics]) if len(pics) else np.array([0.])
    larg = [np.sum(np.abs(f[max(0, i - 15):i + 16]) > 0.5 * np.abs(f[max(0, i - 15):i + 16]).max()) / FS for i in pics]
    ampl = dict(ampm=amp.mean(), amps=amp.std() / (amp.mean() + 1e-9),
                larg=np.mean(larg) if larg else 0, largs=np.std(larg) if larg else 0)
    Pw = np.abs(np.fft.rfft(x - x.mean())) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / FS)
    tot = Pw.sum() + 1e-9
    dens = Pw / tot
    pos = dens[dens > 0]
    bandes = dict(b1=Pw[(fr >= .5) & (fr < 5)].sum() / tot, b2=Pw[(fr >= 5) & (fr < 15)].sum() / tot,
                  b3=Pw[(fr >= 15) & (fr < 40)].sum() / tot, b4=Pw[fr >= 40].sum() / tot,
                  fdom=float(fr[np.argmax(Pw)]), centroide=float(np.sum(fr * dens)),
                  ent_spec=float(-np.sum(pos * np.log(pos))), part_qrs=Pw[(fr >= 5) & (fr < 25)].sum() / tot)
    hist, bords = np.histogram(rr, bins=10, density=True)
    hist = hist[hist > 0]
    irr = dict(pnn=np.mean(np.abs(d) > 0.05), sd1=np.std(d) / np.sqrt(2),
               sd2=np.sqrt(max(2 * rr.std() ** 2 - np.std(d) ** 2 / 2, 0)), samp=sampen(rr),
               court=np.mean(rr < 0.8 * med), long=np.mean(rr > 1.3 * med),
               rrr=np.mean(np.abs(rr / med - np.round(rr / med))),
               ent_rr=float(-np.sum(hist * np.log(hist)) * (bords[1] - bords[0])), ratio_pause=float(rr.max() / (med + 1e-6)))
    return base, ampl, bandes, irr


def toutes_les_features(x, age, sexe, k_mad):
    """Dictionnaire {nom_groupe: {descripteur: valeur}} pour une fenêtre."""
    return dict(zip(GROUPES, descripteurs_fenetre(np.asarray(x, "float64"), float(age), float(sexe), k_mad)))


def vecteur(feats, groupes_idx):
    """Concatène les groupes dans l'ordre demandé (même ordre que `matrice` du notebook)."""
    return np.hstack([list(feats[GROUPES[g]].values()) for g in groupes_idx]).astype("float32")
