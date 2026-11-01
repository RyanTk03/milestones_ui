"""Onglet Jalon 4 : le rythme est une séquence (RNN, LSTM, GRU, hybride CNN→GRU, suite de battements)."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from common.signal4 import FS, N, N_BATT, CLASSES4, normaliser_fenetres, suite_battements, norm_batt

ART = Path(__file__).resolve().parent.parent / "artifacts" / "jalon4"
COUL = ["#2E86AB", "#D1495B", "#EDAE49", "#66A182", "#8D6A9F"]
T_S = np.arange(N) / FS

ICON_ALERTE, ICON_OK, ICON_WARN = ":material/notification_important:", ":material/check_circle:", ":material/warning:"
VUES = {"demo": ":material/science: Démonstration",
        "res": ":material/bar_chart: Résultats & expériences",
        "theo": ":material/function: Vérifications théoriques"}


@st.cache_resource(show_spinner="Chargement des modèles du jalon 4…")
def charger():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras
    cfg = json.load(open(ART / "config.json")); res = json.load(open(ART / "resultats_jalon4.json")); hist = json.load(open(ART / "historiques.json"))
    mod = {k: keras.models.load_model(ART / v["fichier"], compile=False) for k, v in cfg["modeles"].items()}
    for k, m in mod.items():                               # échauffement (le RNN simple est lent au premier appel)
        m(np.zeros((1, N, 1) if cfg["modeles"][k]["entree"] == "brut" else (1, N_BATT, 2), "float32"), training=False)
    return cfg, res, hist, mod, pd.read_csv(ART / "journal.csv"), dict(np.load(ART / "exemples.npz")), dict(np.load(ART / "scores.npz"))


def entree(cfg, k, xwin, batt_norm):
    return normaliser_fenetres(xwin[None]) if cfg["modeles"][k]["entree"] == "brut" else batt_norm[None]


def predire(mod, cfg, k, xwin, batt_norm):
    x = entree(cfg, k, xwin, batt_norm); m = mod[k]
    t0 = time.perf_counter(); p = m(x, training=False).numpy()[0]; return p, 1000 * (time.perf_counter() - t0)


def saliency(mod, cfg, k, xwin, classe):
    """‖∂p(classe)/∂x_t‖ pour chaque pas de la fenêtre (disparition du gradient, observée sur cette fenêtre)."""
    import tensorflow as tf
    x = tf.convert_to_tensor(normaliser_fenetres(xwin[None]))
    with tf.GradientTape() as tape:
        tape.watch(x); p = mod[k](x, training=False)[0, classe]
    return np.abs(tape.gradient(p, x).numpy()[0, :, 0])


def lisser(v, k=25): return np.convolve(v, np.ones(k) / k, mode="same")


def heat(cm, noms, titre):
    cm = np.array(cm); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    f = go.Figure(go.Heatmap(z=cmn, x=noms, y=noms, colorscale="Blues", zmin=0, zmax=1, showscale=False,
                             text=[[f"{cm[i, j]}" for j in range(5)] for i in range(5)], texttemplate="%{text}"))
    f.update_yaxes(autorange="reversed", title="vrai"); f.update_xaxes(title="prédit"); f.update_layout(title=titre, height=330, margin=dict(l=10, r=10, t=45, b=10)); return f


# ------------------------------------------------------------------ démonstration
# @st.fragment : un changement de widget ne relance que ce bloc -> la page ne remonte plus en haut.
@st.fragment
def section_demo(cfg, res, hist, mod, ex, sc):
    g, d = st.columns([1, 2.3], gap="large")
    with g:
        st.subheader("1 · Signal")
        src = st.radio("Source", ["Holter (test)", "Holter + bruit ajouté", "Fichier .csv / .npy"], key="j4_src")
        x = None; vrai = None; batt_ref = None
        if src != "Fichier .csv / .npy":
            cl = st.selectbox("Rythme réel", range(5), format_func=lambda k: CLASSES4[k], key="j4_cl")
            idx = np.where(ex["y"] == cl)[0]; i = idx[st.slider("Fenêtre n°", 1, len(idx), 1, key="j4_i") - 1]
            x = ex["X"][i].astype("float32"); vrai = int(ex["y"][i]); batt_ref = ex["batt"][i]; st.caption(f"Fenêtre test n°{ex['idx'][i]} · FC annotée {ex['fc'][i]:.0f} bpm")
            if src.endswith("ajouté"):
                s = st.slider("Bruit blanc ajouté (σ, mV)", 0.0, 0.5, 0.1, 0.01, key="j4_bruit")
                x = (x + np.random.default_rng(0).normal(0, s, x.shape)).astype("float32"); batt_ref = None
                st.caption("Test de robustesse : les modèles ont été appris sur du Holter propre ; la représentation en battements dépend du détecteur de QRS.")
        else:
            f = st.file_uploader("1 250 points (10 s, 125 Hz, mV)", type=["csv", "npy", "txt"], key="j4_up")
            if f is None: st.info("Déposez un fichier."); return
            try: a = np.load(f).ravel() if f.name.endswith(".npy") else np.loadtxt(f, delimiter=",").ravel()
            except Exception as e: st.error(f"Lecture impossible : {e}"); return
            if a.size != N: st.error(f"{a.size} valeurs reçues, {N} attendues."); return
            x = a.astype("float32")
        st.subheader("2 · Modèle détaillé")
        k = st.selectbox("Modèle à analyser", list(cfg["modeles"]), index=list(cfg["modeles"]).index(cfg["choix"]),
                         format_func=lambda s: ("★ " if s == cfg["choix"] else "") + cfg["modeles"][s]["nom"], key="j4_k")
        st.caption("Les 6 modèles principaux (+ l'essai sans masquage) sont comparés côte à côte ; ★ = modèle retenu (hybride CNN+GRU).")

    with d:
        seq, pics = suite_battements(x)
        if batt_ref is not None and not np.allclose(seq, batt_ref, atol=1e-5):
            st.warning("La suite de battements recalculée par l'application diffère de celle du notebook : le détecteur de QRS du notebook n'est pas la version de remplacement. "
                       "Les exemples utilisent la suite exacte du notebook ; adaptez `common/signal4.py` pour les fichiers déposés.", icon=ICON_WARN)
        sb = norm_batt(batt_ref if batt_ref is not None else seq, cfg["mu_b"], cfg["sd_b"])
        L, P = [], {}
        for kk, m in cfg["modeles"].items():
            p, ms = predire(mod, cfg, kk, x, sb); P[kk] = p
            L.append({"modèle": m["nom"], "prédiction": CLASSES4[int(p.argmax())], "confiance": round(float(p.max()), 3), "score FA": round(float(p[1]), 3), "latence (ms)": round(ms, 1),
                      "paramètres": m["params"], **{f"p({c})": round(float(v), 3) for c, v in zip(CLASSES4, p)}})
        p = P[cfg["choix"]]; kc = int(p.argmax())
        st.subheader("3 · Décision du modèle retenu")
        if kc != 0:
            st.error(f"Alerte — **{CLASSES4[kc]}** ({p[kc]:.1%}) · hybride CNN+GRU", icon=ICON_ALERTE)
        else:
            st.success(f"Pas d'alerte — **{CLASSES4[kc]}** ({p[kc]:.1%}) · hybride CNN+GRU", icon=ICON_OK)
        seuil = cfg["modeles"][cfg["choix"]]["seuil_fa"]
        c = st.columns(3); c[0].metric("Score FA", f"{p[1]:.3f}"); c[1].metric("Seuil FA (spéc. 95 % en validation)", f"{seuil:.3f}"); c[2].metric("FA au seuil ?", "oui" if p[1] >= seuil else "non")
        if vrai is not None: st.caption(f"Rythme réel : **{CLASSES4[vrai]}**")
        fig = go.Figure()
        for kk, m in cfg["modeles"].items():
            if kk != "gru_battements_sans_masque": fig.add_bar(name=m["nom"], x=CLASSES4, y=P[kk])
        fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.05], title="probabilité"), legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(L), hide_index=True, width="stretch")

        st.markdown(f"#### Pourquoi ? — {cfg['modeles'][k]['nom']}")
        kp = int(P[k].argmax())
        if cfg["modeles"][k]["entree"] == "brut":
            fg = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.5, .5], vertical_spacing=.1,
                               subplot_titles=("Signal brut (mV)", f"‖∂p({CLASSES4[kp]})/∂x_t‖ — influence de chaque pas sur la décision (log)"))
            fg.add_scatter(x=T_S, y=x, line=dict(width=1, color="#2E86AB"), showlegend=False, row=1, col=1); fg.add_scatter(x=T_S[pics], y=x[pics], mode="markers", marker=dict(symbol="triangle-down", color="#D1495B"), showlegend=False, row=1, col=1)
            for kk, ls in [(k, "solid")] + [(o, "dot") for o in ("rnn_simple", "lstm_brut", "hybride_gru") if o != k]:
                gs = lisser(saliency(mod, cfg, kk, x, int(P[kk].argmax()))) + 1e-30
                fg.add_scatter(x=T_S, y=gs, name=cfg["modeles"][kk]["nom"], line=dict(width=2 if kk == k else 1, dash=ls), row=2, col=1)
            fg.update_yaxes(type="log", row=2, col=1); fg.update_xaxes(title_text="temps (s) — la sortie est lue au dernier pas", row=2, col=1)
            fg.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15, font=dict(size=9))); st.plotly_chart(fg, width="stretch")
            st.caption("Si la courbe d'un RNN s'effondre vers le début de la fenêtre, ses premiers pas n'influencent pas la décision : c'est la disparition du gradient (§ vérification théorique). "
                       "L'hybride, qui réduit 1 250 pas à 78, garde une influence sur toute la fenêtre.")
        else:
            n_r = int((seq[:, 0] > 0).sum()); p0 = mod[k](sb[None], training=False).numpy()[0, kp]; imp = []
            for b in range(n_r):
                z = sb.copy(); z[b] = 0.0; imp.append(p0 - mod[k](z[None], training=False).numpy()[0, kp])
            fg = make_subplots(rows=3, cols=1, row_heights=[.4, .3, .3], vertical_spacing=.12,
                               subplot_titles=("Signal et battements détectés", "Suite (RR en s) — rempli de zéros, masqué", f"Importance de chaque battement pour p({CLASSES4[kp]})"))
            fg.add_scatter(x=T_S, y=x, line=dict(width=1, color="#2E86AB"), showlegend=False, row=1, col=1); fg.add_scatter(x=T_S[pics], y=x[pics], mode="markers", marker=dict(symbol="triangle-down", color="#D1495B"), showlegend=False, row=1, col=1)
            fg.add_bar(x=list(range(1, N_BATT + 1)), y=seq[:, 0], marker_color=["#EDAE49" if v > 0 else "#DDD" for v in seq[:, 0]], showlegend=False, row=2, col=1)
            fg.add_bar(x=list(range(1, n_r + 1)), y=imp, marker_color=["#D1495B" if v > 0 else "#66A182" for v in imp], showlegend=False, row=3, col=1)
            fg.update_xaxes(title_text="rang du battement", row=3, col=1); fg.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fg, width="stretch")
            st.caption("Importance = baisse de probabilité quand le battement est retiré de la suite (masqué). Un seul battement mal détecté fabrique de l'irrégularité : faiblesse de cette représentation.")


# ------------------------------------------------------------------ résultats
def courbes(hist, cfg, ks, titre):
    fg = make_subplots(rows=1, cols=2, subplot_titles=("perte de validation (pointillé : entraînement)", "F1 macro (validation)"))
    for kk in ks:
        h = hist[kk]; nom = cfg["modeles"][kk]["nom"]
        fg.add_scatter(y=h["perte_val"], name=nom, mode="lines", row=1, col=1, legendgroup=nom); fg.add_scatter(y=h["perte"], mode="lines", line=dict(dash="dot"), showlegend=False, legendgroup=nom, row=1, col=1)
        fg.add_scatter(y=h["f1_val"], name=nom, mode="lines", showlegend=False, legendgroup=nom, row=1, col=2)
    fg.update_xaxes(title_text="époque"); fg.update_yaxes(range=[0, 1.02], row=1, col=2); fg.update_layout(height=340, title=titre, margin=dict(t=70)); return fg


@st.fragment
def section_resultats(cfg, res, hist, journal, sc):
    M = cfg["modeles"]; princ = [k for k in M if k != "gru_battements_sans_masque"]
    q = st.segmented_control("Étape", ["Q1 · RNN sur le brut", "Q2 · Hybride vs CNN", "Q3 · Battements", "Q4 · Coût & embarqué", "Sélection & test", "Journal"], default="Q4 · Coût & embarqué", key="j4_q") or "Q4 · Coût & embarqué"
    if q == "Q1 · RNN sur le brut":
        st.plotly_chart(courbes(hist, cfg, ["rnn_simple", "lstm_brut"], "RNN simple vs LSTM sur 1 250 pas"), width="stretch")
        a, b = st.columns(2)
        for col, kk in [(a, "rnn_simple"), (b, "lstm_brut")]:
            G = pd.DataFrame(hist[kk]["grad"]); fg = go.Figure()
            for c in G.columns: fg.add_scatter(y=G[c].values + 1e-30, name="/".join(c.split("/")[-2:]), mode="lines+markers")
            fg.update_yaxes(type="log", title="‖∇‖"); fg.update_xaxes(title="époque (0 = init.)"); fg.update_layout(title=f"Gradient par variable — {M[kk]['nom']}", height=330, legend=dict(font=dict(size=9))); col.plotly_chart(fg, width="stretch")
        fg = go.Figure()
        for kk, dash in [("rnn_simple", "dot"), ("lstm_brut", "dash"), ("hybride_gru", "solid")]:
            fg.add_scatter(x=T_S, y=lisser(np.array(hist[kk]["gpas_fin"])) + 1e-30, name=M[kk]["nom"] + " (fin)", line=dict(dash=dash))
        fg.add_scatter(x=T_S, y=lisser(np.array(hist["hybride_gru"]["gpas_init"])) + 1e-30, name="Hybride (init.)", line=dict(color="gray", dash="longdash"))
        fg.update_yaxes(type="log", title="‖∂perte/∂x_t‖"); fg.update_xaxes(title="temps (s)"); fg.update_layout(height=360, title="Influence de chaque pas de la fenêtre sur la perte"); st.plotly_chart(fg, width="stretch")
        rows = []
        for kk in ("rnn_simple", "lstm_brut", "hybride_gru"):
            g = np.array(hist[kk]["gpas_fin"]); rows.append({"modèle": M[kk]["nom"], "‖g‖ pas 0-99": g[:100].mean(), "‖g‖ pas 1150-1249": g[-100:].mean(), "rapport début/fin": g[:100].mean() / (g[-100:].mean() + 1e-30)})
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:.2e}" for c in ["‖g‖ pas 0-99", "‖g‖ pas 1150-1249", "rapport début/fin"]}), hide_index=True)
        st.info("RNN simple : perte ≈ ln 5, gradient des 100 premiers pas nul ; LSTM : apprend un peu (budget 15 époques, une graine). Un pas d'apprentissage ×10 n'y change rien (essai infructueux journalisé).")
    elif q == "Q2 · Hybride vs CNN":
        st.plotly_chart(courbes(hist, cfg, ["cnn_seul", "hybride_gru", "hybride_lstm"], "CNN seul vs hybrides CNN+GRU / CNN+LSTM"), width="stretch")
        F = pd.DataFrame({M[k]["nom"]: M[k]["f1_classes"] for k in princ}, index=CLASSES4)
        fg = go.Figure([go.Bar(name=c, x=F.index, y=F[c]) for c in F.columns]); fg.update_layout(barmode="group", height=360, title="F1 par classe (validation)", yaxis=dict(range=[0, 1.02]), legend=dict(font=dict(size=9)))
        st.plotly_chart(fg, width="stretch")
        a, b, c = st.columns(3)
        for col, kk in zip((a, b, c), ("lstm_brut", "hybride_gru", "gru_battements")): col.plotly_chart(heat(M[kk]["cm"], CLASSES4, M[kk]["nom"]), width="stretch")
    elif q == "Q3 · Battements":
        st.plotly_chart(courbes(hist, cfg, ["gru_battements", "gru_battements_sans_masque"], "GRU sur suite de battements : avec / sans masquage"), width="stretch")
        ab = pd.DataFrame([{"modèle": M[k]["nom"], "F1 macro val": round(M[k]["f1_macro_val"], 4), "sens. FA@spec95": round(M[k]["sens_fa95_val"], 4), "paramètres": M[k]["params"]} for k in ("gru_battements", "gru_battements_sans_masque")])
        st.dataframe(ab, hide_index=True)
        st.markdown("**Stabilité (3 graines, validation)**"); st.dataframe(pd.DataFrame(res["stabilite"], index=["F1 macro moy.", "F1 macro σ", "sens. FA moy.", "sens. FA σ"]).T.round(4))
        st.info("Représentation 30× plus courte et 4 677 paramètres, mais F1 plafonné à ≈ 0,94–0,95 : dépend du détecteur de QRS (battements manqués ou faux) et perd la morphologie (onde P). Le masquage aide surtout la sensibilité FA (essai à une graine).")
    elif q == "Q4 · Coût & embarqué":
        T = pd.DataFrame(res["tableau"]).set_index("modèle"); st.dataframe(T, width="stretch")
        a, b, c = st.columns(3)
        for col, (champ, ti) in zip((a, b, c), (("paramètres", "paramètres (log)"), ("temps/époque (s)", "temps / époque (s, log)"), ("latence/fenêtre (ms)", "latence / fenêtre (ms, log ; cible 100)"))):
            fg = go.Figure(go.Bar(x=T[champ], y=T.index, orientation="h", marker_color=COUL[:1] * len(T))); fg.update_xaxes(type="log"); fg.update_layout(title=ti, height=300, margin=dict(l=10), yaxis=dict(showticklabels=(col is a)))
            if champ.startswith("latence"): fg.add_vline(x=100, line_dash="dash")
            col.plotly_chart(fg, width="stretch")
        st.info("Le nombre de paramètres ne mesure pas le coût : il faut le multiplier par la longueur de la séquence (1 250 pas pour le RNN brut, 78 pour l'hybride, ≤ 40 pour les battements). "
                "Latences mesurées sur le matériel du notebook (GPU Colab), pas sur la cible embarquée ; le RNN simple, sans noyau cuDNN, est très lent.")
    elif q == "Sélection & test":
        st.markdown(f"**Modèle retenu : {cfg['choix_nom']}** (règle fixée avant le test : meilleur F1 de validation parmi les modèles ≤ 2 Mo, puis plus faible latence si écart < 0,005).")
        ci = pd.DataFrame(res["cibles"]); c = st.columns(4)
        for col, (_, r) in zip(c, ci.iterrows()): col.metric(r["cible"], f"{r['mesuré']}", "atteinte" if r["atteint"] else "non atteinte", delta_color="normal" if r["atteint"] else "inverse")
        t = res["test"]; a, b = st.columns(2); a.plotly_chart(heat(t["cm"], CLASSES4, "Test (une seule fois)"), width="stretch")
        b.dataframe(pd.DataFrame({"F1": t["f1_classes"], "rappel": t["rappel_classes"]}, index=CLASSES4).round(3))
        b.caption(f"Seuil FA fixé sur la validation : sensibilité {t['sens_seuil_val']:.3f}, spécificité {t['spec_seuil_val']:.3f} sur le test.")
        from sklearn.metrics import roc_curve
        fg = go.Figure(); yb = (sc["y_val"] == 1).astype(int)
        for k in princ:
            fpr, tpr, _ = roc_curve(yb, sc["val_" + k][:, 1]); fg.add_scatter(x=fpr, y=tpr, name=M[k]["nom"], mode="lines")
        fg.add_vline(x=0.05, line_dash="dot", annotation_text="spécificité 95 %"); fg.update_layout(title="ROC FA (validation)", xaxis_title="1 − spécificité", yaxis_title="sensibilité", height=360); st.plotly_chart(fg, width="stretch")
        st.warning("Signaux simulés, Holter seul : la sensibilité FA est saturée à 1,000 (ne départage pas les modèles) ; la latence vient d'un GPU, pas du matériel embarqué.", icon=ICON_WARN)
    else:
        st.dataframe(journal, hide_index=True, width="stretch"); st.caption("Versions : " + " · ".join(f"{k} {v}" for k, v in cfg["versions"].items()))


# ------------------------------------------------------------------ théorie
@st.fragment
def section_theorie():
    st.markdown("#### Disparition / explosion du gradient : $\\lambda^{T}$")
    c = st.columns(2); lam = c[0].slider("Facteur λ par pas", 0.5, 1.2, 0.9, 0.01, key="j4_lam"); T = c[1].slider("Nombre de pas T", 10, 1250, 1250, 10, key="j4_T")
    lg = T * np.log10(lam); m = st.columns(3); m[0].metric(f"λ^{T}", f"{lam ** T:.2e}"); m[1].metric("log10", f"{lg:.1f}")
    m[2].metric("pas pour diviser par 2", "—" if lam >= 1 else f"{np.log(0.5) / np.log(lam):.1f}")
    st.caption("λ = 0,9 → 6,4×10⁻⁵⁸ (≈ float32 normal franchi après 829 pas) ; λ = 0,99 → 3,5×10⁻⁶ ; λ > 1 → explosion (d'où l'écrêtage `clipnorm`). Les portes du LSTM/GRU offrent un chemin additif de facteur ≈ 1.")
    ls = np.arange(1, 1251); fg = go.Figure()
    for l in (0.9, 0.99, lam): fg.add_scatter(x=ls, y=l ** ls, name=f"λ={l}")
    fg.update_yaxes(type="log", range=[-60, 2]); fg.update_xaxes(title="distance k à la sortie"); fg.update_layout(height=300, margin=dict(t=10)); st.plotly_chart(fg, width="stretch")
    st.markdown("#### Paramètres d'une couche récurrente (formulation Keras)")
    c = st.columns(2); u = c[0].slider("Unités u", 4, 128, 32, key="j4_u"); d = c[1].slider("Variables par pas d", 1, 8, 1, key="j4_d")
    from tensorflow import keras
    from tensorflow.keras import layers
    L = []
    for nom, couche, f in [("SimpleRNN", layers.SimpleRNN, u * (u + d) + u), ("LSTM", layers.LSTM, 4 * (u * (u + d) + u)), ("GRU (reset_after=True)", layers.GRU, 3 * (u * (u + d) + 2 * u))]:
        L.append({"couche": nom, "formule": f, "Keras count_params": keras.Sequential([layers.Input((None, d)), couche(u)]).count_params()})
    st.dataframe(pd.DataFrame(L), hide_index=True)
    st.latex(r"P_{LSTM}=4\,(u(u+d)+u)\;\Rightarrow\;4\,(32\times 33+32)=4\,352\quad(u=32,\ d=1)")


def render():
    if not (ART / "config.json").exists():
        st.error("Artefacts du jalon 4 introuvables (`artifacts/jalon4/`).", icon=ICON_WARN)
        st.markdown("Exécutez la cellule `export_jalon4.py` à la fin de `jalon4.ipynb`, puis copiez `artifacts/jalon4/` à côté de `app.py`."); return
    cfg, res, hist, mod, journal, ex, sc = charger()
    st.caption("RNN simple et LSTM sur 1 250 pas, hybride CNN→GRU, GRU sur suite de battements masquée · signaux simulés, aucune conclusion médicale.")
    vue = st.segmented_control("Vue", list(VUES), format_func=VUES.get, default="demo", key="j4_vue_v2") or "demo"
    if vue == "res": section_resultats(cfg, res, hist, journal, sc)
    elif vue == "theo": section_theorie()
    else: section_demo(cfg, res, hist, mod, ex, sc)