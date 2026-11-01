"""Onglet Jalon 3 : du Holter au patch (décalage de domaine, transfert, augmentation, multitâche)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from common.signal3 import (FS, N, NOMS3, T_S, TOUTES, normaliser_fenetres, augmenter_patch, mesures_fenetre)

ART = Path(__file__).resolve().parent.parent / "artifacts" / "jalon3"
LIB_AUG = {"gain": "gain", "derive": "dérive de ligne de base", "bruit": "bruit large bande", "secteur": "secteur 50 Hz"}

ICON_ALERTE, ICON_OK, ICON_WARN = ":material/notification_important:", ":material/check_circle:", ":material/warning:"
VUES = {"demo": ":material/science: Démonstration",
        "res": ":material/bar_chart: Résultats & expériences",
        "theo": ":material/function: Vérifications théoriques"}


def verdict(kk, p, nom):
    """Bandeau alerte / pas d'alerte sans emoji."""
    if kk != 0:
        st.error(f"Alerte — **{nom[kk]}** ({p[kk]:.1%})", icon=ICON_ALERTE)
    else:
        st.success(f"Pas d'alerte — **{nom[kk]}** ({p[kk]:.1%})", icon=ICON_OK)


@st.cache_resource(show_spinner="Chargement des modèles du jalon 3…")
def charger():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras
    cfg = json.load(open(ART / "config.json")); res = json.load(open(ART / "resultats_jalon3.json"))
    mod = {k: keras.models.load_model(ART / v["fichier"], compile=False) for k, v in cfg["modeles"].items()}
    return cfg, res, mod, pd.read_csv(ART / "journal.csv"), dict(np.load(ART / "exemples.npz", allow_pickle=True)), dict(np.load(ART / "spectres.npz"))


def sortie(cfg, m, x):
    """Probabilités de rythme (+ fréquence cardiaque prédite en bpm si le modèle est multitâche)."""
    o = m(x, training=False)
    if isinstance(o, dict):
        return o["rythme"].numpy(), o["fc"].numpy()[:, 0] * cfg["fc_sd"] + cfg["fc_mu"]
    return o.numpy(), None


def occlusion(cfg, m, xin, classe, fen=125, pas=25):
    p0 = sortie(cfg, m, xin)[0][0, classe]
    debuts = list(range(0, N - fen + 1, pas)); lot = np.repeat(xin, len(debuts), axis=0)
    for i, d in enumerate(debuts): lot[i, d:d + fen, 0] = 0.0
    chute = p0 - sortie(cfg, m, lot)[0][:, classe]
    cum, nb = np.zeros(N), np.zeros(N)
    for d, v in zip(debuts, chute): cum[d:d + fen] += v; nb[d:d + fen] += 1
    return cum / np.maximum(nb, 1)


def heat(cm, titre):
    cm = np.array(cm); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    f = go.Figure(go.Heatmap(z=cmn, x=NOMS3, y=NOMS3, colorscale="Blues", zmin=0, zmax=1, showscale=False,
                             text=[[f"{cm[i, j]}<br>({cmn[i, j]:.0%})" for j in range(5)] for i in range(5)], texttemplate="%{text}"))
    f.update_yaxes(autorange="reversed", title="réel"); f.update_xaxes(title="prédit")
    f.update_layout(title=titre, height=380, margin=dict(l=10, r=10, t=45, b=10)); return f


def cle_par_titre(cfg, titre): return next((k for k, v in cfg["modeles"].items() if v["titre"] == titre), None)


# ------------------------------------------------------------------ démonstration
# @st.fragment : un changement de widget ne relance que ce bloc -> la page ne remonte plus en haut.
@st.fragment
def section_demo(cfg, res, mod, ex, sp):
    g, d = st.columns([1, 2.3], gap="large")
    with g:
        st.subheader("1 · Signal")
        src = st.radio("Source", ["Patch (test)", "Holter (test)", "Holter + augmentation « patch »", "Fenêtre inédite (sans étiquette)", "Fichier .csv / .npy"], key="j3_src")
        x = None; vrai = None; fc_ann = None
        if src in ("Patch (test)", "Holter (test)", "Holter + augmentation « patch »"):
            app = "Patch" if src.startswith("Patch") else "Holter"
            cl = st.selectbox("Rythme réel", range(5), format_func=lambda k: NOMS3[k], key="j3_cl")
            idx = np.where((ex["appareil"] == app) & (ex["y"] == cl))[0]
            i = idx[st.slider("Fenêtre n°", 1, len(idx), 1, key="j3_i") - 1]
            x = ex["X"][i].astype("float32"); vrai = int(ex["y"][i]); fc_ann = float(ex["fc"][i]); st.caption(f"Origine : {ex['source'][i]}")
            if src.startswith("Holter +"):
                act = st.multiselect("Transformations", TOUTES, default=list(TOUTES), format_func=LIB_AUG.get, key="j3_act")
                graine = st.number_input("Graine", 0, 9999, 0, key="j3_seed")
                x = augmenter_patch(x[None], np.random.default_rng(int(graine)), tuple(act), cfg["params_aug"])[0]
                st.caption("Paramètres déduits des mesures du patch : " + ", ".join(f"{k}={v if not isinstance(v, list) else tuple(round(a, 2) for a in v)}" for k, v in cfg["params_aug"].items()))
        elif src.startswith("Fenêtre inédite"):
            idx = np.where(ex["appareil"] == "Inédit")[0]; i = idx[st.slider("Fenêtre n°", 1, len(idx), 1, key="j3_ii") - 1]
            x = ex["X"][i].astype("float32"); fc_ann = float(ex["fc"][i]); st.caption(f"{ex['source'][i]} — pas de classe connue.")
        else:
            f = st.file_uploader("1 250 points (10 s, 125 Hz, mV)", type=["csv", "npy", "txt"], key="j3_up")
            if f is None: st.info("Déposez un fichier."); return
            try: a = np.load(f).ravel() if f.name.endswith(".npy") else np.loadtxt(f, delimiter=",").ravel()
            except Exception as e: st.error(f"Lecture impossible : {e}"); return
            if a.size != N: st.error(f"{a.size} valeurs reçues, {N} attendues."); return
            x = a.astype("float32")

        st.subheader("2 · Stratégie")
        cles = [cfg["cle_retenue"]] + [k for k in cfg["modeles"] if k != cfg["cle_retenue"]]
        cle = st.selectbox("Modèle", cles, format_func=lambda k: ("★ " if k == cfg["cle_retenue"] else "") + cfg["modeles"][k]["titre"], key="j3_mod")
        st.caption("Les stratégies d'adaptation sont ici ré-entraînées sur les 18 patients annotés du patch (réglages du notebook, graine 0). "
                   "★ = stratégie retenue ; elle est comparée à « Holter tel quel » (aucune adaptation).")

    with d:
        xin = normaliser_fenetres(x[None]); k0 = cle_par_titre(cfg, "Holter tel quel") or cles[-1]
        p_sel, fc_sel = sortie(cfg, mod[cle], xin); p_ref, _ = sortie(cfg, mod[k0], xin); p_sel, p_ref = p_sel[0], p_ref[0]
        a, b = int(p_ref.argmax()), int(p_sel.argmax())
        st.subheader("3 · Décision")
        c = st.columns(2)
        for col, titre, kk, p in [(c[0], "Holter tel quel (sans adaptation)", a, p_ref), (c[1], cfg["modeles"][cle]["titre"], b, p_sel)]:
            with col:
                st.markdown(f"**{titre}**")
                verdict(kk, p, NOMS3)
        if vrai is not None and vrai >= 0:
            st.caption(f"Rythme réel : **{NOMS3[vrai]}** — sans adaptation : {'correct' if a == vrai else 'erroné'} · stratégie choisie : {'correct' if b == vrai else 'erroné'}")
        fig = go.Figure(); fig.add_bar(name="Holter tel quel", x=NOMS3, y=p_ref, marker_color="#B0B0B0"); fig.add_bar(name="modèle choisi", x=NOMS3, y=p_sel, marker_color="#2E86AB")
        fig.update_layout(barmode="group", height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.05], title="probabilité")); st.plotly_chart(fig, width="stretch")
        if fc_sel is not None:
            m = st.columns(3); m[0].metric("FC prédite (tête auxiliaire)", f"{fc_sel[0]:.0f} bpm")
            if fc_ann is not None: m[1].metric("FC annotée", f"{fc_ann:.0f} bpm"); m[2].metric("Écart", f"{abs(fc_sel[0]-fc_ann):.1f} bpm")
            st.caption("Tête « fc » du multitâche : erreur ≈ 2–4 bpm sur le Holter mais ≈ 10 bpm sur le patch (elle ne se transfère pas sans adaptation).")

        occ = occlusion(cfg, mod[cle], xin, b)
        fg = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.68, .32], vertical_spacing=.08,
                           subplot_titles=("Signal brut (mV)", f"Sensibilité temporelle (occlusion 1 s) — p({NOMS3[b]})"))
        fg.add_scatter(x=T_S, y=x, line=dict(width=1, color="#2E86AB"), showlegend=False, row=1, col=1)
        fg.add_scatter(x=T_S, y=occ, fill="tozeroy", line=dict(width=1, color="#EDAE49"), showlegend=False, row=2, col=1)
        fg.update_xaxes(title_text="temps (s)", row=2, col=1); fg.update_layout(height=440, margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fg, width="stretch")

        car = {r["index"]: r for r in res["_tables"]["carac"]}; mes = mesures_fenetre(x.astype("float64")); L = []
        cle_map = {"crête (centile 99,5 de |x|, mV)": "crête (centile 99,5 de |x|, mV)", "dérive < 0,5 Hz (écart-type, mV)": "dérive < 0,5 Hz (écart-type, mV)",
                   "raie 50 Hz : amplitude (mV)": "raie 50 Hz : amplitude médiane (mV)", "rapport crête QRS / bruit > 30 Hz": "rapport crête QRS / bruit > 30 Hz",
                   "aplatissement (kurtosis)": "aplatissement (kurtosis) moyen"}
        votes = []
        for nom, val in mes.items():
            r = car[cle_map[nom]]; h, p = r["Holter"], r["patch"]
            plus_proche = "Holter" if abs(np.log((abs(val) + 1e-6) / (abs(h) + 1e-6))) < abs(np.log((abs(val) + 1e-6) / (abs(p) + 1e-6))) else "patch"
            if nom.startswith(("crête", "rapport", "aplat")): votes.append(plus_proche)
            L.append({"mesure": nom, "cette fenêtre": round(val, 4), "Holter (moy.)": round(h, 4), "patch (moy.)": round(p, 4), "plus proche de": plus_proche})
        verdict_dom = "patch" if votes.count("patch") > votes.count("Holter") else "Holter"
        st.markdown(f"**Diagnostic de décalage de domaine** — amplitude, rapport signal/bruit et morphologie évoquent plutôt un enregistrement de type **{verdict_dom}** (indicatif, fenêtre isolée).")
        st.dataframe(pd.DataFrame(L), hide_index=True, width="stretch")
        with st.expander("Comparer toutes les stratégies sur cette fenêtre"):
            rows = []
            for k in cles:
                pp, ff = sortie(cfg, mod[k], xin); rows.append({"modèle": cfg["modeles"][k]["titre"], "prédiction": NOMS3[int(pp[0].argmax())], "confiance": round(float(pp[0].max()), 3),
                                                            **{f"p({n})": round(float(v), 3) for n, v in zip(NOMS3, pp[0])}, "FC prédite": None if ff is None else round(float(ff[0]), 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ------------------------------------------------------------------ résultats
@st.fragment
def section_resultats(cfg, res, journal, sp):
    T = res["_tables"]; R = {k: v for k, v in res.items() if k != "_tables"}
    q = st.segmented_control("Étape", ["Q3 · Décalage", "Q1 · Stratégies", "Q2 · Augmentation", "Q4 · Multitâche (λ)", "Choix, test & annotation", "Journal"], default="Q1 · Stratégies", key="j3_q") or "Q1 · Stratégies"
    if q == "Q3 · Décalage":
        df = pd.DataFrame(T["carac"]).set_index("index"); st.dataframe(df.round(4), width="stretch")
        a, b = st.columns(2); f = sp["f"][1:]
        fg = go.Figure()
        for n, c in [("Holter", "#2E86AB"), ("patch", "#EDAE49"), ("inédits", "#66A182")]: fg.add_scatter(x=f, y=sp["raw_" + n][1:], name=n, line=dict(color=c))
        fg.add_vline(x=50, line_dash="dot"); fg.update_xaxes(type="log", title="Hz"); fg.update_yaxes(type="log", title="DSP (mV²/Hz)"); fg.update_layout(title="Spectre moyen (signal brut)", height=340); a.plotly_chart(fg, width="stretch")
        fg = go.Figure()
        for n, c in [("Holter", "#2E86AB"), ("patch", "#EDAE49"), ("inédits", "#66A182")]: fg.add_scatter(x=f, y=sp["norm_" + n][1:], name=n, line=dict(color=c))
        fg.update_xaxes(type="log", title="Hz"); fg.update_yaxes(type="log"); fg.update_layout(title="Après normalisation par fenêtre (ce que voit le modèle)", height=340); b.plotly_chart(fg, width="stretch")
        fg = go.Figure()
        for n, c in [("Holter", "#2E86AB"), ("patch", "#EDAE49")]:
            m, s = sp["cplx_mean_" + n], sp["cplx_std_" + n]
            fg.add_scatter(x=sp["t_ms"], y=m, name=f"{n} (≈ {T['largeurs'][n]:.0f} ms à mi-hauteur)", line=dict(color=c))
        fg.update_xaxes(title="ms autour du pic"); fg.update_layout(title="Complexe moyen, rythme sinusal (amplitude normalisée)", height=320); st.plotly_chart(fg, width="stretch")
        st.markdown("""| Différence | Stratégie associée |
|---|---|
| Amplitude (crête ×0,58) | z-score par fenêtre ; gain dans l'augmentation (seulement via le rapport signal/bruit) |
| Rapport signal/bruit (×0,33), bruit large bande | bruit + gain dans l'augmentation |
| Dérive lente | dérive dans l'augmentation ; passe-haut possible (non testé) |
| Raie 50 Hz | plus faible sur le patch : ne justifie pas, à elle seule, la transformation « secteur » |
| Morphologie (complexes moins « pointus ») | non simulable : **ajustement fin** avec des annotations du patch |""")
    elif q == "Q1 · Stratégies":
        noms = sorted(R, key=lambda n: R[n]["f1"])
        fg = go.Figure(go.Bar(x=[R[n]["f1"] for n in noms], y=noms, orientation="h", error_x=dict(array=[R[n]["std"] for n in noms]),
                              marker_color=["#D1495B" if n == cfg["retenu"] else "#2E86AB" for n in noms]))
        fg.add_vline(x=0.85, line_dash="dash", line_color="red", annotation_text="cible finale 0,85"); fg.update_xaxes(range=[0, 1], title="F1 macro (CV par patient, patch, 12 patients)")
        fg.update_layout(height=560, margin=dict(l=10), title="Stratégies d'adaptation (barres : écart-type entre graines)"); st.plotly_chart(fg, width="stretch")
        df = pd.DataFrame([{"stratégie": n, "F1 CV": round(R[n]["f1"], 3), "σ graines": round(R[n]["std"], 3), "σ plis": round(R[n]["std_plis"], 3),
                            **{f"R_{c}": round(v, 2) for c, v in zip(NOMS3, R[n]["rappels"])}} for n in R]); st.dataframe(df, hide_index=True, width="stretch")
        st.info("Le σ entre plis (3 patients par pli) domine le σ entre graines (2 graines) : la variabilité entre patients est sous-estimée par les barres.")
    elif q == "Q2 · Augmentation":
        n4 = ["Holter tel quel", "Holter augmenté, tel quel", "Ajustement fin complet", "Holter augmenté + ajustement fin"]
        fg = go.Figure(go.Bar(x=n4, y=[R[n]["f1"] for n in n4], error_y=dict(array=[R[n]["std"] for n in n4]), marker_color="#2E86AB"))
        fg.update_yaxes(range=[0, 1], title="F1 CV"); fg.update_layout(height=340, title="Augmentation vs ajustement fin vs combinaison"); st.plotly_chart(fg, width="stretch")
        ab = pd.DataFrame(T["abl_aug"]); st.markdown("**Ablation (une graine, Holter tel quel, 18 patients du patch)**"); st.dataframe(ab.round(3), hide_index=True, width="stretch")
        st.info("L'augmentation « toutes » comble l'essentiel de l'écart sans aucune annotation du patch. L'ablation transformation par transformation est dans le bruit (le gain seul devrait être neutre après normalisation, pourtant −0,09).")
    elif q == "Q4 · Multitâche (λ)":
        lam = pd.DataFrame(T["lam"]); st.dataframe(lam.round(3), hide_index=True, width="stretch")
        fg = go.Figure()
        fg.add_scatter(x=lam["lam"], y=lam["F1_CV_ajust_fin"], mode="lines+markers", name="après ajustement fin", error_y=dict(array=lam["ecart_type"]))
        fg.add_scatter(x=lam["lam"], y=lam["F1_CV_tel_quel"], mode="lines+markers", name="Holter tel quel")
        fg.update_xaxes(title="λ (poids de la tête fc)"); fg.update_yaxes(title="F1 CV patch"); fg.update_layout(height=340); st.plotly_chart(fg, width="stretch")
        st.info("λ intermédiaire (0,5) : gain suggéré mais pas démontré (2 graines, courbe non monotone). La tête fc est bien apprise sur le Holter, pas transférée au patch.")
    elif q == "Choix, test & annotation":
        st.markdown("**Liste restreinte confirmée sur la validation (6 patients)**"); st.dataframe(pd.DataFrame(T["val"]).round(3), hide_index=True, width="stretch")
        ci = pd.DataFrame(T["cible"]); c = st.columns(2)
        for col, (_, r) in zip(c, ci.iterrows()): col.metric(r["Mesure"], f"{r['Test']:.3f}", f"{r['Écart']:+.3f} vs cible {r['Cible finale']}", delta_color="normal")
        st.caption(f"IC 95 % (bootstrap par patient) du F1 patch : [{T['ic_test'][0]:.3f} ; {T['ic_test'][1]:.3f}] — test utilisé une seule fois.")
        a, b = st.columns(2); a.plotly_chart(heat(T["cm_test"], "Modèle retenu — test patch"), width="stretch")
        fg = go.Figure(go.Bar(x=NOMS3, y=T["rappels_test"], marker_color="#2E86AB")); fg.update_yaxes(range=[0, 1.05], title="rappel"); fg.update_layout(title="Rappel par classe (test patch)", height=380); b.plotly_chart(fg, width="stretch")
        st.warning("Erreur dominante : fausses alertes sur le rythme sinusal (≈ 32 % des fenêtres sinusales du patch).", icon=ICON_WARN)
        cb = pd.DataFrame(T["courbe"]); fg = go.Figure()
        for n, c in [("stratégie retenue", "#2E86AB"), ("patch seul", "#EDAE49")]:
            g = cb[cb["stratégie"] == n]; fg.add_scatter(x=g["n_patients"], y=g["mean"], error_y=dict(array=g["std"].fillna(0)), mode="lines+markers", name=n, line=dict(color=c))
        fg.add_hline(y=T["zs_val"], line_dash="dot", annotation_text="Holter tel quel (0 patient)"); fg.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="cible 0,85")
        fg.update_xaxes(title="patients du patch annotés"); fg.update_yaxes(range=[0, 1], title="F1 (validation, 6 patients)"); fg.update_layout(height=380, title="Courbe d'apprentissage : combien de patients annoter ?")
        st.plotly_chart(fg, width="stretch")
        ef = pd.DataFrame(T["effectifs"]).set_index("row_0") if "row_0" in T["effectifs"][0] else pd.DataFrame(T["effectifs"]).set_index(list(T["effectifs"][0])[0])
        st.markdown("**Fenêtres par patient et rythme (patch, entraînement)** — la couverture des rythmes pèse plus que le nombre de patients."); st.dataframe(ef, width="stretch")
    else:
        ess = ["(tous)"] + sorted(journal["essai"].unique()); e = st.selectbox("Type d'essai", ess, key="j3_ess")
        st.dataframe(journal if e == "(tous)" else journal[journal["essai"] == e], hide_index=True, width="stretch")
        st.caption("Versions : " + " · ".join(f"{k} {v}" for k, v in cfg["versions"].items()))


# ------------------------------------------------------------------ théorie
@st.fragment
def section_theorie(cfg, mod, ex):
    st.markdown("#### Normalisation par fenêtre : $z=(x-\\mu)/\\sigma$ est insensible à $x'=a\\,x+b$ ($a>0$)")
    c = st.columns(4)
    app = c[0].selectbox("Fenêtre", ["Patch", "Holter"], key="j3_t_app"); a = c[1].slider("Gain a", -1.0, 3.0, 0.3, 0.1, key="j3_t_a")
    b = c[2].slider("Décalage b (mV)", -5.0, 5.0, 1.0, 0.5, key="j3_t_b"); bruit = c[3].slider("Bruit fixe (σ, mV)", 0.0, 0.2, 0.0, 0.01, key="j3_t_n")
    i = np.where(ex["appareil"] == app)[0][0]; x = ex["X"][i].astype("float32")
    xp = a * x + b + np.random.default_rng(0).normal(0, bruit, x.shape).astype("float32")
    z0, z1 = normaliser_fenetres(x[None]), normaliser_fenetres(xp[None])
    cle = cle_par_titre(cfg, "Holter tel quel") or list(cfg["modeles"])[0]
    p0, p1 = sortie(cfg, mod[cle], z0)[0], sortie(cfg, mod[cle], z1)[0]
    m = st.columns(2); m[0].metric("max |z' − z|", f"{np.abs(z1 - z0).max():.2e}"); m[1].metric("max |p' − p|", f"{np.abs(p1 - p0).max():.2e}")
    if a > 0 and bruit == 0: st.success("a > 0, pas de bruit : z' = z, probabilités identiques (au ε = 1e-6 près).", icon=ICON_OK)
    elif a < 0: st.error("a < 0 (inversion de polarité) : z' = −z, pas d'invariance.", icon=ICON_WARN)
    elif bruit > 0: st.warning("Un bruit d'amplitude absolue fixe n'est pas mis à l'échelle comme le signal : le gain modifie le rapport signal/bruit, donc z.", icon=ICON_WARN)
    fg = go.Figure(); fg.add_scatter(x=T_S, y=z0[0, :, 0], name="z (original)", line=dict(width=1)); fg.add_scatter(x=T_S, y=z1[0, :, 0], name="z' (transformé)", line=dict(width=1, dash="dot"))
    fg.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10)); st.plotly_chart(fg, width="stretch")

    st.markdown("#### Paramètres entraînables du CNN de départ")
    m0 = mod[cle]; conv = [l.name for l in m0.layers if "conv" in l.name]
    lignes = [{"couche": l.name, "paramètres": l.count_params()} for l in m0.layers if l.count_params()]
    st.dataframe(pd.DataFrame(lignes), hide_index=True)
    ent = {"Extraction de caractéristiques (tout gelé sauf `rythme`)": {"rythme"}, "Ajustement fin partiel (dernier bloc conv + `representation` + sortie)": {conv[-1], "representation", "rythme"},
           "Ajustement fin complet": {l.name for l in m0.layers}}
    st.dataframe(pd.DataFrame([{"mode": k, "paramètres entraînables": sum(l.count_params() for l in m0.layers if l.name in v)} for k, v in ent.items()]), hide_index=True)
    st.caption("Calcul à la main : 128 + 3 616 + 14 400 + 28 736 + 4 160 + 325 = 51 365 ; en extraction : 64×5 + 5 = 325.")


def render():
    if not (ART / "config.json").exists():
        st.error("Artefacts du jalon 3 introuvables (`artifacts/jalon3/`).", icon=ICON_WARN)
        st.markdown("Exécutez la cellule `export_jalon3.py` à la fin de `jalon3.ipynb`, puis copiez `artifacts/jalon3/` à côté de `app.py`."); return
    cfg, res, mod, journal, ex, sp = charger()
    st.caption("Décalage de domaine, extraction de caractéristiques, ajustement fin, augmentation et multitâche (rythme + fréquence cardiaque) · signaux simulés, aucune conclusion médicale.")
    vue = st.segmented_control("Vue", list(VUES), format_func=VUES.get, default="demo", key="j3_vue_v2") or "demo"
    if vue == "res": section_resultats(cfg, res, journal, sp)
    elif vue == "theo": section_theorie(cfg, mod, ex)
    else: section_demo(cfg, res, mod, ex, sp)