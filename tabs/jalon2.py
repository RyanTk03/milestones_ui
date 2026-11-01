"""Onglet Jalon 2 : apprendre sur le signal brut (CNN 1D) et débruiter le patch (auto-encodeurs)."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from common.signal2 import (FS, N, NOMS2, TEMPS, normaliser_fenetres, normaliser_hp, bruit_calibre,
                            descripteurs_fenetre, champ_receptif, couches_cnn)

ART = Path(__file__).resolve().parent.parent / "artifacts" / "jalon2"
COUL = ["#2E86AB", "#D1495B", "#EDAE49", "#66A182", "#8D6A9F"]
AE_TITRES = {"AE_V1_calibre": "AE_V1 (conv) · bruit calibré", "AE_V2_calibre": "AE_V2 (U-Net) · bruit calibré",
             "AE_V2_depart": "AE_V2 (U-Net) · bruit de départ [contrôle]"}


@st.cache_resource(show_spinner="Chargement des modèles du jalon 2…")
def charger():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras
    cfg = json.load(open(ART / "config.json")); res = json.load(open(ART / "resultats_jalon2.json"))
    mod = {p.stem: keras.models.load_model(p, compile=False) for p in sorted(ART.glob("*.keras"))}
    return cfg, res, mod, pd.read_csv(ART / "journal_experiences.csv"), dict(np.load(ART / "exemples.npz", allow_pickle=True))


def titre_cls(k, cfg):
    if k == "mlp_descripteurs": return "MLP sur 13 descripteurs RR (comparateur jalon 1)"
    if k == "cnn_augmente": return "CNN retenu entraîné sur Holter bruité (exploratoire)"
    if k == "d4_k7_gap_passehaut": return "d4_k7_gap + passe-haut 0,5 Hz avant z-score"
    c = cfg["configs"][k]
    return f"CNN {k} · {c['profondeur']} blocs, noyau {c['noyau']}, {'moyenne globale' if c['agregation']=='gap' else 'aplatissement'}" + (f", dilatations {c['dilatations']}" if c["dilatations"] else "")


# ------------------------------------------------------------------ chaîne de traitement
def debruiter_1d(ae, x):
    """Entrée AE : fenêtre normalisée ; sortie : signal débruité 1D (le notebook le renormalise ensuite)."""
    return ae(normaliser_fenetres(x[None]), training=False).numpy()[0, :, 0]


def entree_classifieur(cle, sig, cfg):
    if cle == "mlp_descripteurs":
        m = cfg["mlp"]; d = np.array(descripteurs_fenetre(sig), dtype="float64")
        d = np.where(np.isnan(d), np.array(m["med"]), d)
        return ((d - np.array(m["mean"])) / np.array(m["scale"]))[None].astype("float32")
    return normaliser_hp(sig[None]) if cle == "d4_k7_gap_passehaut" else normaliser_fenetres(sig[None])


def classifier(mod, cfg, cle, sig):
    t0 = time.perf_counter(); xin = entree_classifieur(cle, sig, cfg)
    p = mod[cle](xin, training=False).numpy()[0]
    return p, xin, 1000 * (time.perf_counter() - t0)


def occlusion(model, xin, classe, fen=125, pas=25):
    """Sensibilité temporelle : baisse de p(classe) quand une fenêtre de 1 s est remplacée par la moyenne (0)."""
    p0 = model(xin, training=False).numpy()[0, classe]
    debuts = list(range(0, N - fen + 1, pas)); lot = np.repeat(xin, len(debuts), axis=0)
    for i, d in enumerate(debuts): lot[i, d:d + fen, 0] = 0.0
    chute = p0 - model(lot, training=False).numpy()[:, classe]
    cum, nb = np.zeros(N), np.zeros(N)
    for d, v in zip(debuts, chute): cum[d:d + fen] += v; nb[d:d + fen] += 1
    return cum / np.maximum(nb, 1)


def heat(cm, noms, titre):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    f = go.Figure(go.Heatmap(z=cmn, x=noms, y=noms, colorscale="Blues", zmin=0, zmax=1, showscale=False,
                             text=[[f"{cm[i, j]}" for j in range(5)] for i in range(5)], texttemplate="%{text}"))
    f.update_yaxes(autorange="reversed", title="réel"); f.update_xaxes(title="prédit")
    f.update_layout(title=titre, height=340, margin=dict(l=10, r=10, t=45, b=10)); return f


# ------------------------------------------------------------------ démonstration
def section_demo(cfg, res, mod, ex):
    g, d = st.columns([1, 2.3], gap="large")
    with g:
        st.subheader("1 · Signal")
        src = st.radio("Source", ["Holter (test)", "Patch (test / validation)", "Holter + bruit patch simulé", "Fichier .csv / .npy"], key="j2_src")
        x = None; vrai = None
        if src != "Fichier .csv / .npy":
            app = "Patch" if src.startswith("Patch") else "Holter"
            cl = st.selectbox("Rythme réel", range(5), format_func=lambda k: NOMS2[k], key="j2_cl")
            idx = np.where((ex["appareil"] == app) & (ex["y"] == cl))[0]
            i = idx[st.slider("Fenêtre n°", 1, len(idx), 1, key="j2_i") - 1]
            x = ex["X"][i].astype("float32"); vrai = int(ex["y"][i]); st.caption(f"Origine : {ex['source'][i]} · patient {ex['patient'][i]}")
            if src.startswith("Holter +"):
                graine = st.number_input("Graine du bruit", 0, 9999, 1, key="j2_seed")
                x = bruit_calibre(x[None], np.random.default_rng(int(graine)), cfg["params_bruit"])[0]
                st.caption("Bruit de type patch calibré (gain, dérive, 50 Hz, bruit musculaire) ajouté au Holter propre.")
        else:
            f = st.file_uploader("1 250 points (10 s, 125 Hz, mV)", type=["csv", "npy", "txt"], key="j2_up")
            if f is None: st.info("Déposez un fichier."); return
            try: a = np.load(f).ravel() if f.name.endswith(".npy") else np.loadtxt(f, delimiter=",").ravel()
            except Exception as e: st.error(f"Lecture impossible : {e}"); return
            if a.size != N: st.error(f"{a.size} valeurs reçues, {N} attendues."); return
            x = a.astype("float32")

        st.subheader("2 · Chaîne de traitement")
        cles_cls = [cfg["nom_cnn"]] + [k for k in cfg["configs"] if k != cfg["nom_cnn"]] + ["d4_k7_gap_passehaut", "cnn_augmente", "mlp_descripteurs"]
        cls = st.selectbox("Classifieur", cles_cls, format_func=lambda k: ("★ " if k == cfg["nom_cnn"] else "") + titre_cls(k, cfg), key="j2_cls")
        aes = [k for k in AE_TITRES if k in mod]
        ae_k = st.selectbox("Débruiteur", aes, index=aes.index("AE_V2_calibre") if "AE_V2_calibre" in aes else 0,
                            format_func=lambda k: AE_TITRES[k], key="j2_ae")
        st.caption("Les CNN ont tous été appris sur le Holter propre ; la comparaison « sans / avec débruitage » est affichée côte à côte.")

    with d:
        sig0 = x.astype("float64"); sig_ae = debruiter_1d(mod[ae_k], x).astype("float64")
        t0 = time.perf_counter(); debruiter_1d(mod[ae_k], x); t_ae = 1000 * (time.perf_counter() - t0)
        p0, xin0, t0c = classifier(mod, cfg, cls, sig0); p1, xin1, t1c = classifier(mod, cfg, cls, sig_ae)
        k0, k1 = int(p0.argmax()), int(p1.argmax())

        st.subheader("3 · Décision")
        c = st.columns(2)
        for col, titre, k, p in [(c[0], "Sans débruitage", k0, p0), (c[1], f"Avec débruitage ({AE_TITRES[ae_k].split(' ·')[0]})", k1, p1)]:
            with col:
                st.markdown(f"**{titre}**")
                (st.error if k != 0 else st.success)(("🚨 Alerte — " if k != 0 else "✅ Pas d'alerte — ") + f"**{NOMS2[k]}** ({p[k]:.1%})")
        if vrai is not None:
            st.caption(f"Rythme réel : **{NOMS2[vrai]}** — sans débruitage : {'✔' if k0 == vrai else '✘'} · avec : {'✔' if k1 == vrai else '✘'}")

        fig = go.Figure()
        fig.add_bar(name="sans débruitage", x=NOMS2, y=p0, marker_color="#B0B0B0"); fig.add_bar(name="avec débruitage", x=NOMS2, y=p1, marker_color="#2E86AB")
        fig.update_layout(barmode="group", height=250, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.05], title="probabilité"))
        st.plotly_chart(fig, width="stretch")

        n0 = normaliser_fenetres(sig0[None])[0, :, 0]; n1 = normaliser_fenetres(sig_ae[None])[0, :, 0]
        occ_ok = cls != "mlp_descripteurs"
        fg = make_subplots(rows=2 if occ_ok else 1, cols=1, shared_xaxes=True, vertical_spacing=.08, row_heights=[.7, .3] if occ_ok else [1],
                           subplot_titles=("Signal normalisé : entrée vs sortie de l'auto-encodeur", f"Sensibilité temporelle (occlusion 1 s) — p({NOMS2[k1]}) avec débruitage") if occ_ok else None)
        fg.add_scatter(x=TEMPS, y=n0, name="entrée (normalisée)", line=dict(width=1, color="#D1495B"), opacity=.55, row=1, col=1)
        fg.add_scatter(x=TEMPS, y=n1, name="sortie AE", line=dict(width=1.4, color="#2E86AB"), row=1, col=1)
        if occ_ok:
            imp = occlusion(mod[cls], xin1, k1)
            fg.add_scatter(x=TEMPS, y=imp, fill="tozeroy", line=dict(width=1, color="#EDAE49"), name="importance", row=2, col=1)
            fg.update_yaxes(title_text="Δ proba", row=2, col=1)
        fg.update_xaxes(title_text="temps (s)", row=2 if occ_ok else 1, col=1)
        fg.update_layout(height=470 if occ_ok else 330, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fg, width="stretch")

        m = st.columns(4)
        m[0].metric("Latence AE", f"{t_ae:.1f} ms"); m[1].metric("Latence classifieur", f"{t1c:.1f} ms")
        m[2].metric("Paramètres AE", f"{mod[ae_k].count_params():,}".replace(",", " ")); m[3].metric("Paramètres classifieur", f"{mod[cls].count_params():,}".replace(",", " "))
        st.caption("Latences Keras sur le serveur de l'app (cible embarquée ≤ 100 ms : voir « Test final & embarquement » pour les mesures TFLite).")
        with st.expander("Comparer tous les classifieurs sur cette fenêtre"):
            L = []
            for k in cles_cls:
                a, _, _ = classifier(mod, cfg, k, sig0); b, _, _ = classifier(mod, cfg, k, sig_ae)
                L.append({"classifieur": titre_cls(k, cfg), "sans débruitage": NOMS2[int(a.argmax())], "conf. sans": round(float(a.max()), 3),
                          "avec débruitage": NOMS2[int(b.argmax())], "conf. avec": round(float(b.max()), 3)})
            st.dataframe(pd.DataFrame(L), hide_index=True, width="stretch")


# ------------------------------------------------------------------ résultats
def section_resultats(cfg, res, journal):
    q = st.segmented_control("Étape", ["Q1 · CNN vs MLP", "Q2 · Ablations", "Bruit patch", "Q3 · Auto-encodeurs", "Q4 · Débruitage", "Test final & embarquement", "Journal"],
                             default="Q4 · Débruitage", key="j2_q")
    if q == "Q1 · CNN vs MLP":
        t = pd.DataFrame(res["comparaison_classes"]).set_index("classe")
        fig = go.Figure([go.Bar(name=c, x=t.index, y=t[c]) for c in t.columns]); fig.update_layout(barmode="group", height=380, title="F1 par classe (validation)")
        st.plotly_chart(fig, width="stretch"); st.dataframe(t, width="stretch")
        st.info("Sur le Holter le CNN dépasse le MLP (surtout FA) ; sur le patch brut les deux s'effondrent : c'est le décalage d'appareil que l'auto-encodeur doit combler.")
    elif q == "Q2 · Ablations":
        rows = [{"essai": k, "params": v.get("params"), "RF (éch.)": v.get("rf"), "RF (s)": None if v.get("rf") is None else round(v["rf"] / FS, 2),
                 "F1 Holter val": round(v["holter"]["f1_macro"], 3), "F1 patch val": round(v["patch"]["f1_macro"], 3)} for k, v in res["cnn_val"].items()]
        df = pd.DataFrame(rows); st.dataframe(df, hide_index=True, width="stretch")
        d2 = df.dropna(subset=["RF (éch.)"])
        fig = px.scatter(d2, x="RF (s)", y="F1 Holter val", size="params", text="essai", color=[("retenu" if e == res["cnn_retenu"] else "autres") for e in d2.essai],
                         color_discrete_map={"retenu": "#D1495B", "autres": "#2E86AB"})
        fig.add_vline(x=214 / FS, line_dash="dash", annotation_text="2 RR à 70 bpm (1,71 s)"); fig.update_traces(textposition="top center")
        fig.update_layout(height=430, title="F1 Holter vs champ réceptif (taille = paramètres)"); st.plotly_chart(fig, width="stretch")
        st.info("Tendance, pas loi : sous 214 éch. les modèles sont nettement moins bons ; au-delà, plus de champ ne change presque rien. Une seule graine : écarts de 0,005–0,01 non conclusifs.")
    elif q == "Bruit patch":
        t = pd.DataFrame(res["calibration_bruit"]); st.dataframe(t, width="stretch")
        st.json(res["params_bruit"], expanded=False)
        st.caption("Le bruit de départ laissait le Holter bruité trop proche du Holter (gain, phases identiques). Le bruit calibré rapproche les quatre mesures du patch réel.")
    elif q == "Q3 · Auto-encodeurs":
        t = pd.DataFrame(res["ae"]); st.dataframe(t.round(4), hide_index=True, width="stretch")
        fig = go.Figure([go.Bar(name="SNR avant", x=t["modèle"], y=t["SNR avant (dB)"]), go.Bar(name="SNR après", x=t["modèle"], y=t["SNR après (dB)"])])
        fig.update_layout(barmode="group", height=360, yaxis_title="dB"); st.plotly_chart(fig, width="stretch")
        st.info(f"AE retenu (plus faible MSE de validation Holter à bruit calibré) : **{res['ae_retenu']}**.")
    elif q == "Q4 · Débruitage":
        t = pd.DataFrame(res["debruitage_val"]); ic = t["IC95 % patients"].str.extract(r"\[([\d.]+) ; ([\d.]+)\]").astype(float)
        fig = go.Figure(go.Bar(x=t["F1 macro"], y=t["condition"], orientation="h", marker_color="#2E86AB",
                               error_x=dict(type="data", symmetric=False, array=(ic[1] - t["F1 macro"]).clip(lower=0), arrayminus=(t["F1 macro"] - ic[0]).clip(lower=0))))
        fig.update_layout(height=480, margin=dict(l=10), title="F1 macro (validation) — IC 95 % bootstrap sur les patients"); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch"); st.dataframe(t.drop(columns="F1 par classe"), hide_index=True, width="stretch")
        st.caption("Validation patch : 150 fenêtres, 6 patients → intervalles larges.")
    elif q == "Test final & embarquement":
        F = res["final"]; cib = {"holter": 0.97, "patch_sans_debruitage": 0.85, "patch_avec_debruitage": 0.85}
        st.caption(f"Évaluation sur : **{F['split']}** (une seule fois, modèles figés)")
        c = st.columns(3)
        for col, k in zip(c, cib):
            col.metric(k.replace("_", " "), f"{F[k]['f1_macro']:.3f}", f"{F[k]['f1_macro']-cib[k]:+.3f} vs cible {cib[k]}", delta_color="normal")
        c = st.columns(3)
        for col, (k, nom) in zip(c, [("confusion_holter", "Holter"), ("confusion_patch_sans", "Patch sans débruitage"), ("confusion_patch_avec", "Patch avec débruitage")]):
            col.plotly_chart(heat(np.array(F[k]), NOMS2, nom), width="stretch")
        st.markdown("**Embarquement** (TFLite, 1 cœur de serveur ; cibles ≤ 2 Mo et ≤ 100 ms)"); st.dataframe(pd.DataFrame(res["embarque"]), hide_index=True, width="stretch")
    else:
        st.dataframe(journal[["horodatage", "jalon", "essai", "config", "metriques", "remarque"]], hide_index=True, width="stretch")
        st.caption("Versions : " + " · ".join(f"{k} {v}" for k, v in cfg["versions"].items()))


# ------------------------------------------------------------------ théorie
def section_theorie(cfg, res):
    st.markdown("#### Champ réceptif : $r \\leftarrow r + (k-1)\\,d\\,j$, puis $j \\leftarrow j\\,s$ (départ $r=j=1$)")
    c = st.columns(4)
    prof = c[0].slider("Blocs (conv + max-pool 2)", 1, 7, 4, key="j2_prof"); k = c[1].select_slider("Noyau", [3, 5, 7, 11, 15], 7, key="j2_k")
    dil = c[2].radio("Dilatations", ["aucune", "doublées (1-2-4-8…)"], key="j2_dil"); bpm = c[3].slider("Fréquence (bpm)", 40, 150, 70, key="j2_bpm")
    dl = None if dil == "aucune" else [2 ** i for i in range(prof)]
    couches = couches_cnn(prof, k, dl); r, j = 1, 1; lignes = []
    for (kk, s, d) in couches:
        r += (kk - 1) * d * j; j *= s
        lignes.append({"couche": f"conv {kk}" + (f" (d={d})" if d > 1 else "") if kk != 2 or s == 1 else "pool 2", "r": r, "j": j})
    rr = 60 / bpm * FS; rf = champ_receptif(couches)[0]
    m = st.columns(4); m[0].metric("Champ réceptif", f"{rf} éch."); m[1].metric("en secondes", f"{rf/FS:.2f} s"); m[2].metric("1 intervalle RR", f"{rr:.0f} éch."); m[3].metric("2 intervalles RR", f"{2*rr:.0f} éch.")
    ok1, ok2 = rf >= rr, rf >= 2 * rr
    (st.success if ok2 else st.warning if ok1 else st.error)(
        "Couvre 2 intervalles RR : la FA et la pause du bloc AV sont visibles par les convolutions." if ok2 else
        "Couvre un RR mais pas deux : aucun RR n'est comparé à un autre." if ok1 else "Ne couvre même pas un intervalle RR : seule la forme d'un battement est vue.")
    fig = go.Figure(); fig.add_bar(x=[rf / FS], y=["champ réceptif"], orientation="h", marker_color="#2E86AB")
    fig.add_vline(x=rr / FS, line_dash="dot", annotation_text="1 RR"); fig.add_vline(x=2 * rr / FS, line_dash="dash", annotation_text="2 RR")
    fig.update_xaxes(range=[0, 10], title="secondes (fenêtre de 10 s)"); fig.update_layout(height=170, margin=dict(l=10, r=10, t=20, b=10)); st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(lignes), hide_index=True)
    st.caption("Cas du sujet (4 blocs, noyau 7) : 106 éch. = 0,85 s — un battement, mais pas deux RR. Remèdes : profondeur (5 blocs → 218), dilatations 1-2-4-8 (526 éch. = 4,2 s), agrégation globale. "
               "Mesure par gradient dans le notebook : " + ", ".join(f"{d['architecture']} → {d['RF mesuré']}" for d in res["champ_receptif"][:3]) + "…")


def render():
    if not (ART / "config.json").exists():
        st.error("Artefacts du jalon 2 introuvables (`artifacts/jalon2/`).")
        st.markdown("Exécutez la cellule `export_jalon2.py` à la fin de `jalon2.ipynb`, puis copiez `artifacts/jalon2/` à côté de `app.py`."); return
    cfg, res, mod, journal, ex = charger()
    st.markdown("### Jalon 2 · Apprendre sur le signal brut")
    st.caption("CNN 1D sur signal normalisé par fenêtre · auto-encodeur débruiteur pour passer du Holter au patch · signaux simulés, aucune conclusion médicale.")
    vue = st.segmented_control("Vue", ["🔬 Démonstration", "📊 Résultats & expériences", "🧮 Champ réceptif"], default="🔬 Démonstration", key="j2_vue")
    if vue == "📊 Résultats & expériences": section_resultats(cfg, res, journal)
    elif vue == "🧮 Champ réceptif": section_theorie(cfg, res)
    else: section_demo(cfg, res, mod, ex)
