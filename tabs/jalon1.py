"""Onglet Jalon 1 : des descripteurs à la décision (MLP sur descripteurs RR + seuil d'alerte FA)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from common.features import (FS, N_POINTS, NOMS, NOMS_LONGS, GROUPES, detecter_qrs, fc_depuis_pics,
                             toutes_les_features, vecteur)

ART = Path(__file__).resolve().parent.parent / "artifacts" / "jalon1"
COULEURS = ["#2E86AB", "#D1495B", "#EDAE49", "#66A182", "#8D6A9F"]   # une couleur par rythme

ICON_ALERTE, ICON_OK, ICON_WARN = ":material/notification_important:", ":material/check_circle:", ":material/warning:"
VUES = {"demo": ":material/science: Démonstration",
        "res": ":material/bar_chart: Résultats & expériences",
        "theo": ":material/function: Vérifications théoriques"}


# --------------------------------------------------------------------------- chargement
@st.cache_resource(show_spinner="Chargement des modèles du jalon 1…")
def charger():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras
    cfg = json.load(open(ART / "config.json"))
    modeles = {k: keras.models.load_model(ART / v["fichier"], compile=False) for k, v in cfg["modeles"].items()}
    lire = lambda n, **kw: pd.read_csv(ART / n, **kw)
    tab = dict(pertes=lire("table_pertes.csv", index_col=0), ablation=lire("table_ablation.csv", index_col=0),
               es=lire("table_early_stopping.csv", index_col=0), seuil_A=lire("table_seuil_A.csv"),
               seuil_B=lire("table_seuil_B.csv"), journal=lire("journal.csv"), seuils=lire("table_seuils.csv", index_col=0),
               cibles=lire("table_cibles.csv"), profils=lire("profils_classes.csv"))
    res = dict(np.load(ART / "resultats.npz"))
    ex = dict(np.load(ART / "exemples.npz", allow_pickle=True))
    return cfg, modeles, tab, res, ex


@st.cache_data(show_spinner=False)
def calculer_features(x_bytes, age, sexe, k_mad):
    """Mêmes prétraitements que le notebook : détection QRS + 32 descripteurs."""
    x = np.frombuffer(x_bytes, dtype="float32")
    return toutes_les_features(x, age, sexe, k_mad)


def predire(cfg, modeles, cle, feats):
    """Normalisation (moyenne/écart-type du train) puis probabilités softmax du MLP."""
    m = cfg["modeles"][cle]
    x = (vecteur(feats, m["groupes"]) - np.array(m["mean"], "float32")) / np.array(m["scale"], "float32")
    return modeles[cle](x[None].astype("float32"), training=False).numpy()[0], x


def occlusion_groupes(cfg, modeles, cle, x_norm, classe):
    """Contribution de chaque groupe : baisse de p(classe) quand ses descripteurs sont remplacés par la moyenne du train (0 après normalisation)."""
    m = cfg["modeles"][cle]; base = modeles[cle](x_norm[None], training=False).numpy()[0][classe]
    dim = [11, 4, 8, 9]          # taille de chaque groupe (base, amplitude/largeur, bandes, irrégularité)
    out, pos = {}, 0
    for g in m["groupes"]:
        z = x_norm.copy(); z[pos:pos + dim[g]] = 0.0; pos += dim[g]
        out[GROUPES[g]] = float(base - modeles[cle](z[None], training=False).numpy()[0][classe])
    return out


# --------------------------------------------------------------------------- graphiques
def graphique_signal(x, pics, fc_annotee=None):
    t = np.arange(len(x)) / FS
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[.68, .32], vertical_spacing=.14,
                        subplot_titles=("Signal ECG (10 s) et QRS détectés", "Intervalles RR (s)"))
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", line=dict(width=1, color="#2E86AB"), name="ECG"), 1, 1)
    snap = [int(max(0, p - 12) + np.argmax(x[max(0, p - 12):p + 13])) for p in pics]   # affichage seulement : marqueur calé sur le sommet de l'onde R
    fig.add_trace(go.Scatter(x=t[snap], y=x[snap], mode="markers", marker=dict(symbol="triangle-down", size=10, color="#D1495B"),
                             name="QRS détectés"), 1, 1)
    if len(pics) > 1:
        rr = np.diff(pics) / FS
        fig.add_trace(go.Bar(x=t[pics][1:], y=rr, marker_color="#EDAE49", name="RR"), 2, 1)
        fig.add_hline(y=float(np.median(rr)), line_dash="dot", line_color="gray", row=2, col=1,
                      annotation_text="médiane", annotation_position="top left")
    fig.update_xaxes(title_text="temps (s)", row=2, col=1); fig.update_yaxes(title_text="mV", row=1, col=1)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


def graphique_probas(p, seuil, p_fa):
    fig = go.Figure(go.Bar(x=p, y=NOMS_LONGS, orientation="h", marker_color=COULEURS,
                           text=[f"{v:.1%}" for v in p], textposition="outside"))
    fig.add_vline(x=seuil, line_dash="dash", line_color="#D1495B", annotation_text=f"seuil FA {seuil:.3f}", annotation_position="top")
    fig.update_xaxes(range=[0, 1.12], title="probabilité softmax"); fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def heatmap_cm(cm, titre):
    cmn = cm / cm.sum(1, keepdims=True)
    fig = go.Figure(go.Heatmap(z=cmn, x=NOMS, y=NOMS, colorscale="Blues", zmin=0, zmax=1, showscale=False,
                               text=[[f"{cm[i, j]}<br>({cmn[i, j]:.0%})" for j in range(5)] for i in range(5)],
                               texttemplate="%{text}", hovertemplate="réel %{y} → prédit %{x}<extra></extra>"))
    fig.update_xaxes(title="prédit", side="bottom"); fig.update_yaxes(title="réel", autorange="reversed")
    fig.update_layout(title=titre, height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# --------------------------------------------------------------------------- interprétation
CLES = [("fc (bpm)", None), ("cv", "cv"), ("rmssd", "rmssd"), ("pnn", "pnn"), ("sd1", "sd1"), ("samp", "samp"),
        ("court", "court"), ("long", "long"), ("ratio_pause", "ratio_pause")]
LIBELLES = {"cv": "variabilité RR (CV)", "rmssd": "RMSSD (s)", "pnn": "pNN50 (part de ΔRR > 50 ms)", "sd1": "SD1 (Poincaré)",
            "samp": "entropie d'échantillon des RR", "court": "part de RR courts (< 0,8 médiane)",
            "long": "part de RR longs (> 1,3 médiane)", "ratio_pause": "ratio pause (RR max / médiane)", "fc (bpm)": "fréquence cardiaque (bpm)"}


def profil(tab, desc, classe, col="mediane"):
    p = tab["profils"]
    return float(p[(p.descripteur == desc) & (p.classe == NOMS[classe])][col].iloc[0])


def table_cles(tab, feats, fc_detectee, cfg, cle):
    utilises = set(cfg["modeles"][cle]["features"])
    lignes = []
    for nom, d in CLES:
        if d is None:
            val = fc_detectee
            meds = [60.0 / profil(tab, "mrr", k) for k in range(5)]
        else:
            g = next(g for g in GROUPES if d in feats[g]); val = float(feats[g][d])
            meds = [profil(tab, d, k) for k in range(5)]
        lignes.append({"descripteur": LIBELLES[nom], "cette fenêtre": round(val, 3),
                       **{f"méd. {NOMS[k]}": round(meds[k], 3) for k in range(5)},
                       "utilisé par le modèle": "oui" if (d is None and "mrr" in utilises) or d in utilises else "non"})
    return pd.DataFrame(lignes)


def lecture_clinique(k, feats, fc_detectee):
    b, irr = feats["base"], feats["irrégularité"]
    if k == 0:
        return (f"Rythme **régulier** à ~{fc_detectee:.0f} bpm : variabilité RR faible (CV = {b['cv']:.3f}, pNN50 = {irr['pnn']:.2f}) "
                "et aucune pause marquée. Profil compatible avec un rythme sinusal (éventuellement une extrasystole isolée).")
    if k == 1:
        return (f"Rythme **irrégulièrement irrégulier** : CV = {b['cv']:.3f}, pNN50 = {irr['pnn']:.2f}, SD1 = {irr['sd1']:.3f} s, "
                "bien au-dessus d'un sinusal. C'est la signature de la fibrillation atriale (absence d'organisation des RR) — "
                "alerte cardiologue si le score FA dépasse le seuil.")
    if k == 2:
        return (f"**Intervalles courts** ({irr['court']:.0%} des RR < 0,8 × médiane) suivis d'intervalles **longs** ({irr['long']:.0%} > 1,3 × médiane) : "
                "motif de battements prématurés suivis d'une pause compensatrice, typique d'extrasystoles ventriculaires fréquentes.")
    if k == 3:
        return f"**Fréquence élevée** (~{fc_detectee:.0f} bpm, RR moyen {b['mrr']:.2f} s) avec un rythme régulier (CV = {b['cv']:.3f}) : tachycardie sinusale."
    return (f"**Pauses** dans la séquence des RR : ratio RR max / médiane = {irr['ratio_pause']:.2f} (≈ 1 en rythme régulier). "
            "Des QRS « manquants » à intervalles réguliers évoquent un bloc auriculo-ventriculaire du 2e degré.")


# --------------------------------------------------------------------------- sections
# @st.fragment : un changement de widget ne relance que ce bloc -> la page ne remonte plus en haut.
@st.fragment
def section_demo(cfg, modeles, tab, res, ex):
    gauche, droite = st.columns([1, 2.2], gap="large")
    with gauche:
        st.subheader("1 · Entrée")
        source = st.radio("Source du signal", ["Exemple (jeu de test)", "Échec de détection QRS (validation)", "Fichier .csv / .npy"], key="j1_source")
        x = None; fc_annotee = 0.0; age, sexe, vrai = 60, 0, None
        if source.startswith("Exemple"):
            classe_f = st.selectbox("Rythme réel de l'exemple", range(5), format_func=lambda k: NOMS_LONGS[k], key="j1_cf")
            idx = np.where((ex["y"] == classe_f) & (~ex["echec_qrs"]))[0]
            i = idx[st.slider("Fenêtre n°", 1, len(idx), 1, key="j1_i") - 1]
        elif source.startswith("Échec"):
            idx = np.where(ex["echec_qrs"])[0]
            if len(idx) == 0:
                st.info("Aucun échec de détection exporté."); return
            i = idx[st.slider("Cas n°", 1, len(idx), 1, key="j1_e") - 1]
        else:
            f = st.file_uploader("Une fenêtre de 1 250 points (10 s à 125 Hz, en mV)", type=["csv", "npy", "txt"], key="j1_up")
            if f is None:
                st.info("Déposez un fichier : une seule colonne/ligne de 1 250 valeurs.")
                return
            try:
                arr = np.load(f).ravel() if f.name.endswith(".npy") else np.loadtxt(f, delimiter=",").ravel()
            except Exception as e:
                st.error(f"Lecture impossible : {e}"); return
            if arr.size != N_POINTS:
                st.error(f"{arr.size} valeurs reçues, {N_POINTS} attendues (10 s × 125 Hz)."); return
            x = arr.astype("float32"); i = None
        if i is not None:
            x = ex["X"][i].astype("float32"); fc_annotee = float(ex["fc"][i]); age = int(ex["age"][i]); sexe = int(ex["sexe"][i]); vrai = int(ex["y"][i])
            st.caption(f"Origine : {ex['source'][i]}")
        age = st.number_input("Âge (descripteur du modèle)", 1, 110, int(age), key=f"j1_age_{source}_{i}")
        sexe = 1 if st.radio("Sexe", ["F", "M"], index=int(sexe), horizontal=True, key=f"j1_sex_{source}_{i}") == "M" else 0

        st.subheader("2 · Modèle et seuil")
        cles = list(cfg["modeles"])
        cle = st.selectbox("Modèle", cles, format_func=lambda k: ("★ " if k == "final" else "") + cfg["modeles"][k]["titre"] +
                           {"final": "", "perte": "  [comparaison des pertes]", "ablation": "  [ablation]"}[cfg["modeles"][k]["famille"]], key="j1_modele")
        choix = st.radio("Seuil d'alerte FA", [f"B – hors pli (retenu) : {cfg['seuil_B']:.3f}", f"A – validation seule : {cfg['seuil_A']:.3f}", "Manuel"], key="j1_seuil")
        seuil = cfg["seuil_B"] if choix.startswith("B") else cfg["seuil_A"] if choix.startswith("A") else st.slider("Seuil", 0.0, 1.0, float(cfg["seuil_B"]), 0.005, key="j1_man")
        if cle != "final":
            st.caption(":material/warning: Les seuils A/B ont été calibrés pour le modèle retenu uniquement ; sur un modèle auxiliaire ils ne garantissent pas 95 % de spécificité.")

    with droite:
        feats = calculer_features(x.tobytes(), float(age), float(sexe), cfg["k_mad"])
        pics, _ = detecter_qrs(x.astype("float64"), cfg["k_mad"]); fc_det = fc_depuis_pics(pics)
        p, xn = predire(cfg, modeles, cle, feats); k = int(p.argmax()); signalee = (k != 0) or (p[1] > seuil)

        st.subheader("3 · Décision")
        if signalee:
            motif = "classe prédite ≠ sinusal" if k != 0 else f"score FA {p[1]:.3f} > seuil {seuil:.3f}"
            st.error(f"**Alerte transmise au cardiologue** — {NOMS_LONGS[k] if k else 'rythme sinusal, mais score FA au-dessus du seuil'} ({motif})", icon=ICON_ALERTE)
        else:
            st.success("**Pas d'alerte** — rythme sinusal, score FA sous le seuil", icon=ICON_OK)
        c = st.columns(4)
        c[0].metric("Rythme prédit", NOMS[k], f"{p[k]:.1%}", delta_color="off", delta_arrow="off")
        c[1].metric("Score FA", f"{p[1]:.3f}", f"seuil {seuil:.3f}", delta_color="inverse" if p[1] > seuil else "off", delta_arrow="off")
        c[2].metric("FC détectée", f"{fc_det:.0f} bpm", f"annotée {fc_annotee:.0f}" if fc_annotee else None, delta_color="off", delta_arrow="off")
        c[3].metric("QRS détectés", len(pics))
        if vrai is not None:
            (st.success if vrai == k else st.warning)(f"Rythme réel (annotation) : **{NOMS_LONGS[vrai]}** — " + ("prédiction correcte." if vrai == k else "prédiction différente."),
                                                       icon=ICON_OK if vrai == k else ICON_WARN)
        if fc_annotee and abs(len(pics) * 6 - fc_annotee) / fc_annotee > 0.25:
            st.warning(f"**Détection QRS suspecte** : {len(pics)} pics ×6 = {len(pics)*6} bpm contre {fc_annotee:.0f} bpm annotés. "
                       "Sur-comptage typique des BAV2 (ondes P non conduites prises pour des QRS) : les descripteurs RR sont faussés, prudence sur la prédiction.", icon=ICON_WARN)
        st.plotly_chart(graphique_signal(x, pics), width="stretch")
        a, b = st.columns([1.1, 1])
        with a:
            st.plotly_chart(graphique_probas(p, seuil, p[1]), width="stretch")
        with b:
            st.markdown("**Interprétation**"); st.markdown(lecture_clinique(k, feats, fc_det))
            occ = occlusion_groupes(cfg, modeles, cle, xn, k)
            fig = go.Figure(go.Bar(x=list(occ.values()), y=list(occ.keys()), orientation="h", marker_color=["#2E86AB" if v >= 0 else "#D1495B" for v in occ.values()]))
            fig.update_layout(title=f"Poids de chaque groupe dans p({NOMS[k]})", height=190, margin=dict(l=10, r=10, t=40, b=10))
            fig.update_xaxes(title="baisse de probabilité si le groupe est neutralisé"); st.plotly_chart(fig, width="stretch")
        with st.expander("Descripteurs clés de cette fenêtre vs profils des 5 rythmes (médianes du train)", expanded=False):
            st.dataframe(table_cles(tab, feats, fc_det, cfg, cle), hide_index=True, width="stretch")
        with st.expander("Comparer les 14 modèles (modèle final + pertes + ablation) sur cette fenêtre", expanded=False):
            lignes = []
            for kk, m in cfg["modeles"].items():
                pp, _ = predire(cfg, modeles, kk, feats)
                lignes.append({"modèle": m["titre"], "famille": m["famille"], "prédiction": NOMS[int(pp.argmax())], "confiance": round(float(pp.max()), 3),
                               "score FA": round(float(pp[1]), 3), **{f"p({n})": round(float(v), 3) for n, v in zip(NOMS, pp)}})
            st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch")
        with st.expander("Descripteurs calculés (32)"):
            st.dataframe(pd.DataFrame({g: pd.Series(v) for g, v in feats.items()}).T.round(4), width="stretch")


@st.fragment
def section_resultats(cfg, tab, res):
    st.markdown("Tous les chiffres viennent du notebook (validation, 5 graines) ; le test a été utilisé **une seule fois** pour le modèle retenu.")
    q = st.segmented_control("Étape", ["Q1 · Référence", "Q2 · Pertes", "Q3 · Ablation", "Q4 · Seuil FA", "Test final", "Journal"], default="Q2 · Pertes", key="j1_q") or "Q2 · Pertes"
    if q == "Q1 · Référence":
        a, b = st.columns(2)
        a.plotly_chart(heatmap_cm(res["cm_ref"], "MLP de référence (11 descripteurs, CE pondérée) — validation"), width="stretch")
        fig = px.bar(x=NOMS, y=res["rappels_ref"], color=NOMS, color_discrete_sequence=COULEURS, labels=dict(x="", y="rappel"), title="Rappel par classe")
        fig.update_yaxes(range=[0, 1.05]).update_layout(showlegend=False, height=380); b.plotly_chart(fig, width="stretch")
        st.caption("Échecs de détection : seule la BAV2 pose problème (sur-comptage, ~1–2 % des fenêtres). Voir l'option « Échec de détection QRS » dans la démo.")
    elif q == "Q2 · Pertes":
        t = tab["pertes"]
        fig = go.Figure()
        for n in NOMS:
            fig.add_bar(name=n, x=t.index, y=t["R_" + n])
        fig.update_layout(barmode="group", title="Rappel par classe selon la perte (validation, moyenne de 5 graines)", height=400); fig.update_yaxes(range=[.85, 1.0])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(t[["F1", "ecart_type", "R_arythmies", "R_tachy.", "R_sinusal", "fausses_alertes", "sens_FA"]].round(3), width="stretch")
        st.info(f"F1 quasi indiscernable entre pertes ; l'effet est sur les classes rares : la pondération monte le rappel des arythmies mais augmente les fausses alertes. **Perte retenue** : {cfg['nom_perte']} (règle fixée avant les chiffres).")
    elif q == "Q3 · Ablation":
        t = tab["ablation"].sort_values("F1")
        fig = go.Figure(go.Bar(x=t["F1"], y=t.index, orientation="h", error_x=dict(array=t["ecart_type"]),
                               marker_color=["gold" if i == cfg["config"] else "#2E86AB" for i in t.index]))
        fig.add_vline(x=0.97, line_dash="dash", line_color="red", annotation_text="cible 0,97")
        fig.update_xaxes(range=[.95, .99], title="F1 macro (validation, 5 graines)"); fig.update_layout(title="Ablation par groupe de descripteurs (or : retenue)", height=400)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(tab["ablation"][["n_desc", "F1", "ecart_type", "R_sinusal", "R_arythmies", "sens_FA", "gain_F1_vs_base"]].round(3), width="stretch")
        st.info("Le groupe **irrégularité** est le seul à l'effet clairement robuste (cohérent avec la FA : RR irréguliers). Les deux autres groupes restent dans le bruit entre graines.")
    elif q == "Q4 · Seuil FA":
        sv = st.slider("Seuil d'alerte FA", 0.0, 1.0, float(cfg["seuil_B"]), 0.005, key="j1_roc")
        def sens_spec(sc, y): return float(np.mean(sc[y == 1] > sv)), float(np.mean(sc[y != 1] <= sv))
        sv_, sp_v = sens_spec(res["fa_val"], res["y_val"]); st_, sp_t = sens_spec(res["fa_test"], res["y_test"])
        c = st.columns(4); c[0].metric("Sensibilité (val.)", f"{sv_:.3f}"); c[1].metric("Spécificité (val.)", f"{sp_v:.3f}")
        c[2].metric("Sensibilité (test)", f"{st_:.3f}"); c[3].metric("Spécificité (test)", f"{sp_t:.3f}")
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=res["roc_val"][0], y=res["roc_val"][1], name=f"validation (AUC {res['auc'][0]:.3f})")
        fig.add_scatter(x=res["roc_test"][0], y=res["roc_test"][1], name=f"test (AUC {res['auc'][2]:.3f})")
        fig.add_scatter(x=[1 - sp_v], y=[sv_], mode="markers", marker=dict(size=12, color="red", symbol="x"), name="seuil choisi (val.)")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False)
        fig.update_layout(title="ROC fibrillation atriale", xaxis_title="1 − spécificité", yaxis_title="sensibilité", height=380); a.plotly_chart(fig, width="stretch")
        fig = go.Figure()
        fig.add_histogram(x=res["fa_val"][res["y_val"] != 1], name="non-FA", xbins=dict(start=0, end=1, size=.04), marker_color="#2E86AB")
        fig.add_histogram(x=res["fa_val"][res["y_val"] == 1], name="FA", xbins=dict(start=0, end=1, size=.04), marker_color="#D1495B")
        fig.add_vline(x=sv, line_dash="dash"); fig.update_yaxes(type="log", title="effectif (log)")
        fig.update_layout(barmode="overlay", title="Scores FA en validation", xaxis_title="score FA", height=380); fig.update_traces(opacity=.7); b.plotly_chart(fig, width="stretch")
        st.dataframe(tab["seuils"].round(3), width="stretch")
        s1, s2 = st.columns(2); s1.markdown("**Stabilité seuil A (5 graines)**"); s1.dataframe(tab["seuil_A"], hide_index=True); s2.markdown("**Stabilité seuil B (3 graines)**"); s2.dataframe(tab["seuil_B"], hide_index=True)
        st.caption("Une fausse alerte coûte une relecture ; une FA manquée expose le patient : le seuil est un choix de coût clinique, pas de statistique.")
    elif q == "Test final":
        c = tab["cibles"]; cols = st.columns(len(c))
        for col, (_, r) in zip(cols, c.iterrows()):
            col.metric(r["Mesure"].split(" (")[0], f"{r['Test']:.3f}", f"{r['Écart']:+.3f} vs cible {r['Cible finale']:.2f}", delta_color="normal")
        a, b = st.columns(2)
        a.plotly_chart(heatmap_cm(res["cm_test"], "Modèle retenu — test (utilisé une seule fois)"), width="stretch")
        b.dataframe(c.round(3), hide_index=True, width="stretch")
        b.markdown(f"Sinus non signalés : règle retenue **{res['sinus'][0]:.3f}** · argmax seul {res['sinus'][1]:.3f} · score FA seul {res['sinus'][2]:.3f}")
        b.warning("La cible « sinus non signalés ≥ 0,97 » n'est pas atteinte : l'écart vient du modèle/descripteurs (fenêtres sinusales classées FA ou BAV2), pas du calibrage du seuil.", icon=ICON_WARN)
    else:
        st.dataframe(tab["journal"], hide_index=True, width="stretch")
        st.markdown("**Comparaison arrêt précoce**"); st.dataframe(tab["es"].round(3))
        st.caption("Versions : " + " · ".join(f"{k} {v}" for k, v in cfg["versions"].items()))


@st.fragment
def section_theorie(cfg):
    st.markdown("#### Poids de classes  $w_k = N / (K\\,N_k)$")
    Nk = np.array(cfg["effectifs_train"], float); N, K = Nk.sum(), 5
    st.dataframe(pd.DataFrame({"classe": NOMS_LONGS, "N_k": Nk.astype(int), "calcul": [f"{int(N)} / (5 × {int(n)})" for n in Nk], "w_k": (N / (K * Nk)).round(3)}), hide_index=True)
    st.markdown("#### Entropie croisée et gradient par rapport aux logits")
    st.caption("Pour un softmax : $\\partial L/\\partial z_k = p_k - y_k$. Modifiez les probabilités (la somme doit valoir 1).")
    cols = st.columns(5); defaut = [.50, .30, .10, .05, .05]
    p = np.array([c.number_input(NOMS[i], 0.0, 1.0, defaut[i], 0.01, key=f"j1_p{i}") for i, c in enumerate(cols)])
    vraie = st.selectbox("Vraie classe", range(5), index=1, format_func=lambda k: NOMS_LONGS[k], key="j1_vraie")
    if abs(p.sum() - 1) > 1e-6:
        st.error(f"Somme des probabilités = {p.sum():.3f} ≠ 1.", icon=ICON_WARN); return
    y = np.eye(5)[vraie]
    st.latex(rf"L = -\ln p_{{{vraie}}} = {-np.log(max(p[vraie], 1e-12)):.4f}")
    st.dataframe(pd.DataFrame([(p - y).round(3)], columns=NOMS, index=["p − y"]))
    st.caption("Le logit de la vraie classe monte (gradient négatif), les autres baissent ; la somme est nulle.")


def render():
    if not (ART / "config.json").exists():
        st.error("Artefacts du jalon 1 introuvables (`artifacts/jalon1/`).", icon=ICON_WARN)
        st.markdown("Exécutez la cellule d'export `export_jalon1.py` à la fin de `jalon1.ipynb`, puis copiez le dossier `artifacts/jalon1/` à côté de `app.py`.")
        return
    cfg, modeles, tab, res, ex = charger()
    st.caption("MLP sur descripteurs d'intervalles RR · perte adaptée aux classes déséquilibrées · seuil d'alerte FA choisi selon le coût clinique. "
               "Signaux simulés : aucune conclusion médicale.")
    vue = st.segmented_control("Vue", list(VUES), format_func=VUES.get, default="demo", key="j1_vue_v2") or "demo"
    if vue == "res":
        section_resultats(cfg, tab, res)
    elif vue == "theo":
        section_theorie(cfg)
    else:
        section_demo(cfg, modeles, tab, res, ex)