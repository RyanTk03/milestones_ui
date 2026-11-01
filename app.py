import streamlit as st
from tabs import jalon1

st.set_page_config(page_title="CardioPatch - Deep learning avancé", page_icon="🫀", layout="wide")
st.title("🫀 CardioPatch - télésurveillance cardiaque")
st.caption("M2 SDIA · Advanced Artificial Intelligence · Étude de cas 3 — un onglet par jalon")

t1, t2, t3, t4 = st.tabs(["Jalon 1 · Descripteurs → décision", "Jalon 2", "Jalon 3", "Jalon 4"])
with t1:
    jalon1.render()
for t, n in ((t2, 2), (t3, 3), (t4, 4)):
    with t:
        st.info(f"Le jalon {n} sera ajouté ici (modèles, prétraitements et affichage dédiés).")
