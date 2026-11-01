import streamlit as st
from tabs import jalon1, jalon2, jalon3, jalon4

st.set_page_config(page_title="CardioPatch - Deep learning avancé", page_icon="🫀", layout="wide")
st.title("🫀 CardioPatch - télésurveillance cardiaque")
st.caption("M2 SDIA · Advanced Artificial Intelligence · Étude de cas 3 — un onglet par jalon")

t1, t2, t3, t4 = st.tabs(["Jalon 1 · Descripteurs → décision", "Jalon 2 · Signal brut", "Jalon 3 · Holter → patch", "Jalon 4 · Séquences"])
with t1:
    jalon1.render()
with t2:
    jalon2.render()
with t3:
    jalon3.render()
with t4:
    jalon4.render()
