# CardioPatch – application Streamlit (jalon 1)

1. Ouvrir `jalon1_avec_export.ipynb` (votre notebook + une dernière cellule d'export), l'exécuter en entier
   avec `uc3_holterA.npz` → crée `artifacts/jalon1/` (à copier ici, à côté de `app.py`).
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

Structure : `app.py` (4 onglets, un par jalon) · `tabs/jalon1.py` (onglet) · `common/features.py`
(détection QRS + 32 descripteurs, copie fidèle du notebook) · `export_jalon1.py` (cellule d'export).
Ajouter un jalon = créer `tabs/jalonN.py` avec une fonction `render()` et l'appeler dans son onglet.
