# CardioPatch – interface Streamlit des quatre jalons

M2 SDIA · Advanced Artificial Intelligence · Étude de cas 3 « Télésurveillance cardiaque ».
Cette application déploie les modèles des quatre jalons du notebook correspondant. Elle permet de choisir un jalon (un onglet par jalon),
de fournir un signal ECG, d'appliquer **les mêmes prétraitements que dans le notebook**, puis d'afficher la prédiction avec une interprétation.

> Les signaux sont simulés : aucune conclusion médicale.

---

## 1. Installation et lancement

```bash
pip install -r requirements.txt        # Python ≥ 3.10 ; TensorFlow/Keras 3 requis (modèles au format .keras)
streamlit run app.py                    # à lancer depuis ce dossier
```
ou avec uv:
```bash
uv sync
uv run streamlit run app.py
```

Utilisez, si possible, **les mêmes versions de TensorFlow/Keras que dans vos notebooks** : elles sont indiquées dans l'en-tête de chaque
notebook et recopiées dans `artifacts/jalonN/config.json`.

## 2. Préparer les artefacts (une fois par jalon)

L'application ne ré-entraîne rien : elle charge les modèles et les résultats produits par vos notebooks.

1. Exécutez `jalonN.ipynb` **en entier** (avec les fichiers `.npz`).
2. Ajoutez en dernière cellule le contenu de `export_jalonN.py` et exécutez-la. Elle crée `artifacts/jalonN/`
   (modèles `.keras`, `config.json`, tableaux de résultats, fenêtres d'exemple).
3. Copiez le dossier `artifacts/jalonN/` dans `artifacts/` à côté de `app.py`.

Si un dossier manque, l'onglet concerné affiche les instructions au lieu de s'arrêter ; les autres onglets continuent de fonctionner.

| Jalon | Ce qui est exporté |
|---|---|
| 1 | modèle final, 5 modèles « perte » (CE, CE pondérée, 3 focales), 8 configurations d'ablation, normalisations, seuils A/B, tableaux, ROC |
| 2 | 8 CNN d'ablation + CNN passe-haut + CNN entraîné sur Holter bruité, MLP comparateur, 3 auto-encodeurs, résultats, bruit patch |
| 3 | toutes les stratégies d'adaptation, ré-entraînées sur les 18 patients du patch (réglages du notebook, graine 0), 4 modèles d'ablation de l'augmentation, spectres, courbe d'apprentissage |
| 4 | RNN simple, LSTM, CNN seul, hybrides CNN+GRU / CNN+LSTM, GRU sur battements (avec et sans masquage), historiques de gradients |

## 3. Organisation de l'interface

```
app.py                  4 onglets : un par jalon
tabs/jalonN.py          contenu de l'onglet N
common/                 prétraitements copiés du notebook (features.py, signal2.py, signal3.py, signal4.py)
artifacts/jalonN/       modèles et résultats
```

Chaque onglet propose **trois vues** (sélecteur en haut de l'onglet) :

- **🔬 Démonstration** : choisir une entrée, un modèle, lire la décision et son interprétation.
- **📊 Résultats & expériences** : tous les essais du notebook (tableaux, courbes, test final, journal d'expériences) lus depuis les fichiers exportés.
- **🧮 Vérifications théoriques** : les calculs demandés par les consignes, rendus interactifs.

### Entrées acceptées
- **Exemples** exportés depuis le jeu de test (par rythme réel) : pratique pour la soutenance, avec la bonne réponse connue.
- **Fichier `.csv`, `.txt` ou `.npy`** : une seule fenêtre de **1 250 valeurs** (10 s à 125 Hz, en mV). Toute autre longueur est refusée avec un message.
- Selon l'onglet : un signal modifié (bruit, augmentation) pour tester la robustesse.

## 4. Mode d'emploi par onglet

### Jalon 1 · Descripteurs → décision (MLP sur descripteurs RR)
- **Démonstration** : exemple de test, cas d'échec de la détection QRS (validation) ou fichier ; âge et sexe sont des entrées du modèle.
  Choix parmi 14 modèles (modèle retenu ★, comparaison des pertes, ablation des groupes de descripteurs) et du **seuil d'alerte FA** (A : validation seule ; B : hors pli, retenu ; manuel).
  Sortie : alerte oui/non (règle : classe ≠ sinusal **ou** score FA > seuil), probabilités, QRS détectés et intervalles RR, avertissement si la détection QRS est suspecte,
  texte d'interprétation clinique, poids de chaque groupe de descripteurs dans la prédiction, comparaison aux profils médians des cinq rythmes, comparaison des 14 modèles.
- **Résultats** : Q1 référence, Q2 pertes, Q3 ablation, Q4 seuil FA (curseur donnant sensibilité/spécificité sur validation **et** test), test final face aux cibles, journal.
- **Théorie** : poids de classes w_k = N/(K·N_k) ; calculateur d'entropie croisée et de gradient p − y.

### Jalon 2 · Signal brut (CNN 1D, auto-encodeurs)
- **Démonstration** : Holter, patch, ou Holter + bruit « patch » calibré (graine réglable). On choisit le **classifieur** (CNN retenu ★, ablations, passe-haut, entraîné sur bruit, MLP) et le **débruiteur** (V1, V2 U-Net, V2 avec le bruit de départ).
  Sortie : décision **sans** puis **avec** débruitage côte à côte, signal avant/après auto-encodeur, sensibilité temporelle (occlusion d'1 s), latences et paramètres.
- **Résultats** : CNN vs MLP, ablations vs champ réceptif, calibration du bruit, auto-encodeurs, effet du débruitage (IC par patients), test final et embarquement (TFLite), journal.
- **Théorie** : calculateur de champ réceptif (blocs, noyau, dilatations, fréquence cardiaque) — 4 blocs noyau 7 → 106 éch. (0,85 s) ; dilatations 1-2-4-8 → 526 éch. (4,2 s).

### Jalon 3 · Holter → patch (décalage de domaine, transfert, multitâche)
- **Démonstration** : patch, Holter, **Holter + augmentation « patch »** (choix des transformations gain/dérive/bruit/secteur), fenêtres inédites sans étiquette, fichier.
  Modèle au choix (stratégie retenue ★ : extraction, ajustement fin partiel/complet, augmentation, multitâche λ…) comparé à « Holter tel quel ».
  Sortie : alerte, probabilités, fréquence cardiaque prédite si le modèle est multitâche (avec l'écart à l'annotation), sensibilité temporelle, **diagnostic de décalage** (la fenêtre ressemble-t-elle à un Holter ou à un patch ?).
- **Résultats** : Q3 caractérisation du décalage (tableau, spectres, complexe moyen), Q1 stratégies en validation croisée, Q2 augmentation, Q4 effet de λ, choix/test/courbe d'apprentissage (« combien de patients annoter »), journal.
- **Théorie** : invariance de z = (x − µ)/σ au gain et au décalage (curseurs a, b, bruit ; la polarité inversée casse l'invariance) ; paramètres entraînables (325 en extraction, 51 365 en ajustement fin complet).

### Jalon 4 · Le rythme est une séquence (RNN, LSTM, GRU)
- **Démonstration** : Holter de test, Holter + bruit blanc (test de robustesse), fichier. Les 6 modèles sont comparés côte à côte avec leur latence mesurée ;
  le modèle retenu (hybride CNN+GRU) donne la décision et le score FA face au seuil calibré en validation. Un modèle « à analyser » au choix :
  pour les modèles sur signal brut, courbe de l'influence de chaque pas de temps sur la décision (la **disparition du gradient** du RNN simple s'y voit) ;
  pour le GRU sur battements, suite des RR (avec remplissage masqué) et importance de chaque battement.
- **Résultats** : RNN brut et normes de gradient, hybride vs CNN, battements (ablation du masquage, stabilité sur 3 graines), coût et embarqué, sélection et test, journal.
- **Théorie** : λ^T interactif (disparition/explosion du gradient) ; paramètres de SimpleRNN/LSTM/GRU comparés à Keras (LSTM 32 unités, 1 variable : 4 352).

## 5. Lire les résultats avec précaution

- **Données simulées**, signaux d'un seul type de patients : les chiffres décrivent les jeux fournis, pas la clinique.
- **Les chiffres affichés sont ceux de vos notebooks.** L'application n'invente aucune métrique ; elle relit les fichiers exportés.
- **Test utilisé une fois** : la vue « Résultats » affiche le test pour le modèle retenu uniquement. Le curseur de seuil du jalon 1 montre aussi sensibilité/spécificité sur le test en lecture seule : ne réglez pas le seuil dessus.
- **Jalon 3** : les stratégies comparables dans la démonstration sont ré-entraînées à l'export sur les 18 patients annotés (le notebook ne garde pas les modèles de chaque pli de validation croisée). Les tableaux de la vue « Résultats » restent ceux de la validation croisée du notebook.
- **Jalon 4** : l'essai « GRU sans masquage » est ré-entraîné à l'export (même graine) ; ses chiffres peuvent différer légèrement de ceux du notebook si le GPU n'est pas déterministe. Les latences viennent du matériel du notebook, pas d'un système embarqué.
- **Interprétations** (occlusion, gradient, diagnostic de décalage) : explications indicatives sur une fenêtre isolée, pas des preuves.

## 6. Dépannage

| Symptôme | Cause probable / solution |
|---|---|
| « Artefacts du jalon N introuvables » | exécuter la cellule d'export et copier `artifacts/jalonN/` |
| Erreur au chargement d'un modèle | version de TensorFlow/Keras différente de celle du notebook ; installer les versions de `config.json` |
| « 1 250 valeurs attendues » | le fichier doit contenir **une** fenêtre de 10 s à 125 Hz |
| Avertissement « suite de battements différente » (jalon 4) | le détecteur de QRS du notebook n'est pas la version de remplacement : adapter `common/signal4.py` |
| Premier affichage lent | chargement des modèles (mis en cache ensuite) ; le RNN simple est lent par nature |
