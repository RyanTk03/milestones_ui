# UC3 — CardioPatch : télésurveillance cardiaque

**M2 SDIA – Advanced Artificial Intelligence – ENSET – 2026-2027**
**Groupe :** `10`
**Membres :** AARROUB Wijdane · AIT EL MAHJOUB Abdessamad · TOKO Rayane
**Cas :** Étude de cas 3 (UC3) — classification de l'ECG par fenêtres de 10 s (1 250 points à 125 Hz) en 5 rythmes : sinusal, fibrillation atriale (FA), extrasystoles ventriculaires (ESV), tachycardie sinusale, bloc auriculo-ventriculaire du 2ᵉ degré (BAV2).

Ce dépôt contient :

- les **4 jalons** du cas, sous forme de notebooks **autonomes** (chacun recharge les données et redéfinit ses propres outils : aucun ne dépend de l'état d'un autre notebook) ;
- une **interface Streamlit** qui déploie les modèles des quatre jalons (voir la [section 8](#8-interface-streamlit--résumé)).

| Jalon | Notebook | Thème | Appareil |
|---|---|---|---|
| 1 | `Groupe10_UC3_Jalon1.ipynb` | Des descripteurs à la décision (MLP, pertes, seuil FA) | Holter A |
| 2 | `Groupe10_UC3_Jalon2.ipynb` | Signal brut : CNN 1D et auto-encodeurs débruiteurs | Holter A + Patch B |
| 3 | `Groupe10_UC3_Jalon3.ipynb` | Du Holter au patch : transfert, augmentation, multitâche | Holter A + Patch B |
| 4 | `Groupe10_UC3_Jalon4.ipynb` | Le rythme est une séquence : RNN, LSTM, GRU | Holter A |

---

## 1. Structure du dossier

```text
UC3-CardioPatch-G10/
├── README.md
│
├── data/                                # données fournies par l'enseignant
│   ├── uc3_holterA.npz
│   ├── uc3_patchB.npz
│   └── uc3_inedits.npz
│
├── notebooks/                           # 4 notebooks EXÉCUTÉS
│   ├── Groupe10_UC3_Jalon1.ipynb
│   ├── Groupe10_UC3_Jalon2.ipynb
│   ├── Groupe10_UC3_Jalon3.ipynb
│   └── Groupe10_UC3_Jalon4.ipynb
│
├── syntheses/                           # 4 synthèses de décisions
│   ├── Synthese_Jalon1.pdf
│   ├── Synthese_Jalon2.pdf
│   ├── Synthese_Jalon3.pdf
│   └── Synthese_Jalon4.pdf
│
├── presentation/                        # présentation modifiable + PDF
│   ├── Groupe10_UC3_Presentation.pptx
│   └── Groupe10_UC3_Presentation.pdf
│
├── journaux/                            # journal d'expériences
│   ├── journal_jalon1.csv
│   ├── journal_jalon2.csv
│   ├── resultats_jalon2.json
│   ├── journal_jalon3.csv
│   └── journal_experiences_jalon4.csv
│
├── modeles/                             # modèles sauvegardés
│   ├── jalon2/                          # CNN, AE V1, AE V2, pipeline TFLite
│   ├── jalon3/                          # CNN Holter, multitâche λ=0.5, FT complet
│   └── jalon4/                          # RNN simple, LSTM, hybrides, GRU battements
│
└── interface/                           # application Streamlit (voir section 8)
    ├── app.py                           # 4 onglets : un par jalon
    ├── requirements.txt
    ├── tabs/                            # tabs/jalonN.py : contenu de l'onglet N
    ├── common/                          # prétraitements copiés des notebooks
    │   ├── features.py
    │   ├── signal2.py
    │   ├── signal3.py
    │   └── signal4.py
    └── artifacts/                       # modèles et résultats exportés
        ├── jalon1/
        ├── jalon2/
        ├── jalon3/
        └── jalon4/
```

> **Attention.** Les jalons 2 et 4 écrivent tous deux dans `modeles/`. Les noms de fichiers diffèrent, mais pour éviter tout mélange, exécuter chaque jalon dans son propre dossier de travail ou vider `modeles/` entre les deux.

## 2. Dépendances

Environnement de référence (versions exactes) :

| Paquet | Version |
|---|---|
| Python | 3.13.15 |
| NumPy | 2.1.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.6.1 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| pandas | 2.2.3 |
| Streamlit | *version indiquée dans `interface/requirements.txt` (interface uniquement)* |
| matplotlib | *version installée avec les paquets ci-dessus (non figée par les notebooks)* |
| Jupyter (notebook / ipykernel, IPython) | nécessaire pour exécuter les notebooks (`IPython.display` est utilisé) |

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate

# Installation avec les versions de référence
pip install numpy==2.1.3 scipy==1.16.3 pandas==2.2.3 scikit-learn==1.6.1 \
            tensorflow==2.21.0 keras==3.13.2 matplotlib jupyter

# Pour l'interface Streamlit
pip install streamlit

# Version non figée (équivalente à la commande minimale)
# pip install numpy scipy pandas matplotlib scikit-learn tensorflow
```

Sur Google Colab, TensorFlow, NumPy, pandas, scikit-learn et matplotlib sont déjà installés ; vérifier les versions affichées dans la première cellule de chaque notebook.

## 4. Données

Les notebooks lisent des fichiers `.npz` de données fournis par l'enseignant dans le cadre du TP.

| Fichier | Contenu | Utilisé par |
|---|---|---|
| `uc3_holterA.npz` | Holter (appareil A) : 280 / 60 / 60 patients en train / val / test, soit 8 400 / 1 800 / 1 800 fenêtres | Jalons 1, 2, 3, 4 |
| `uc3_patchB.npz` | Patch (appareil B) : 12 / 6 / 42 patients, soit 300 / 150 / 1 050 fenêtres | Jalons 2, 3 |
| `uc3_inedits.npz` | 150 fenêtres du patch **sans étiquette** (diagnostic uniquement) | Jalon 3 |

**Clés attendues** (pour chaque ensemble `s` ∈ {`train`, `val`, `test`}) :

| Clé | Contenu |
|---|---|
| `X_<s>` | signaux, forme `(fenêtres, 1250)`, en mV |
| `classe_<s>` | classe : 0 sinusal, 1 FA, 2 ESV, 3 tachycardie, 4 BAV2 *(le jalon 1 accepte aussi `y_<s>`)* |
| `fc_<s>` | fréquence cardiaque annotée (bpm) |
| `patient_<s>` | identifiant du patient (le découpage est **par patient** : aucun patient dans deux ensembles, vérifié par `assert` dans chaque notebook) |
| `age_<s>`, `sexe_<s>` | utilisés uniquement par le jalon 1 (sur `uc3_holterA.npz`) |

`uc3_inedits.npz` contient seulement `X` et `fc`.

**Où placer les fichiers.** Les notebooks cherchent `uc3_holterA.npz` (puis `uc3_patchB.npz` et `uc3_inedits.npz` dans le même dossier) dans cet ordre :

1. le dossier indiqué par la variable d'environnement `UC3_DATA` (jalons 2 et 3) ;
2. `./` (à côté du notebook) et `data/` ;
3. `/content/` et `/content/data/` (Colab), ainsi que `/content/drive/MyDrive/tp-ia-avancee-npy/` ;
4. `/mnt/user-data/uploads/`.

Le plus simple : créer un dossier `data/` à côté des notebooks et y déposer les trois fichiers. Si un fichier est introuvable, le notebook s'arrête avec un message d'erreur explicite.

## 5. Procédure de lancement

1. Installer l'environnement (section 3) et placer les données (section 4).
2. Lancer Jupyter (`jupyter notebook`) ou ouvrir les notebooks dans Colab / VS Code.
3. Pour **chaque notebook**, dans l'ordre **jalon 1 → 2 → 3 → 4** : menu *Exécuter → Tout exécuter* (*Run all*), sur un noyau neuf.
4. Ne pas relancer à la main les cellules de test : sur chaque notebook, **le test n'est évalué qu'une seule fois**, en fin de notebook (le jalon 4 lève une erreur si la cellule de test est rejouée dans la même session).

Les jalons sont indépendants : l'ordre 1 → 4 suit la progression de l'énoncé, mais aucun résultat n'est transmis d'un notebook à l'autre.

Exécution en ligne de commande (optionnel) :

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Groupe10_UC3_Jalon1.ipynb
# idem pour les jalons 2, 3 et 4
```

**Test de fumée (quelques minutes, pour vérifier que le code tourne ; résultats non significatifs) :**

| Jalon | Activation |
|---|---|
| 2 et 3 | variable d'environnement `UC3_RAPIDE=1` avant de lancer Jupyter (1 époque, données sous-échantillonnées) |
| 4 | `MODE_RAPIDE = True` dans la cellule « Réglages globaux » (1 époque, 1 000 fenêtres) |

## 6. Matériel de calcul et temps d'exécution

Les expérimentations ont été réalisées dans l'environnement Google Colab, en utilisant une instance standard équipée d'un GPU NVIDIA T4 (avec 2 cœurs CPU et environ 16 Go de RAM alloués).

| Jalon | Matériel utilisé | Temps d'exécution total (approx.) |
|---|---|---|
| 1 | Google Colab (GPU NVIDIA T4) | ≈ 30 min (5 graines × 5 pertes, ablation, validation croisée du seuil FA) |
| 2 | Google Colab (GPU NVIDIA T4) | ≈ 30 min (CNN 1D, entraînement des AE, comparaisons) |
| 3 | Google Colab (GPU NVIDIA T4) | ≈ 40 min (transfert, augmentation, multitâche) |
| 4 | Google Colab (GPU NVIDIA T4) | ≈ 60 min (RNN, LSTM, GRU, ablation et stabilité) |

La configuration matérielle standard de Colab T4 suffit très largement pour tous les jalons. Les mesures de latence effectuées au jalon 4 portent sur ce GPU Colab, et non sur une cible embarquée réelle (comme le patch), ce qui est une limite à garder en tête.

## 7. Reproductibilité

**Graines.**

| Jalon | Graines |
|---|---|
| 1 | `SEED_FINAL = 42` (modèle final, seuil, test) ; `SEEDS = [0, 1, 2, 3, 4]` (toutes les comparaisons) ; stabilité du seuil B : graines 42, 0, 1 |
| 2 | `SEED = 0` |
| 3 | `SEED = 42` ; `SEEDS_H = (0, 1)` (un CNN Holter par graine, adaptation avec la même graine) ; modèle final issu de la graine 0 ; bootstrap par patient avec graine 0 |
| 4 | `SEED = 42` (Python, NumPy, Keras) ; 3 graines de stabilité : 1, 2, 3 |

Les graines sont fixées pour Python, NumPy et Keras/TensorFlow (`fixer_graine` / `keras.utils.set_random_seed`).

**Variables de contrôle** (modifiables en tête de notebook).

| Jalon | Variable | Valeur livrée | Rôle |
|---|---|---|---|
| 1 | `USE_ES`, `ES_PATIENCE` | `True`, 15 | arrêt précoce sur `val_loss` (`False` → 60 époques fixes) |
| 1 | `K_MAD` | 6,0 *(déterminée par le notebook)* | seuil `médiane + k × MAD` du détecteur QRS, réglé sur le train seul |
| 1 | `PERTE` | entropie croisée simple (`ce`) *(déterminée)* | perte retenue par la règle de la section 5 |
| 1 | `CONFIG` | « sans bandes » : groupes base + amplitude/largeur + irrégularité *(déterminée)* | configuration de descripteurs retenue par l'ablation (section 6) |
| 1 | `S_B` (et `S_A`) | 0,039 (et 0,049) *(déterminés)* | seuil d'alerte FA à spécificité 95 % : B calibré hors pli sur train + validation (retenu), A sur la validation seule |
| 2 et 3 | `RAPIDE` (`UC3_RAPIDE`) | `0` | mode test de fumée |
| 2 et 3 | `UC3_DATA` | non définie | dossier des données |
| 3 | `EP_H`, `USE_ES` | 12, `True` | époques maximales et arrêt précoce (patience 3) du CNN Holter |
| 3 | `RECALCULER` | `False` | si `modeles_jalon3/` contient déjà un modèle, il est rechargé au lieu d'être réentraîné ; mettre `True` (ou vider le dossier) pour tout recalculer |
| 4 | `MODE_RAPIDE`, `STABILITE` | `False`, `True` | test de fumée ; répétition sur 3 graines des deux modèles les moins coûteux |

Les valeurs marquées *(déterminée)* ne sont pas saisies à la main : elles sont calculées par le notebook selon des règles de décision fixées avant d'examiner le test. Les valeurs indiquées sont celles de l'exécution livrée ; une réexécution sur un autre environnement peut les faire légèrement varier.

**Protocole.** Tous les réglages sont faits sur la validation (ou en validation croisée groupée par patient) ; le test n'est utilisé qu'une seule fois, pour le modèle figé. Chaque fenêtre est normalisée individuellement (z-score par fenêtre) : aucune statistique n'est apprise sur un ensemble, donc pas de fuite.

**Non-déterminisme résiduel.** Sur CPU, TensorFlow active les opérations oneDNN (légers écarts d'arrondi possibles) ; sur GPU, certaines opérations cuDNN ne sont pas strictement déterministes. C'est pourquoi les comparaisons sont répétées sur plusieurs graines (5 au jalon 1, 2 au jalon 3, 3 au jalon 4) et que les écarts sont donnés avec leur écart-type.

---

## 8. Interface Streamlit

L'application déploie les modèles des quatre jalons : **un onglet par jalon**. On fournit un signal ECG, l'application applique **les mêmes prétraitements que dans le notebook**, puis affiche la prédiction avec une interprétation.

> ⚠️ Les signaux sont simulés : aucune conclusion médicale.

### Lancement rapide

```bash
cd interface
pip install -r requirements.txt        # Python ≥ 3.10 ; TensorFlow/Keras 3 requis (modèles .keras)
streamlit run app.py
```

Avec `uv` :

```bash
cd interface
uv sync
uv run streamlit run app.py
```

> Utiliser, si possible, **les mêmes versions de TensorFlow/Keras que dans les notebooks** (rappelées dans `artifacts/jalonN/config.json`).

### Préparer les artefacts (une fois par jalon)

L'application ne ré-entraîne rien : elle charge les modèles et résultats produits par les notebooks.

1. Exécuter `jalonN.ipynb` **en entier** (avec les fichiers `.npz`).
2. Ajouter en dernière cellule le contenu de `export_jalonN.py` et l'exécuter : cela crée `artifacts/jalonN/` (modèles `.keras`, `config.json`, tableaux de résultats, fenêtres d'exemple).
3. Copier `artifacts/jalonN/` dans `interface/artifacts/`, à côté de `app.py`.

Si un dossier manque, l'onglet concerné affiche les instructions ; les autres onglets continuent de fonctionner.

### Trois vues par onglet

| Vue | Contenu |
|---|---|
|  **Démonstration** | choisir une entrée et un modèle, lire la décision et son interprétation |
|  **Résultats & expériences** | tous les essais du notebook (tableaux, courbes, test final, journal), lus depuis les fichiers exportés |
|  **Vérifications théoriques** | les calculs demandés par les consignes, rendus interactifs |

**Entrées acceptées :** exemples exportés du jeu de test (avec la bonne réponse connue) ; fichier `.csv`, `.txt` ou `.npy` contenant **une seule fenêtre de 1 250 valeurs** (10 s à 125 Hz, en mV) ; selon l'onglet, un signal modifié (bruit, augmentation) pour tester la robustesse.

### Ce que propose chaque onglet

| Jalon | Démonstration | Théorie interactive |
|---|---|---|
| **1 — Descripteurs → décision** | 14 modèles (retenu ★, pertes, ablation), seuil d'alerte FA (A / B retenu / manuel), probabilités, QRS et intervalles RR, interprétation clinique, poids des groupes de descripteurs | poids de classes `w_k = N/(K·N_k)` ; entropie croisée et gradient `p − y` |
| **2 — Signal brut (CNN 1D, AE)** | Holter, patch ou Holter + bruit « patch » ; classifieur et débruiteur au choix ; décision **sans** puis **avec** débruitage ; sensibilité temporelle (occlusion 1 s) ; latences et paramètres | calculateur de champ réceptif (4 blocs noyau 7 → 106 éch. ; dilatations 1-2-4-8 → 526 éch.) |
| **3 — Holter → patch** | patch, Holter, Holter + augmentation, fenêtres inédites, fichier ; stratégies d'adaptation comparées à « Holter tel quel » ; FC prédite (multitâche) ; **diagnostic de décalage** | invariance de `z = (x − µ)/σ` au gain et au décalage ; paramètres entraînables (325 vs 51 365) |
| **4 — Séquences (RNN, LSTM, GRU)** | 6 modèles comparés avec latence mesurée ; modèle retenu (hybride CNN+GRU) face au seuil FA ; influence de chaque pas de temps (disparition du gradient) ; importance de chaque battement | `λ^T` interactif (disparition/explosion du gradient) ; paramètres SimpleRNN/LSTM/GRU vs Keras (LSTM 32 unités : 4 352) |

### Lire les résultats avec précaution

- **Données simulées** : les chiffres décrivent les jeux fournis, pas la clinique.
- **Les chiffres affichés sont ceux des notebooks** : l'application relit les fichiers exportés, elle n'invente aucune métrique.
- **Test utilisé une fois** : la vue « Résultats » n'affiche le test que pour le modèle retenu ; ne pas régler le seuil du jalon 1 sur le test.
- **Jalon 3** : les stratégies de la démonstration sont ré-entraînées à l'export sur les 18 patients annotés ; les tableaux « Résultats » restent ceux de la validation croisée du notebook.
- **Jalon 4** : le « GRU sans masquage » est ré-entraîné à l'export (écarts possibles si le GPU n'est pas déterministe) ; les latences viennent du matériel du notebook, pas d'un système embarqué.
- **Interprétations** (occlusion, gradient, diagnostic de décalage) : indicatives, sur une fenêtre isolée, pas des preuves.

### Dépannage

| Symptôme | Cause probable / solution |
|---|---|
| « Artefacts du jalon N introuvables » | exécuter la cellule d'export et copier `artifacts/jalonN/` |
| Erreur au chargement d'un modèle | version de TensorFlow/Keras différente ; installer les versions de `config.json` |
| « 1 250 valeurs attendues » | le fichier doit contenir **une** fenêtre de 10 s à 125 Hz |
| Avertissement « suite de battements différente » (jalon 4) | le détecteur de QRS du notebook diffère : adapter `common/signal4.py` |
| Premier affichage lent | chargement des modèles (mis en cache ensuite) ; le RNN simple est lent par nature |
