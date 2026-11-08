# 🌱 AgroCalculator — Système Décisionnel Agrotechnologique

Application Streamlit de **phénotypage non-destructif** et de **prédiction de la biomasse**
d'*Ocimum basilicum L.* (basilic) sous régimes de fertigation **N-P-K**.

À partir de traits morphologiques mesurés *in situ* (sans sacrifice végétal), le modèle
d'estimation prédit le **poids sec aérien (PDS AS)** et en déduit plusieurs indicateurs
physiologiques.

---

## ✨ Fonctionnalités

| Module | Description |
|---|---|
| 🧪 **Régime N-P-K** | Saisie du stade phénologique, de l'ET (mm/j) et des doses N / P / K (ppm) |
| 📐 **Traits non-destructifs** | Hauteur de tige, poids frais aérien, nombre de feuilles, surface foliaire |
| ⚡ **Estimations** | PDS AS (aérien sec), PDS RS (racinaire), ratio shoot/root |
| 🧬 **Diagnostics** | Teneur en matière sèche (DMC), teneur en eau, SLA (cm²/g) |
| 📈 **Cinétique** | Courbe de croissance prédictive interactive (Plotly) |
| 🕸️ **Radar** | Profil phénotypique normalisé (vigueur, surface, N, K, matière sèche) |
| 📑 **Export** | Fiche d'inférence agronomique téléchargeable en CSV |

**Performance du modèle :** R² = 0.9937 — Gradient Boosting Regressor.

---

## 🚀 Installation locale

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

L'application est servie sur **http://localhost:8501**.

---

## 🧠 Réentraîner le modèle

```bash
python3 train_model.py
```

Le script lit `1-Table 1.csv`, entraîne un `GradientBoostingRegressor` et régénère :

- `modele_basilic.pkl` — modèle sérialisé (joblib)
- `model_features.pkl` — ordre des variables d'entrée

> ⚠️ Les noms de colonnes du CSV doivent correspondre à :
> `SEMAINE, ET, Ni, Pi, Ki, HAUT, PDS A F, NBR F, SUR F` (cible : `PDS A S`).
> La correspondance est insensible aux espaces, underscores et à la casse.

---

## ☁️ Déploiement sur Streamlit Community Cloud

1. Pousser le dépôt sur GitHub.
2. Se rendre sur [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Sélectionner le dépôt, la branche `main` et le fichier `app.py`.
4. **Deploy**. Les fichiers `.pkl` étant versionnés, aucun réentraînement n'est nécessaire.

---

## 📁 Structure du projet

```
.
├── app.py                    # Interface Streamlit (UI + inférence)
├── train_model.py            # Pipeline d'entraînement et de sérialisation
├── modele_basilic.pkl        # Modèle entraîné
├── model_features.pkl        # Ordre/noms des features attendues
├── 1-Table 1.csv             # Jeu de données d'entraînement
├── requirements.txt          # Dépendances Python
├── .streamlit/config.toml    # Thème et configuration serveur
└── *.pdf                     # Documentation scientifique du projet
```

---

## 📊 Variables du modèle

**Entrées (9)**
`SEMAINE` · `ET` · `Ni` · `Pi` · `Ki` · `HAUT` · `PDS A F` · `NBR F` · `SUR F`

**Sortie**
`PDS A S` — poids sec aérien (g)

---

## ⚠️ Avertissement

Outil d'aide à la décision à vocation **expérimentale et pédagogique**. Les prédictions
doivent être validées par des mesures destructives de contrôle avant tout usage
agronomique à grande échelle.
