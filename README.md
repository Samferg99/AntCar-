# AntCar — Bio-Inspired Visual Navigation in CARLA

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

> Navigation visuelle bio-inspirée basée sur le modèle neuronal **Mushroom Body** de la fourmi *Cataglyphis*, adapté à un véhicule autonome simulé dans CARLA.

Projet de recherche — M2 Intelligence Artificielle & Robotique, [ETIS Laboratory](https://www.etis-lab.fr/) (CNRS UMR 8051 / CY Cergy Paris Université / ENSEA)  
Encadrant : **Nicolas Cuperlier**

---

## Principe

Le véhicule apprend une route en capturant des vues fisheye panoramiques basse résolution (32×32 px), sans carte ni GPS. Six caméras perspective en configuration cubemap sont reprojetées en une image fisheye équidistante 512×512 px. L'image est encodée via canal vert + flou gaussien (σ=3 px) + filtre Sobel, produisant ~800 Projection Neurons (PNs).

Deux Mushroom Bodies (MB1, MB2) latéralisés sont entraînés en one-shot anti-Hebbien avec des oscillations angulaires opposées (MB1 : +5°→+45°, MB2 : −5°→−45°). En navigation, le signal de direction est le déséquilibre d'unfamiliarity entre les deux circuits : Δλ = λ1 − λ2. Le steering est lissé par filtre EWA (α=0.15) pour filtrer le bruit de quantification du κ-WTA.

Le modèle est directement issu des travaux de **Gattaux et al. (2025, Nature Communications)**, ici transféré à l'échelle d'un véhicule urbain simulé (Tesla Model 3, CARLA Town05).

---

## Résultats

Expériences sur une route de **300 waypoints (300 m) dans Town05** (CARLA 0.9.15, ClearNoon) :

### Protocole 1 — Baseline (10 runs, offset 0 m)

| Métrique | Runs réussis (n=8) | Tous (n=10) |
|---|---|---|
| Taux de succès | **80%** | — |
| XTE médian (m) | **0.53 ± 0.06** | 0.60 ± 0.13 |
| XTE max absolu (m) | **3.78** | — |
| HE médian (°) | 2.75 ± 0.39 | 2.79 ± 0.57 |
| Vitesse médiane (km/h) | 7.12 ± 0.17 | 7.00 ± 0.29 |

### Protocole 2 — Robustesse aux offsets latéraux (1 run par offset)

Succès de −2.0 m à +2.0 m (8/10), avec deux échecs aux offsets −0.5 m et −1.0 m (composante stochastique du signal κ-WTA). XTE médian convergé : 0.82 ± 0.13 m indépendamment de l'offset initial.

### Protocole 3 — Sensibilité météo

Échec total sous toutes les conditions testées : Night, HardRain, Fog, Sunset. Le modèle encode des gradients Sobel spécifiques aux conditions d'apprentissage — limitation fondamentale de l'encodage purement visuel, biologiquement cohérente.

### Comparaison normalisée avec AntCar physique

| | AntCar robot [Gattaux 2025] | Ce travail |
|---|---|---|
| XTE médian / largeur de voie | 0.15 | 0.19 |
| Taux de succès | 100% (11/11) | 80% (8/10) |
| Route | 1.6 km (réel) | 300 m (simulé) |

---

## Structure du projet

```
AntCar-/
├── src/
│   ├── memory.py                  # Modèle neuronal Mushroom Body (PN→KC→MBON)
│   ├── antcar_sim.py              # Traitement visuel et simulation offline
│   ├── ant_pip.py                 # Pipeline complet CARLA : capture, entraînement, navigation
│   ├── ant_pip_log.py             # Pipeline avec logging XTE/HE
│   ├── meteo.py                   # Expériences météo (pluie, nuit, brouillard, crépuscule)
│   └── run_parallel_learning.py   # Apprentissage multi-trajets (offsets latéraux)
├── paper/                         # Paper final (IEEE format)
├── results/                       # Figures de résultats
├── requirements.txt
└── README.md
```

---

## Paramètres clés

| Paramètre | Valeur |
|---|---|
| Résolution visuelle | 32×32 px |
| Projection Neurons | ~800 |
| Kenyon Cells (NKC) | 15 000 |
| Sparsité κ | 0.005 (75 KCs actifs) |
| Oscillations | ±45°, pas 5° |
| Lissage steering α | 0.15 |
| Gain KP | 25 |
| Clip steering | ±0.75 |
| Vitesse | 2–6 km/h |
| Fréquence de contrôle | ~5 Hz |

---

## Prérequis

- Python >= 3.8
- CARLA Simulator 0.9.15 (map Town05)
- Voir `requirements.txt` pour les dépendances Python

---

## Installation

```bash
git clone https://github.com/Samferg99/AntCar-.git
cd AntCar-
pip install -r requirements.txt
```

Lancer le serveur CARLA avant d'exécuter le pipeline :
```bash
./CarlaUE4.sh        # Linux
CarlaUE4.exe         # Windows
```

---

## Utilisation

```bash
cd src/

# Entraînement offline (phases 1 à 4)
python ant_pip.py --mode offline

# Navigation dynamique (phase 5, nécessite un modèle déjà entraîné)
python ant_pip.py --mode navigate --model_dir ./antcar_out

# Pipeline complet
python ant_pip.py --mode all
```

---

## Pipeline

1. **Capture apprentissage** — téléportation statique le long des 300 waypoints, capture cubemap → fisheye à chaque position
2. **Augmentation + entraînement** — rotation horizontale ±45° (pas 5°), entraînement anti-Hebbien one-shot des deux MBs (3 000 patterns chacun)
3. **Capture test** — même procédure avec décalage latéral configurable
4. **Évaluation offline** — unfamiliarity, XTE et HE sur les images test statiques
5. **Navigation dynamique** — boucle fermée temps réel, steering piloté par Δλ = λ1 − λ2, vitesse modulée par λmin

---

## Références

- Gattaux et al. (2025). *Route-centric ant-inspired memories enable panoramic route-following in a car-like robot*. Nature Communications, 16, 8328.
- Gattaux et al. (2023). *AntCar: Simple Route Following Task with Ants-Inspired Vision and Neural Model*. HAL preprint hal-04060451.
- Ardin et al. (2016). *Using an insect mushroom body circuit to encode route memory in complex natural environments*. PLoS Computational Biology.
- Dosovitskiy et al. (2017). *CARLA: An open urban driving simulator*. CoRL.
