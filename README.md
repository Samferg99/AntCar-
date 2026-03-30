# AntCar — Bio-Inspired Visual Navigation in CARLA

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

> Navigation visuelle bio-inspirée basée sur le modèle neuronal **Mushroom Body** de la fourmi *Cataglyphis*, adapté à un véhicule autonome simulé dans CARLA.

Projet de recherche — M2 Intelligence Artificielle & Robotique, [ETIS Laboratory](https://www.etis-lab.fr/) (CNRS UMR 8051 / CY Cergy Paris Université / ENSEA)  
Encadrant : **Nicolas Cuperlier**

---

## Principe

Le véhicule apprend une route en capturant des vues panoramiques fisheye basse résolution (32×32 px), sans carte ni GPS. Deux Mushroom Bodies (MBs) latéralisés sont entraînés avec des oscillations en directions opposées. Pendant la navigation, le signal de direction est calculé à partir du déséquilibre d'*unfamiliarity* entre les deux MBs : si MB1 est plus "surpris" que MB2, le véhicule tourne d'un côté, et inversement.

Le modèle est directement inspiré des travaux de **Gattaux et al. (2023, 2025)** sur mini-robot, ici transféré à l'échelle d'un véhicule urbain simulé.

---

## Résultats

Expériences sur une route de **300 waypoints dans Town05** (CARLA 0.9.15) :

| Condition | Taux de succès | XTE médian | XTE max absolu |
|-----------|---------------|------------|----------------|
| Baseline (0 m offset) | 10/10 | **0.53 ± 0.06 m** | 3.78 m |
| Offset latéral (±0.5 m) | 8/10 | 0.71 ± 0.12 m | — |

Figures disponibles dans `results/`.

---

## Structure du projet

```
AntCar-/
├── src/
│   ├── memory.py               # Modèle neuronal Mushroom Body
│   ├── antcar_sim.py           # Traitement visuel et simulation offline
│   ├── ant_pip.py              # Pipeline complet CARLA : capture, entraînement, navigation
│   ├── ant_pip_log.py          # Pipeline avec logging XTE/HE
│   ├── meteo.py                # Expériences météo (pluie, nuit, brouillard, crépuscule)
│   └── run_parallel_learning.py # Apprentissage multi-trajets depuis plusieurs offsets
├── paper/                      # Paper final (IEEE format)
├── results/                    # Figures de résultats
├── requirements.txt
└── README.md
```

---

## Prérequis

- Python >= 3.8
- CARLA Simulator >= 0.9.15 (testé sur Town05)
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

1. **Capture apprentissage** — le véhicule est téléporté le long de la route, images fisheye capturées à chaque waypoint
2. **Augmentation + entraînement** — rotation des images (oscillations ±θ) et entraînement anti-Hebbien des deux MBs
3. **Capture test** — même procédure avec un décalage latéral configurable
4. **Évaluation offline** — calcul de l'unfamiliarity, du Cross-Track Error (XTE) et du Heading Error (HE)
5. **Navigation dynamique** — boucle fermée temps réel, signal de steering issu du déséquilibre MB1/MB2

---

## Références

- Gattaux et al. (2025). *Lateralized mushroom body model for visual route navigation*. Nature Communications.
- Gattaux et al. (2023). *AntCar: bio-inspired navigation on a mini-robot*. IEEE/AICAS-HAL.
- Ardin et al. (2016). *Using an insect mushroom body circuit to encode route memory*. PLOS Computational Biology.
- Dosovitskiy et al. (2017). *CARLA: An open urban driving simulator*. CoRL.
