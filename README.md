# AntCar-CARLA

Navigation visuelle bio-inspirée (modèle fourmi) appliquée à un véhicule autonome dans CARLA Simulator.

## Description

Ce projet reprend le modèle de navigation des fourmis (*Cataglyphis*) basé sur les Mushroom Bodies (corps pédonculés) et l'adapte pour un véhicule simulé dans CARLA. Le principe est simple : le véhicule apprend une route en capturant des vues panoramiques basse résolution, puis il navigue en comparant la scène courante avec ce qu'il a mémorisé.

Deux Mushroom Bodies sont entraînés avec des oscillations dans des directions opposées, ce qui permet de calculer un signal de direction : si MB1 est plus "surpris" que MB2, le véhicule tourne d'un côté, et inversement.

Le code est basé sur les travaux de Gattaux et al. (2023, 2025) et utilise leur implémentation du Mushroom Body.

## Fichiers

- `src/memory.py` - Modèle neuronal Mushroom Body 
- `src/antcar_sim.py` - Traitement visuel et simulation offline 
- `src/ant_pip.py` - Pipeline complet CARLA : capture, entrainement, navigation
- `src/ant_pip_log.py` - Même chose que ant_pip.py mais avec logging XTE/HE pendant la navigation
- `src/meteo.py` - Expériences météo : teste la navigation sous pluie, nuit, brouillard, crépuscule
- `src/run_parallel_learning.py` - Expérience multi-trajets : apprentissage incrémental depuis plusieurs offsets latéraux


## Prérequis

- Python >= 3.8
- CARLA Simulator >= 0.9.13
- Voir `requirements.txt` pour les dépendances Python

## Installation

```bash
git clone <url>
cd antcar_carla_git
pip install -r requirements.txt
```

Il faut aussi lancer le serveur CARLA avant d'exécuter le pipeline (on utilise **Town05**) :
```bash
./CarlaUE4.sh   # Linux
```

## Utilisation

```bash
cd src/

# Entrainement offline (phases 1 a 4)
python ant_pip.py --mode offline

# Navigation dynamique (phase 5, necessite un modele deja entrainé)
python ant_pip.py --mode navigate --model_dir ./antcar_out

# Tout d'un coup
python ant_pip.py --mode all
```

## Pipeline

1. **Capture apprentissage** : le véhicule est téléporté le long de la route, on capture des images fisheye à chaque position
2. **Augmentation + entrainement** : on augmente les images par rotation (oscillations) et on entraîne les deux MBs
3. **Capture test** : même chose mais avec un décalage latéral
4. **Evaluation offline** : on mesure l'unfamiliarity, le Cross-Track Error et le Heading Error
5. **Navigation dynamique** : le véhicule roule en boucle fermée, guidé par le signal des MBs

