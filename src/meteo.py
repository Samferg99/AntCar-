#!/usr/bin/env python3
"""
Experiences meteo - Navigation bio-inspiree dans CARLA
Teste le modele sous differentes conditions meteo.
Le modele a ete appris en condition claire (jour, beau temps).

IMPORTANT :
  - Relancer CARLA frais avant chaque condition
  - Lancer une condition a la fois (changer dans main())

Usage:
  python meteo.py
"""

import os, sys, shutil, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---- Configuration ----

MODEL_DIR = "./antcar_out"
EXPERIMENTS_DIR = "./experiments"
N_RUNS = 10


# ---- Conditions meteo ----

METEO_NUIT = {
    "name": "nuit",
    "description": "Nuit - soleil a -30 deg sous l'horizon",
    "sun_altitude_angle": -30.0,
    "sun_azimuth_angle": 45.0,
    "cloudiness": 20.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "fog_density": 0.0,
    "fog_distance": 0.0,
    "fog_falloff": 0.0,
    "wetness": 0.0,
    "wind_intensity": 10.0,
}

METEO_PLUIE = {
    "name": "pluie",
    "description": "Pluie forte - 80% precipitations, flaques, vent",
    "sun_altitude_angle": 40.0,
    "sun_azimuth_angle": 45.0,
    "cloudiness": 80.0,
    "precipitation": 80.0,
    "precipitation_deposits": 60.0,
    "fog_density": 10.0,
    "fog_distance": 0.0,
    "fog_falloff": 0.0,
    "wetness": 80.0,
    "wind_intensity": 50.0,
}

METEO_BROUILLARD = {
    "name": "brouillard",
    "description": "Brouillard epais - visibilite ~20m",
    "sun_altitude_angle": 30.0,
    "sun_azimuth_angle": 45.0,
    "cloudiness": 90.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "fog_density": 70.0,
    "fog_distance": 20.0,
    "fog_falloff": 2.0,
    "wetness": 0.0,
    "wind_intensity": 10.0,
}

METEO_CREPUSCULE = {
    "name": "crepuscule",
    "description": "Crepuscule - soleil rasant 5 deg, ombres longues",
    "sun_altitude_angle": 5.0,
    "sun_azimuth_angle": 270.0,
    "cloudiness": 30.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "fog_density": 0.0,
    "fog_distance": 0.0,
    "fog_falloff": 0.0,
    "wetness": 0.0,
    "wind_intensity": 10.0,
}


# ---- Fonctions ----

def set_carla_weather(weather_params):
    """Applique la meteo dans CARLA."""
    import carla
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    params = {k: v for k, v in weather_params.items()
              if k not in ('name', 'description')}
    weather = carla.WeatherParameters(**params)
    world.set_weather(weather)


def run_one_navigation(model_dir, nav_output_dir):
    """Lance une navigation identique a ant_pip_log.py --mode navigate."""
    import carla
    import ant_pip_log
    from ant_pip_log import (CFG, connect_carla, spawn_vehicle, build_route,
                         FisheyeRig, AntNavigator, load_model,
                         run_navigation, restore_carla)

    cfg = CFG()
    cfg.OUTDIR = model_dir
    cfg.NAV_MAX_STEPS = 3000

    client, world, cmap = connect_carla(cfg)
    orig_settings = world.get_settings()

    try:
        vehicle, spawn_tf = spawn_vehicle(world, cmap, cfg)
        rig = FisheyeRig(world, vehicle, cfg)

        learn_dir = os.path.join(cfg.OUTDIR, "images/raw_learn/")
        sim = load_model(learn_dir, cfg.OUTDIR, cfg)
        navigator = AntNavigator(sim, cfg)

        route = build_route(cmap, spawn_tf,
                            cfg.N_POINTS_LEARN, cfg.STEP_M,
                            lateral_offset=0.0)

        run_navigation(world, vehicle, rig, navigator,
                       route, cfg, nav_output_dir)

    finally:
        try: rig.destroy()
        except: pass
        try: vehicle.destroy()
        except: pass
        restore_carla(world, orig_settings)

    return os.path.join(nav_output_dir, "nav_log.csv")


def run_meteo_condition(meteo, n_runs=N_RUNS):
    """Lance N runs sous une condition meteo."""
    name = meteo["name"]
    cond_dir = os.path.join(EXPERIMENTS_DIR, name)
    os.makedirs(cond_dir, exist_ok=True)

    with open(os.path.join(cond_dir, "condition_config.json"), "w") as f:
        json.dump(meteo, f, indent=2, ensure_ascii=False)

    print(f"\n---- CONDITION : {name} ----")
    print(f"  {meteo['description']}")
    print(f"  {n_runs} runs")

    all_metrics = []

    for run_id in range(n_runs):
        run_dir = os.path.join(cond_dir, f"run_{run_id:02d}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n  Run {run_id+1}/{n_runs}")

        # appliquer la meteo avant chaque run
        print(f"    Meteo : {name} (sun={meteo['sun_altitude_angle']} deg, "
              f"precip={meteo['precipitation']}%, fog={meteo['fog_density']}%)")
        set_carla_weather(meteo)
        time.sleep(2.0)

        t0 = time.time()

        try:
            log_path = run_one_navigation(
                model_dir=MODEL_DIR,
                nav_output_dir=run_dir,
            )

            df = pd.read_csv(log_path)
            xte = df["xte"].abs()
            he = df["he"].abs()

            metrics = {
                "run_id": run_id,
                "condition": name,
                "n_steps": len(df),
                "max_wp": int(df["nearest_idx"].max()),
                "xte_median": round(float(xte.median()), 4),
                "xte_mad": round(float((xte - xte.median()).abs().median()), 4),
                "xte_max": round(float(xte.max()), 4),
                "he_median": round(float(he.median()), 2),
                "he_mad": round(float((he - he.median()).abs().median()), 2),
                "he_max": round(float(he.max()), 2),
                "speed_median": round(float(df["speed"].median()), 2),
                "duration_s": round(time.time() - t0, 1),
            }
            all_metrics.append(metrics)

            print(f"    OK  XTE = {metrics['xte_median']:.3f} +/- {metrics['xte_mad']:.3f} m  |  "
                  f"HE = {metrics['he_median']:.1f} +/- {metrics['he_mad']:.1f} deg  |  "
                  f"wp = {metrics['max_wp']}  |  {metrics['duration_s']}s")

        except Exception as e:
            print(f"    ERREUR : {e}")
            import traceback
            traceback.print_exc()
            all_metrics.append({
                "run_id": run_id, "condition": name, "error": str(e)
            })

        time.sleep(3.0)

    # sauvegarder
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(cond_dir, "metrics_all_runs.csv"), index=False)

    # resume
    valid = [m for m in all_metrics if "xte_median" in m]
    if valid:
        xte_meds = [m["xte_median"] for m in valid]
        he_meds = [m["he_median"] for m in valid]
        print(f"\n  RESUME {name} ({len(valid)}/{n_runs} runs OK)")
        print(f"    XTE : {np.median(xte_meds):.3f} +/- "
              f"{np.median(np.abs(np.array(xte_meds) - np.median(xte_meds))):.3f} m")
        print(f"    HE  : {np.median(he_meds):.1f} +/- "
              f"{np.median(np.abs(np.array(he_meds) - np.median(he_meds))):.1f} deg")

    return all_metrics


# ---- Main ----
# Decommente la condition a tester, relance CARLA frais avant chaque.

def main():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    weights = os.path.join(MODEL_DIR, "mb_weights.npz")
    if not os.path.exists(weights):
        print(f"ERREUR : {weights} introuvable.")
        print("Lance d'abord : python ant_pip.py --mode offline")
        sys.exit(1)

    t_global = time.time()

    #run_meteo_condition(METEO_NUIT)
    #run_meteo_condition(METEO_PLUIE)
    #run_meteo_condition(METEO_BROUILLARD)
    run_meteo_condition(METEO_CREPUSCULE)

    dt = time.time() - t_global
    print(f"\nTerminé en {dt/60:.1f} min")
    print(f"Resultats dans : {EXPERIMENTS_DIR}/")


if __name__ == "__main__":
    main()
