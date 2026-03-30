#!/usr/bin/env python3
"""
Experience : Apprentissage multi-trajets paralleles

Teste si le MB peut apprendre plus en enrichissant la memoire
avec des vues depuis des positions laterales differentes.

Principe :
  - Passage 1 : apprentissage au centre de la voie (offset = 0m)
  - Passage 2 : apprentissage a +1m (droite)
  - Passage 3 : apprentissage a -1m (gauche)

A chaque etape on mesure la saturation KC et la performance (XTE/HE).
Les poids MB ne sont PAS reinitialises entre les passages :
on accumule les souvenirs comme une fourmi qui emprunte plusieurs
fois le meme couloir.

Usage:
  1. Lancer CARLA
  2. python run_parallel_learning.py

Prerequis:
  - ant_pip.py, antcar_sim.py, memory.py dans le meme dossier
  - Un apprentissage initial deja fait (dossier antcar_out/)
    OU le script fera la capture + apprentissage initial

Duree estimee : ~2-3h (capture + 3 entrainements + 30 runs navigation)
"""

import os, sys, time, math, shutil, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ant_pip
import antcar_sim
import memory
import carla


# ---- Configuration ----

PARALLEL_OFFSETS = [0.0, 1.0, -1.0]
OFFSET_NAMES = ["centre (0m)", "droite (+1m)", "gauche (-1m)"]
N_RUNS = 10
NAV_START_OFFSET = 0.0
BASE_DIR = "./parallel_learning_exp"


# ---- Etape 1 : capture des images depuis les routes paralleles ----

def capture_parallel_routes(cfg):
    """Capture les images fisheye depuis chaque route parallele."""
    print("\n---- Etape 1 : Capture des routes paralleles ----")

    client, world, cmap = ant_pip.connect_carla(cfg)
    orig_settings = world.get_settings()

    image_dirs = {}

    try:
        vehicle, spawn_tf = ant_pip.spawn_vehicle(world, cmap, cfg)
        rig = ant_pip.FisheyeRig(world, vehicle, cfg)

        for i, offset in enumerate(PARALLEL_OFFSETS):
            name = f"route_offset_{offset:+.1f}m"
            img_dir = os.path.join(BASE_DIR, "images", name)

            if os.path.exists(img_dir) and len(os.listdir(img_dir)) >= cfg.N_POINTS_LEARN:
                print(f"\n  [{i+1}/{len(PARALLEL_OFFSETS)}] {OFFSET_NAMES[i]} - "
                      f"deja capturé ({img_dir}), skip.")
                image_dirs[offset] = img_dir
                continue

            print(f"\n  [{i+1}/{len(PARALLEL_OFFSETS)}] Capture {OFFSET_NAMES[i]}...")

            route = ant_pip.build_route(
                cmap, spawn_tf,
                cfg.N_POINTS_LEARN, cfg.STEP_M,
                lateral_offset=offset
            )

            ant_pip.capture_static_route(world, vehicle, rig, route, img_dir, cfg)
            image_dirs[offset] = img_dir

    finally:
        rig.destroy()
        try: vehicle.destroy()
        except Exception: pass
        ant_pip.restore_carla(world, orig_settings)

    return image_dirs


# ---- Etape 2 : apprentissage incremental ----

def incremental_training(image_dirs, cfg):
    """
    Entraine le MB de facon incrementale :
      - Passage 1 : apprend centre -> sauvegarde poids
      - Passage 2 : charge poids, apprend +1m -> sauvegarde
      - Passage 3 : charge poids, apprend -1m -> sauvegarde
    """
    print("\n---- Etape 2 : Apprentissage incremental ----")

    results = {}

    center_dir = image_dirs[0.0]
    # dossier test bidon (obligatoire pour SimpleArgs mais pas utilisé ici)
    dummy_test = os.path.join(BASE_DIR, "images", "dummy_test")
    os.makedirs(dummy_test, exist_ok=True)
    first_img = sorted([f for f in os.listdir(center_dir) if f.endswith(".jpg")])[:1]
    for f in first_img:
        src = os.path.join(center_dir, f)
        dst = os.path.join(dummy_test, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    prev_weights = None

    for passage_idx, offset in enumerate(PARALLEL_OFFSETS):
        n_passages = passage_idx + 1
        img_dir = image_dirs[offset]

        print(f"\n  Passage {n_passages}/{len(PARALLEL_OFFSETS)} : {OFFSET_NAMES[passage_idx]}")

        out_dir = os.path.join(BASE_DIR, f"model_{n_passages}_passages")
        os.makedirs(out_dir, exist_ok=True)

        # creer le AntCarSim (reinitialise les MB)
        args = ant_pip.SimpleArgs(img_dir, dummy_test, out_dir, cfg)
        sim = antcar_sim.AntCarSim(args)

        # charger les poids du passage precedent si dispo
        if prev_weights is not None:
            sim._mbs[0].load_memory(prev_weights["W_mb1"])
            sim._mbs[1].load_memory(prev_weights["W_mb2"])
            kc_L_before = 1.0 - sim._mbs[0].get_unused_KC_ratio()
            kc_R_before = 1.0 - sim._mbs[1].get_unused_KC_ratio()
            print(f"  Poids chargés du passage precedent")
            print(f"  KC utilises avant : L={kc_L_before:.1%}  R={kc_R_before:.1%}")

        ant_pip._purge_non_images(img_dir)

        _, traj = sim.read_raw_img_folder(img_dir)
        _, uidx = np.unique(np.round(traj[:, 1:3], 2), axis=0, return_index=True)
        traj = traj[np.sort(uidx)]

        # augmentation et entrainement
        step = cfg.OSCIL_STEP
        maxo = cfg.OSCIL_MAX
        arr_pos = np.arange(0, maxo + step, step, dtype="int")
        arr_neg = np.arange(0, -(maxo + step), -step, dtype="int")

        src_mb1 = os.path.join(out_dir, "images/learning1/augmented/")
        src_mb2 = os.path.join(out_dir, "images/learning2/augmented/")

        print(f"  Augmentation ({OFFSET_NAMES[passage_idx]})...")
        sim.augment_dataset(img_dir, src_mb1, arr_pos, batch_size=cfg.BATCH_SIZE)
        sim.augment_dataset(img_dir, src_mb2, arr_neg, batch_size=cfg.BATCH_SIZE)

        print(f"  Entrainement MB1 (oscil positive)...")
        _, mem1 = sim.train(0, src_mb1, batch_size=cfg.BATCH_SIZE)

        print(f"  Entrainement MB2 (oscil negative)...")
        _, mem2 = sim.train(1, src_mb2, batch_size=cfg.BATCH_SIZE)

        # mesurer la saturation KC
        kc_ratio_L = 1.0 - sim._mbs[0].get_unused_KC_ratio()
        kc_ratio_R = 1.0 - sim._mbs[1].get_unused_KC_ratio()
        kc_remaining_L = sim._mbs[0].get_unused_KC_ratio()
        kc_remaining_R = sim._mbs[1].get_unused_KC_ratio()

        print(f"  Apres {n_passages} passage(s) :")
        print(f"    MBON_L : {kc_ratio_L:.1%} KC utilises ({kc_remaining_L:.1%} libres)")
        print(f"    MBON_R : {kc_ratio_R:.1%} KC utilises ({kc_remaining_R:.1%} libres)")

        # sauvegarder les poids
        weights_path = os.path.join(out_dir, "mb_weights.npz")
        W_mb1 = np.copy(sim._mbs[0].W_KCtoMBON)
        W_mb2 = np.copy(sim._mbs[1].W_KCtoMBON)
        np.savez(weights_path, W_mb1=W_mb1, W_mb2=W_mb2)
        print(f"  Poids sauvegardes -> {weights_path}")

        prev_weights = {"W_mb1": W_mb1, "W_mb2": W_mb2}

        results[n_passages] = {
            "weights_path": weights_path,
            "kc_used_L": kc_ratio_L,
            "kc_used_R": kc_ratio_R,
            "kc_free_L": kc_remaining_L,
            "kc_free_R": kc_remaining_R,
            "sim": sim,
        }

    return results


# ---- Etape 3 : tests de navigation ----

def run_navigation_tests(training_results, cfg):
    """Pour chaque niveau d'apprentissage, lance N_RUNS de navigation."""
    print("\n---- Etape 3 : Tests de navigation ----")

    client, world, cmap = ant_pip.connect_carla(cfg)
    orig_settings = world.get_settings()

    all_nav_results = {}

    try:
        vehicle, spawn_tf = ant_pip.spawn_vehicle(world, cmap, cfg)
        rig = ant_pip.FisheyeRig(world, vehicle, cfg)

        route_center = ant_pip.build_route(
            cmap, spawn_tf, cfg.N_POINTS_LEARN, cfg.STEP_M, lateral_offset=0.0
        )
        route_array = np.array([
            [tf.location.x, tf.location.y, tf.rotation.yaw]
            for tf in route_center
        ])

        for n_passages, data in training_results.items():
            print(f"\n  Navigation avec {n_passages} passage(s)")

            sim = data["sim"]
            navigator = ant_pip.AntNavigator(sim, cfg)

            nav_dir = os.path.join(BASE_DIR, f"navigation_{n_passages}_passages")
            os.makedirs(nav_dir, exist_ok=True)

            run_metrics = []

            for run_idx in range(N_RUNS):
                print(f"\n    Run {run_idx+1}/{N_RUNS}")

                run_dir = os.path.join(nav_dir, f"run_{run_idx:02d}")
                os.makedirs(run_dir, exist_ok=True)

                navigator._steer_prev = 0.0

                try:
                    ant_pip.run_navigation(
                        world, vehicle, rig, navigator,
                        route_center, cfg, run_dir
                    )
                except Exception as e:
                    print(f"    [ERREUR] Run {run_idx}: {e}")
                    run_metrics.append({
                        "run": run_idx,
                        "n_passages": n_passages,
                        "success": False,
                        "xte_median": np.nan,
                        "he_median": np.nan,
                    })
                    continue

                log_path = os.path.join(run_dir, "nav_log.csv")
                if not os.path.exists(log_path):
                    print(f"    [ERREUR] Pas de nav_log.csv")
                    continue

                df = pd.read_csv(log_path)

                # calcul XTE et HE par rapport a la route centre
                xte_list = []
                he_list = []
                for _, row in df.iterrows():
                    vx, vy, vyaw = row["x"], row["y"], row["yaw"]
                    dists = np.sqrt((route_array[:, 0] - vx)**2 +
                                   (route_array[:, 1] - vy)**2)
                    nearest_idx = np.argmin(dists)
                    xte = dists[nearest_idx]
                    he = vyaw - route_array[nearest_idx, 2]
                    he = (he + 180) % 360 - 180
                    xte_list.append(xte)
                    he_list.append(abs(he))

                xte_arr = np.array(xte_list)
                he_arr = np.array(he_list)

                # critere de succes : vehicule atteint au moins 80% de la route
                max_idx = 0
                for _, row in df.iterrows():
                    vx, vy = row["x"], row["y"]
                    dists = np.sqrt((route_array[:, 0] - vx)**2 +
                                   (route_array[:, 1] - vy)**2)
                    idx = np.argmin(dists)
                    max_idx = max(max_idx, idx)

                success = max_idx >= 0.8 * len(route_array)

                xte_med = np.median(xte_arr)
                he_med = np.median(he_arr)
                xte_mad = np.median(np.abs(xte_arr - xte_med))
                he_mad = np.median(np.abs(he_arr - he_med))

                status = "OK" if success else "FAIL"
                print(f"    {status} XTE={xte_med:.2f}+/-{xte_mad:.2f}m  "
                      f"HE={he_med:.1f}+/-{he_mad:.1f}deg  "
                      f"(wp max={max_idx}/{len(route_array)})")

                run_metrics.append({
                    "run": run_idx,
                    "n_passages": n_passages,
                    "success": success,
                    "xte_median": xte_med,
                    "xte_mad": xte_mad,
                    "he_median": he_med,
                    "he_mad": he_mad,
                    "max_wp": max_idx,
                    "total_wp": len(route_array),
                    "steps": len(df),
                })

            metrics_df = pd.DataFrame(run_metrics)
            metrics_path = os.path.join(nav_dir, "run_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)

            all_nav_results[n_passages] = metrics_df
            print(f"\n  Metriques -> {metrics_path}")

    finally:
        rig.destroy()
        try: vehicle.destroy()
        except Exception: pass
        ant_pip.restore_carla(world, orig_settings)

    return all_nav_results


# ---- Etape 4 : graphiques et analyse ----

def plot_results(training_results, nav_results):
    """Genere les graphiques de l'experience."""
    print("\n---- Etape 4 : Graphiques ----")

    fig_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    passages = sorted(training_results.keys())

    # figure 1 : saturation KC
    fig, ax = plt.subplots(figsize=(8, 5))

    kc_used_L = [training_results[p]["kc_used_L"] * 100 for p in passages]
    kc_used_R = [training_results[p]["kc_used_R"] * 100 for p in passages]

    x = np.arange(len(passages))
    width = 0.35
    bars_L = ax.bar(x - width/2, kc_used_L, width, label="MBON Gauche",
                    color="steelblue", alpha=0.8)
    bars_R = ax.bar(x + width/2, kc_used_R, width, label="MBON Droit",
                    color="tomato", alpha=0.8)

    ax.axhline(85, color="red", ls="--", alpha=0.5, label="Seuil critique (~85%)")
    ax.axhline(90, color="darkred", ls="--", alpha=0.5, label="Saturation (~90%)")

    for bars in [bars_L, bars_R]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9)

    ax.set_xlabel("Nombre de passages d'apprentissage")
    ax.set_ylabel("KC utilises (%)")
    ax.set_title("Saturation memoire du Mushroom Body\nen fonction du nombre de trajets paralleles")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} passage{'s' if p>1 else ''}\n({OFFSET_NAMES[i]})"
                        for i, p in enumerate(passages)])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(fig_dir, "01_kc_saturation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")

    # figure 2 : boxplots XTE et HE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    xte_data = []
    he_data = []
    labels = []
    colors = ["#90CAF9", "#A5D6A7", "#FFCC80"]

    for i, p in enumerate(passages):
        df = nav_results.get(p)
        if df is None or df.empty:
            continue
        ok = df[df["success"] == True]
        xte_data.append(ok["xte_median"].values)
        he_data.append(ok["he_median"].values)
        labels.append(f"{p} passage{'s' if p>1 else ''}\n(n={len(ok)})")

    bp1 = axes[0].boxplot(xte_data, labels=labels, patch_artist=True, widths=0.5)
    for j, patch in enumerate(bp1["boxes"]):
        patch.set_facecolor(colors[j % len(colors)])
    for j, data in enumerate(xte_data):
        x_jitter = np.random.normal(j + 1, 0.03, len(data))
        axes[0].scatter(x_jitter, data, alpha=0.6, s=30, zorder=3,
                        color=["steelblue", "green", "orange"][j % 3])
    axes[0].set_ylabel("XTE median du run (m)")
    axes[0].set_title("Erreur laterale (XTE)")
    axes[0].grid(True, alpha=0.3)

    bp2 = axes[1].boxplot(he_data, labels=labels, patch_artist=True, widths=0.5)
    for j, patch in enumerate(bp2["boxes"]):
        patch.set_facecolor(colors[j % len(colors)])
    for j, data in enumerate(he_data):
        x_jitter = np.random.normal(j + 1, 0.03, len(data))
        axes[1].scatter(x_jitter, data, alpha=0.6, s=30, zorder=3,
                        color=["steelblue", "green", "orange"][j % 3])
    axes[1].set_ylabel("HE median du run (deg)")
    axes[1].set_title("Erreur angulaire (HE)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Performance de navigation vs nombre de passages d'apprentissage",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(fig_dir, "02_boxplots_xte_he.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")

    # figure 3 : resume KC + XTE
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    kc_mean = [(training_results[p]["kc_used_L"] + training_results[p]["kc_used_R"]) / 2 * 100
               for p in passages]
    ax1.plot(passages, kc_mean, "o-", color="red", markersize=10, linewidth=2,
             label="KC utilises (moyenne L/R)")
    ax1.fill_between(passages,
                     [training_results[p]["kc_used_L"] * 100 for p in passages],
                     [training_results[p]["kc_used_R"] * 100 for p in passages],
                     alpha=0.2, color="red", label="Ecart L/R")
    ax1.axhline(85, color="darkred", ls="--", alpha=0.5, label="Seuil critique")
    ax1.set_ylabel("KC utilises (%)")
    ax1.set_title("Saturation memoire")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    xte_medians = []
    xte_mads = []
    for p in passages:
        df = nav_results.get(p)
        if df is not None:
            ok = df[df["success"] == True]
            xte_medians.append(ok["xte_median"].median())
            xte_mads.append(ok["xte_median"].mad())
        else:
            xte_medians.append(np.nan)
            xte_mads.append(np.nan)

    ax2.errorbar(passages, xte_medians, yerr=xte_mads,
                 fmt="s-", color="steelblue", markersize=10, linewidth=2,
                 capsize=5, label="XTE median +/- MAD")
    ax2.set_xlabel("Nombre de passages d'apprentissage")
    ax2.set_ylabel("XTE median (m)")
    ax2.set_title("Performance de navigation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(fig_dir, "03_summary_kc_vs_xte.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")

    # tableau recapitulatif
    summary_rows = []
    for p in passages:
        df = nav_results.get(p)
        if df is not None:
            ok = df[df["success"] == True]
            n_ok = len(ok)
            n_total = len(df)
        else:
            n_ok = n_total = 0
            ok = pd.DataFrame()

        summary_rows.append({
            "passages": p,
            "offsets": " + ".join([f"{PARALLEL_OFFSETS[j]:+.1f}m" for j in range(p)]),
            "kc_used_L_%": f"{training_results[p]['kc_used_L']*100:.1f}",
            "kc_used_R_%": f"{training_results[p]['kc_used_R']*100:.1f}",
            "kc_free_L_%": f"{training_results[p]['kc_free_L']*100:.1f}",
            "kc_free_R_%": f"{training_results[p]['kc_free_R']*100:.1f}",
            "success_rate": f"{n_ok}/{n_total}",
            "xte_median": f"{ok['xte_median'].median():.2f}" if n_ok > 0 else "N/A",
            "xte_mad": f"{ok['xte_mad'].median():.2f}" if n_ok > 0 else "N/A",
            "he_median": f"{ok['he_median'].median():.1f}" if n_ok > 0 else "N/A",
            "he_mad": f"{ok['he_mad'].median():.1f}" if n_ok > 0 else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(BASE_DIR, "summary_parallel_learning.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  RESUME :")
    for row in summary_rows:
        print(f"\n  {row['passages']} passage(s) [{row['offsets']}]:")
        print(f"    KC utilises :  L={row['kc_used_L_%']}%  R={row['kc_used_R_%']}%  "
              f"(libres: L={row['kc_free_L_%']}%  R={row['kc_free_R_%']}%)")
        print(f"    Succes :       {row['success_rate']}")
        print(f"    XTE :          {row['xte_median']} +/- {row['xte_mad']} m")
        print(f"    HE :           {row['he_median']} +/- {row['he_mad']} deg")

    print(f"\n  Tableau -> {summary_path}")
    print(f"  Figures -> {fig_dir}/")

    return summary_df


# ---- Main ----

def main():
    print("---- Experience : Apprentissage multi-trajets paralleles ----")
    print(f"  Offsets : {PARALLEL_OFFSETS}")
    print(f"  Runs de navigation par condition : {N_RUNS}")
    print(f"  Dossier de sortie : {BASE_DIR}")
    print()

    cfg = ant_pip.CFG()
    os.makedirs(BASE_DIR, exist_ok=True)

    t0 = time.time()

    # 1. capturer les images depuis les routes paralleles
    image_dirs = capture_parallel_routes(cfg)

    # 2. entrainer le MB de facon incrementale
    training_results = incremental_training(image_dirs, cfg)

    # 3. tester la navigation pour chaque niveau
    nav_results = run_navigation_tests(training_results, cfg)

    # 4. analyser et tracer
    summary = plot_results(training_results, nav_results)

    dt = time.time() - t0
    print(f"\nExperience terminee en {dt/60:.0f} minutes")
    print(f"Resultats dans : {BASE_DIR}")


if __name__ == "__main__":
    main()
