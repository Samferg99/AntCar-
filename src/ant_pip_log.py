#!/usr/bin/env python3
"""
AntCar pipeline - CARLA integration (version avec logging XTE/HE)
Identique a ant_pip.py mais avec calcul en temps reel du Cross-Track Error
et du Heading Error pendant la navigation dynamique.

Pipeline en 5 phases :
  1) Capture route apprentissage (fisheye + CSV)
  2) Augmentation + entrainement MB1/MB2
  3) Capture route de test
  4) Test offline (unfamiliarity, XTE, HE)
  5) Navigation dynamique (boucle fermée) + logging XTE/HE

Usage:
  python ant_pip_log.py --mode offline
  python ant_pip_log.py --mode navigate --model_dir ./antcar_out
  python ant_pip_log.py --mode all
"""

import os, csv, time, math, queue, argparse, sys, random
import shutil

import numpy as np
import cv2
from skimage import filters
import imutils
import pandas as pd
import matplotlib.pyplot as plt

import carla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory
import antcar_sim


# ---- Configuration ----

class CFG:
    # CARLA
    HOST          = "localhost"
    PORT          = 2000
    TIMEOUT_S     = 10.0
    SYNC          = True
    FPS           = 20.0

    # Route (Town05)
    START_SPAWN_IDX     = 152
    N_POINTS_LEARN      = 300
    N_POINTS_TEST       = 250
    STEP_M              = 1.0
    TEST_LATERAL_OFFSET = -1.0   # decalage lateral route test (m)

    # Cameras fisheye
    FACE_RES      = 512
    FACE_FOV_DEG  = 90.0
    FISH_RES      = 512
    THETA_MAX_DEG = 110.0

    # Parametres MB (cf. Gattaux et al. 2025)
    # kappa adapté pour CARLA : images plus espacees qu'antcar reel (38Hz)
    # donc on reduit kappa de 0.01 a 0.005 pour eviter la saturation
    KC_NB         = 15000
    KC_TO_PN_SYN  = 4
    KC_NORM       = 0.005
    SEED_MB1      = 42
    SEED_MB2      = 99
    OSCIL_MAX     = 45     # degres
    OSCIL_STEP    = 5      # degres
    VISION_RES    = 32     # 32x32 -> ~800 PNs
    VISION_SIGMA  = 3
    BATCH_SIZE    = 20

    # Navigation dynamique
    NAV_SPEED_KMH       = 6
    NAV_SPEED_MIN_KMH   = 2       # vitesse plancher en virage
    NAV_LAMBDA_SLOW     = 0.92
    NAV_LAMBDA_FAST     = 0.996
    NAV_SPEED_ALPHA     = 0.25
    NAV_STOP_LAMBDA     = 0.75    # frein d'urgence
    NAV_STOP_PATIENCE   = 40
    NAV_KP              = 25.0
    NAV_ALPHA           = 0.15    # filtre expo, lisse le bruit KC (~7 ticks)
    NAV_DEADZONE        = 0.005
    NAV_MIN_BOTH        = 0.001
    NAV_MIN_REL         = 0.005
    NAV_CLIP_SOFT       = 0.50
    NAV_STEER_CLIP      = 0.75
    # ralentissement virage
    NAV_STEER_SLOW_THRESH = 0.10
    NAV_STEER_SLOW_MAX   = 0.40
    NAV_MAX_STEPS       = 2000
    NAV_TICK_INTERVAL   = 0.2
    WARMUP_TICKS        = 2
    Z_LIFT              = 0.2

    OUTDIR = "./antcar_out"


# ---- Utilitaires CARLA ----

def connect_carla(cfg):
    client = carla.Client(cfg.HOST, cfg.PORT)
    client.set_timeout(cfg.TIMEOUT_S)
    world = client.get_world()
    cmap = world.get_map()
    if cfg.SYNC:
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = 1.0 / cfg.FPS
        world.apply_settings(s)
    print(f"[CARLA] Connecté - carte : {cmap.name}")
    return client, world, cmap


def restore_carla(world, original_settings):
    try:
        world.apply_settings(original_settings)
    except Exception:
        pass


def spawn_vehicle(world, cmap, cfg):
    bp_lib = world.get_blueprint_library()
    candidates = bp_lib.filter("vehicle.*model3*")
    veh_bp = candidates[0] if candidates else bp_lib.filter("vehicle.*")[0]
    sps = cmap.get_spawn_points()
    preferred = sps[cfg.START_SPAWN_IDX:cfg.START_SPAWN_IDX + 1] or []
    trylist = preferred + sps

    for sp in trylist[:120]:
        tf = carla.Transform(
            carla.Location(sp.location.x, sp.location.y, sp.location.z + cfg.Z_LIFT),
            sp.rotation
        )
        veh = world.try_spawn_actor(veh_bp, tf)
        if veh is not None:
            veh.set_simulate_physics(False)
            print(f"[CARLA] Véhicule spawné : {veh.type_id}")
            return veh, tf
    raise RuntimeError("Impossible de spawner le véhicule.")


def build_route(cmap, start_tf, n, step, lateral_offset=0.0):
    """Genere n poses espacees de step metres le long des waypoints CARLA."""
    w = cmap.get_waypoint(start_tf.location, project_to_road=True,
                          lane_type=carla.LaneType.Driving)
    if w is None:
        raise RuntimeError("Aucun waypoint Driving trouvé au spawn.")

    route = []
    while w and len(route) < n:
        tf = w.transform
        if lateral_offset != 0.0:
            yaw_r = math.radians(tf.rotation.yaw)
            tf = carla.Transform(
                carla.Location(
                    tf.location.x - lateral_offset * math.sin(yaw_r),
                    tf.location.y + lateral_offset * math.cos(yaw_r),
                    tf.location.z
                ),
                tf.rotation
            )
        route.append(tf)
        nxt = w.next(step)
        w = nxt[0] if nxt else None

    print(f"[Route] {len(route)} poses (offset={lateral_offset}m)")
    return route


# ---- Rig Fisheye (6 cameras -> panorama equidistant) ----

FACE_ROTS = {
    0: carla.Rotation(pitch=0, yaw=0, roll=0),      # avant
    1: carla.Rotation(pitch=0, yaw=180, roll=0),     # arriere
    2: carla.Rotation(pitch=0, yaw=90, roll=0),      # droite
    3: carla.Rotation(pitch=0, yaw=-90, roll=0),     # gauche
    4: carla.Rotation(pitch=90, yaw=0, roll=0),      # dessus
    5: carla.Rotation(pitch=-90, yaw=0, roll=0),     # dessous
}


def precompute_fisheye_maps(fish_res, face_res, face_fov_deg=90.0, theta_max_deg=110.0):
    """Precalcule les maps de remapping pour la projection equidistante."""
    cx = cy = (fish_res - 1) / 2.0
    U, V = np.meshgrid(np.arange(fish_res), np.arange(fish_res))
    x = (U - cx).astype(np.float64)
    y = (cy - V).astype(np.float64)
    r = np.sqrt(x**2 + y**2)

    theta_max = np.deg2rad(theta_max_deg)
    r_max = fish_res * 0.5
    f_fish = r_max / theta_max
    theta = r / f_fish
    valid = theta <= theta_max
    phi = np.arctan2(y, x)

    sin_t = np.sin(theta)
    dx = sin_t * np.cos(phi)
    dy = sin_t * np.sin(phi)
    dz = np.cos(theta)

    adx, ady, adz = np.abs(dx), np.abs(dy), np.abs(dz)
    face = np.full((fish_res, fish_res), -1, dtype=np.int32)
    face[(adx >= ady) & (adx >= adz) & (dx >= 0)] = 0
    face[(adx >= ady) & (adx >= adz) & (dx < 0)] = 1
    face[(ady > adx) & (ady >= adz) & (dy >= 0)] = 2
    face[(ady > adx) & (ady >= adz) & (dy < 0)] = 3
    face[(adz > adx) & (adz > ady) & (dz >= 0)] = 4
    face[(adz > adx) & (adz > ady) & (dz < 0)] = 5

    fov = np.deg2rad(face_fov_deg)
    fx = face_res / (2.0 * np.tan(fov / 2.0))
    cxf = cyf = (face_res - 1) / 2.0

    def veh_to_cam(fid, dx_, dy_, dz_):
        if fid == 0: return  dx_,  dy_,  dz_
        if fid == 1: return -dx_, -dy_,  dz_
        if fid == 2: return  dy_, -dx_,  dz_
        if fid == 3: return -dy_,  dx_,  dz_
        if fid == 4: return  dz_,  dy_, -dx_
        if fid == 5: return -dz_,  dy_,  dx_

    maps = {}
    for fid in range(6):
        sel = valid & (face == fid)
        xc, yc, zc = veh_to_cam(fid, dx, dy, dz)
        xc = np.where(sel, xc, 1.0)
        yc = np.where(sel, yc, 0.0)
        zc = np.where(sel, zc, 0.0)
        u = fx * (yc / xc) + cxf
        v = fx * (-zc / xc) + cyf
        ok = sel & (xc > 1e-6) & (u >= 0) & (u < face_res) & (v >= 0) & (v < face_res)
        mapx = np.full((fish_res, fish_res), -1, dtype=np.float32)
        mapy = np.full((fish_res, fish_res), -1, dtype=np.float32)
        mask = np.zeros((fish_res, fish_res), dtype=bool)
        mapx[ok] = u[ok].astype(np.float32)
        mapy[ok] = v[ok].astype(np.float32)
        mask[ok] = True
        maps[fid] = {"mapx": mapx, "mapy": mapy, "mask": mask}
    return maps


def composite_fisheye(faces, maps, fish_res):
    out = np.zeros((fish_res, fish_res, 3), dtype=np.uint8)
    for fid, img in faces.items():
        m = maps[fid]
        warped = cv2.remap(img, m["mapx"], m["mapy"],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[m["mask"]] = warped[m["mask"]]
    return out


def carla_to_bgr(img):
    arr = np.frombuffer(img.raw_data, dtype=np.uint8)
    return arr.reshape((img.height, img.width, 4))[:, :, :3].copy()


class FisheyeRig:
    """Rig de 6 cameras CARLA + conversion fisheye panoramique."""

    def __init__(self, world, vehicle, cfg):
        self._cfg = cfg
        self._world = world
        self._vehicle = vehicle
        self._sensors = {}
        self._queues = {}
        self._actor_list = []
        self._maps = precompute_fisheye_maps(cfg.FISH_RES, cfg.FACE_RES,
                                              cfg.FACE_FOV_DEG, cfg.THETA_MAX_DEG)
        self._attach()

    def _attach(self):
        bp_lib = self._world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self._cfg.FACE_RES))
        cam_bp.set_attribute("image_size_y", str(self._cfg.FACE_RES))
        cam_bp.set_attribute("fov", str(self._cfg.FACE_FOV_DEG))
        cam_bp.set_attribute("sensor_tick", "0.0")

        for fid, rot in FACE_ROTS.items():
            tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0), rot)
            s = self._world.spawn_actor(cam_bp, tf, attach_to=self._vehicle)
            q = queue.Queue()
            s.listen(lambda img, _q=q: _q.put(img))
            self._sensors[fid] = s
            self._queues[fid] = q
            self._actor_list.append(s)
        print("[Rig] 6 caméras attachées.")

    def drain(self):
        for q in self._queues.values():
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

    def capture(self):
        """Tick + recupere les 6 faces -> image fisheye BGR."""
        frame_id = self._world.tick()
        faces = {}
        for fid, q in self._queues.items():
            while True:
                img = q.get(timeout=3.0)
                if img.frame == frame_id:
                    faces[fid] = carla_to_bgr(img)
                    break
        return composite_fisheye(faces, self._maps, self._cfg.FISH_RES)

    def destroy(self):
        for s in self._sensors.values():
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass


# ---- Phase 1 : capture statique ----

def capture_static_route(world, vehicle, rig, route, outdir, cfg):
    """
    Teleporte le vehicule a chaque pose et capture l'image fisheye.
    Format fichier: {idx}_{x}_{y}_{yaw_deg}.jpg
    Le CSV est mis dans le dossier parent pour pas perturber read_raw_img_folder.
    """
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(os.path.dirname(outdir.rstrip("/\\")),
                             os.path.basename(outdir.rstrip("/\\")) + "_poses.csv")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["idx", "x", "y", "z", "pitch", "yaw", "roll", "filename"])

    t0 = time.time()
    for i, tf in enumerate(route):
        vehicle.set_transform(carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + cfg.Z_LIFT),
            tf.rotation
        ))
        for _ in range(cfg.WARMUP_TICKS):
            world.tick()
        rig.drain()

        fish = rig.capture()

        x = tf.location.x
        y_pos = tf.location.y
        yaw = tf.rotation.yaw
        fname = f"{i}_{x:.4f}_{y_pos:.4f}_{yaw:.4f}.jpg"
        cv2.imwrite(os.path.join(outdir, fname), fish)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([i, x, y_pos, tf.location.z,
                                     tf.rotation.pitch, yaw, tf.rotation.roll, fname])

        if i % 20 == 0:
            print(f"  [Capture] {i+1}/{len(route)}", end="\r")

    dt = time.time() - t0
    print(f"\n  [Capture] {len(route)} images en {dt:.1f}s -> {outdir}")
    return outdir


# ---- Phase 2 : entrainement MB ----

class SimpleArgs:
    """Faux namespace pour AntCarSim (qui attend des args de argparse)."""
    def __init__(self, learn_src, test_src, output, cfg):
        self.inputsroutelearn = learn_src
        self.inputtest = test_src
        self.output = output
        self.inputsplacelearn = None
        self.paramsnet = (
            f"{cfg.KC_NB},{cfg.KC_TO_PN_SYN},{cfg.SEED_MB1},{cfg.KC_NORM},"
            f"{cfg.KC_NB},{cfg.KC_TO_PN_SYN},{cfg.SEED_MB2},{cfg.KC_NORM}"
        )
        self.oscillearn = f"{cfg.OSCIL_MAX},{cfg.OSCIL_STEP}"
        self.paramsvision = f"{cfg.VISION_RES},{cfg.VISION_SIGMA}"
        self.save = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _purge_non_images(folder):
    """
    Deplace les fichiers non-image hors du dossier.
    Necessaire car antcar_sim.read_raw_img_folder scanne tout le dossier
    et crash sur les .csv ou sous-dossiers.
    """
    if not os.path.isdir(folder):
        return

    parent = os.path.dirname(os.path.abspath(folder))
    basename = os.path.basename(os.path.abspath(folder))
    meta_dir = os.path.join(parent, f"_meta_{basename}")
    moved = []

    for entry in os.listdir(folder):
        fpath = os.path.join(folder, entry)
        ext = os.path.splitext(entry)[1].lower()
        if os.path.isdir(fpath) or ext not in IMAGE_EXTS:
            os.makedirs(meta_dir, exist_ok=True)
            dest = os.path.join(meta_dir, entry)
            if os.path.exists(dest):
                shutil.rmtree(dest) if os.path.isdir(dest) else os.remove(dest)
            shutil.move(fpath, dest)
            moved.append(entry)

    if moved:
        print(f"  [purge] {len(moved)} fichier(s) déplacé(s) -> {meta_dir}")


def load_model(learn_src, outdir, cfg):
    """Charge les poids MB depuis un entrainement precedent."""
    weights_path = os.path.join(outdir, "mb_weights.npz")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Poids introuvables : {weights_path}\n"
            "Lancez d'abord --mode offline pour entrainer."
        )
    args = SimpleArgs(learn_src, learn_src, outdir, cfg)
    sim = antcar_sim.AntCarSim(args)
    data = np.load(weights_path)
    sim._mbs[0].load_memory(data["W_mb1"])
    sim._mbs[1].load_memory(data["W_mb2"])
    print(f"[load] Poids chargés : {weights_path}")
    print(f"  MB1 KC actifs : {(sim._mbs[0].W_KCtoMBON > 0).sum()} / {len(sim._mbs[0].W_KCtoMBON)}")
    print(f"  MB2 KC actifs : {(sim._mbs[1].W_KCtoMBON > 0).sum()} / {len(sim._mbs[1].W_KCtoMBON)}")
    return sim


def run_training(learn_src, test_src, outdir, cfg):
    """Augmentation + entrainement MB1 & MB2 + test offline."""
    print("\n---- Phase 2 : Entrainement Mushroom Bodies ----")

    args = SimpleArgs(learn_src, test_src, outdir, cfg)
    sim = antcar_sim.AntCarSim(args)

    step = cfg.OSCIL_STEP
    maxo = cfg.OSCIL_MAX

    arr_pos = np.arange(0, maxo + step, step, dtype="int")
    arr_neg = np.arange(0, -(maxo + step), -step, dtype="int")
    oscil_test = np.arange(-180, 180 + step, step, dtype="int")

    src_mb1 = os.path.join(outdir, "images/learning1/augmented/")
    src_mb2 = os.path.join(outdir, "images/learning2/augmented/")
    src_test_aug = os.path.join(outdir, "images/test/augmented/")

    _purge_non_images(learn_src)
    _purge_non_images(test_src)

    _, traj_train = sim.read_raw_img_folder(learn_src)
    # deduplication : une seule entree par position (x,y)
    _, uidx = np.unique(np.round(traj_train[:,1:3], 2), axis=0, return_index=True)
    traj_train = traj_train[np.sort(uidx)]
    sim.save_csv(traj_train, "traj_train.csv")

    print("[Train] Augmentation...")
    sim.augment_dataset(learn_src, src_mb1, arr_pos, batch_size=cfg.BATCH_SIZE)
    sim.augment_dataset(learn_src, src_mb2, arr_neg, batch_size=cfg.BATCH_SIZE)
    sim.augment_dataset(test_src, src_test_aug, oscil_test, batch_size=cfg.BATCH_SIZE)

    print("[Train] MB1 (oscil positive)...")
    mapping_mb1, mem1 = sim.train(0, src_mb1, batch_size=cfg.BATCH_SIZE)

    print("[Train] MB2 (oscil negative)...")
    mapping_mb2, mem2 = sim.train(1, src_mb2, batch_size=cfg.BATCH_SIZE)

    print("[Test] Evaluation offline...")
    # test sur images canoniques (non augmentees) pour les metriques XTE/HE
    mapping_test_canonical = sim.test(test_src, traj_train, batch_size=cfg.BATCH_SIZE)
    # scan heading complet (toutes rotations)
    mapping_test_scan = sim.test(src_test_aug, traj_train, batch_size=cfg.BATCH_SIZE * 2)
    mapping_test = mapping_test_canonical

    # sauvegarde resultats
    memories = np.vstack((mem1, mem2)).T
    mapping_learning = np.concatenate((mapping_mb1, mapping_mb2), axis=1)
    sim.save_csv(mapping_learning, "mapping_learning.csv")
    sim.save_csv(memories, "memories_learning.csv")
    sim.save_csv(mapping_test_canonical, "mapping_test_canonical.csv")
    sim.save_csv(mapping_test_scan, "mapping_test_scan.csv")
    sim.save_csv(mapping_test, "mapping_test.csv")

    _plot_results(mapping_test, outdir)

    weights_path = os.path.join(outdir, "mb_weights.npz")
    np.savez(weights_path, W_mb1=sim._mbs[0].W_KCtoMBON, W_mb2=sim._mbs[1].W_KCtoMBON)
    print(f"[Train] Poids sauvegardes -> {weights_path}")
    return sim


def _plot_results(mapping_test, outdir):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    idx = np.arange(len(mapping_test))

    axes[0].plot(idx, mapping_test[:, 4], label="Unfamiliarity MB1", color="steelblue")
    axes[0].plot(idx, mapping_test[:, 5], label="Unfamiliarity MB2", color="tomato")
    axes[0].set_ylabel("Unfamiliarity")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(idx, mapping_test[:, 6], color="darkorange", label="XTE (m)")
    axes[1].axhline(0, color="k", linestyle="--", lw=0.8)
    axes[1].set_ylabel("Cross-Track Error (m)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(idx, mapping_test[:, 7], color="mediumseagreen", label="HE (deg)")
    axes[2].axhline(0, color="k", linestyle="--", lw=0.8)
    axes[2].set_ylabel("Heading Error (deg)")
    axes[2].set_xlabel("Index image")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle("AntCar - Résultats offline", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "results_offline.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {path}")


# ---- Phase 5 : navigation dynamique ----

class AntNavigator:
    """
    Navigation online inspiree des fourmis.
    MB1 apprend les oscillations positives, MB2 les negatives.
    Le steer est proportionnel a unfam(MB1) - unfam(MB2).
    Ref: Gattaux et al. 2025, IEEE AICAS
    """

    def __init__(self, sim, cfg):
        self._sim = sim
        self._cfg = cfg
        self._steer_prev = 0.0

    def _image_to_pn(self, fish_bgr):
        pn, _ = self._sim.create_pn(fish_bgr)
        return pn

    def compute_steer(self, fish_bgr, speed_kmh=None):
        """Calcule la commande de direction a partir de l'image fisheye.
        Retourne (steer, u1, u2, lam_diff, lam_max, lam_min)."""
        pn = self._image_to_pn(fish_bgr)

        u1 = float(self._sim._mbs[0].get_unfamiliarity(pn))
        u2 = float(self._sim._mbs[1].get_unfamiliarity(pn))

        lam1 = 1.0 - u1
        lam2 = 1.0 - u2
        lam_diff = lam1 - lam2
        lam_max = max(lam1, lam2)
        lam_min = min(lam1, lam2)

        # si les deux MB sont tres familiers, pas besoin de corriger
        if u1 < self._cfg.NAV_MIN_BOTH and u2 < self._cfg.NAV_MIN_BOTH:
            self._steer_prev = 0.0
            return 0.0, u1, u2, lam_diff, lam_max, lam_min

        delta_u = u1 - u2

        if min(u1, u2) >= self._cfg.NAV_MIN_REL:
            denom = u1 + u2
            delta_norm = delta_u / denom if denom > 1e-9 else 0.0
            if abs(delta_norm) < self._cfg.NAV_DEADZONE:
                delta_norm = 0.0
            steer_raw = float(np.clip(self._cfg.NAV_KP * delta_norm,
                                       -self._cfg.NAV_STEER_CLIP,
                                        self._cfg.NAV_STEER_CLIP))
        else:
            raw_abs = self._cfg.NAV_KP * delta_u
            steer_raw = float(np.clip(raw_abs,
                                       -self._cfg.NAV_CLIP_SOFT,
                                        self._cfg.NAV_CLIP_SOFT))

        # filtre exponentiel + decay si le vehicule est immobile
        alpha = self._cfg.NAV_ALPHA
        if speed_kmh is not None and speed_kmh < 0.3:
            decay = 0.8
            self._steer_prev = alpha * steer_raw + decay * (1.0 - alpha) * self._steer_prev
        else:
            self._steer_prev = alpha * steer_raw + (1.0 - alpha) * self._steer_prev

        return self._steer_prev, u1, u2, lam_diff, lam_max, lam_min


def _kmh_to_throttle(kmh, max_kmh=15.0):
    """Conversion lineaire km/h -> throttle CARLA (Model 3)."""
    return float(np.clip(kmh / max_kmh, 0.10, 1.0))


def run_navigation(world, vehicle, rig, navigator, route, cfg, outdir):
    """Boucle de navigation dynamique avec controle de vitesse adaptatif."""
    print("\n---- Phase 5 : Navigation dynamique ----")

    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "nav_log.csv")

    vehicle.set_simulate_physics(True)

    # teleporte au debut de la route
    vehicle.set_transform(carla.Transform(
        carla.Location(route[0].location.x, route[0].location.y,
                       route[0].location.z + cfg.Z_LIFT),
        route[0].rotation
    ))
    for _ in range(10):
        world.tick()
    rig.drain()

    log_rows = []
    stop_count = 0

    # waypoints appris pour le calcul XTE/HE en temps reel
    route_array = np.array([
        [tf.location.x, tf.location.y, tf.rotation.yaw]
        for tf in route
    ])

    print(f"  Vitesse : {cfg.NAV_SPEED_MIN_KMH}-{cfg.NAV_SPEED_KMH} km/h | "
          f"Kp={cfg.NAV_KP} alpha={cfg.NAV_ALPHA} clip={cfg.NAV_STEER_CLIP}")

    for step in range(cfg.NAV_MAX_STEPS):
        vel = vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # signal MB
        fish = rig.capture()
        steer, u1, u2, lam_diff, lam_max, lam_min = navigator.compute_steer(
            fish, speed_kmh=speed_kmh)

        # gestion arret d'urgence
        if lam_min < cfg.NAV_STOP_LAMBDA:
            stop_count += 1
        else:
            stop_count = 0

        # vitesse cible basee sur lambda_min (familiarite)
        span = max(1e-6, cfg.NAV_LAMBDA_FAST - cfg.NAV_LAMBDA_SLOW)
        ratio_lam = float(np.clip((lam_min - cfg.NAV_LAMBDA_SLOW) / span, 0.0, 1.0))
        target_lam = cfg.NAV_SPEED_MIN_KMH + (cfg.NAV_SPEED_KMH - cfg.NAV_SPEED_MIN_KMH) * ratio_lam

        # on ralentit aussi dans les virages (basé sur |steer|)
        abs_steer = abs(steer)
        if abs_steer <= cfg.NAV_STEER_SLOW_THRESH:
            target_steer = cfg.NAV_SPEED_KMH
        elif abs_steer >= cfg.NAV_STEER_SLOW_MAX:
            target_steer = cfg.NAV_SPEED_MIN_KMH
        else:
            span_s = max(1e-6, cfg.NAV_STEER_SLOW_MAX - cfg.NAV_STEER_SLOW_THRESH)
            ratio_s = (abs_steer - cfg.NAV_STEER_SLOW_THRESH) / span_s
            target_steer = cfg.NAV_SPEED_KMH - (cfg.NAV_SPEED_KMH - cfg.NAV_SPEED_MIN_KMH) * ratio_s

        target_kmh = min(target_lam, target_steer)

        # regulateur vitesse
        speed_error = target_kmh - speed_kmh

        if stop_count >= cfg.NAV_STOP_PATIENCE:
            throttle = 0.0
            brake = 1.0
        elif speed_error > 1.0:
            throttle = _kmh_to_throttle(target_kmh)
            brake = 0.0
        elif speed_error < -2.0:
            throttle = 0.0
            brake = float(np.clip(-speed_error * 0.02, 0.0, 0.10))
        else:
            throttle = _kmh_to_throttle(target_kmh) * 0.7
            brake = 0.0

        ctrl = carla.VehicleControl(throttle=throttle, steer=-steer, brake=brake)
        vehicle.apply_control(ctrl)
        world.tick()

        if stop_count >= cfg.NAV_STOP_PATIENCE and brake > 0.0:
            print(f"  [STOP] lam_min={lam_min:.4f} < {cfg.NAV_STOP_LAMBDA} "
                  f"pendant {cfg.NAV_STOP_PATIENCE} ticks")
            break

        tf = vehicle.get_transform()

        # calcul XTE et HE par rapport a la route apprise
        _dx = route_array[:, 0] - tf.location.x
        _dy = route_array[:, 1] - tf.location.y
        _distances = np.sqrt(_dx**2 + _dy**2)
        nearest_idx = int(np.argmin(_distances))
        nearest_wp = route_array[nearest_idx]

        # XTE signé (positif = droite, negatif = gauche)
        _wp_yaw_rad = math.radians(nearest_wp[2])
        _path_dir = np.array([math.cos(_wp_yaw_rad), math.sin(_wp_yaw_rad)])
        _to_vehicle = np.array([tf.location.x - nearest_wp[0],
                                tf.location.y - nearest_wp[1]])
        _cross = _path_dir[0] * _to_vehicle[1] - _path_dir[1] * _to_vehicle[0]
        xte = float(_distances[nearest_idx]) * float(np.sign(_cross))

        # HE normalisé dans [-180, 180]
        he = tf.rotation.yaw - nearest_wp[2]
        he = (he + 180) % 360 - 180

        # detection fin de route
        if nearest_idx >= len(route) - 2:
            log_rows.append({
                "step": step, "x": round(tf.location.x, 3),
                "y": round(tf.location.y, 3), "yaw": round(tf.rotation.yaw, 2),
                "speed": round(speed_kmh, 2), "steer": round(steer, 4),
                "unfam1": round(u1, 5), "unfam2": round(u2, 5),
                "lambda_diff": round(lam_diff, 5), "lambda_max": round(lam_max, 5),
                "lambda_min": round(lam_min, 5), "target_kmh": round(target_kmh, 2),
                "throttle": round(throttle, 3), "xte": round(xte, 4),
                "he": round(he, 2), "nearest_idx": nearest_idx,
            })
            print(f"  [FIN] Route completée step {step} "
                  f"(wp {nearest_idx}/{len(route)-1})  "
                  f"XTE={xte:+.3f}m  HE={he:+.1f}deg")
            break

        log_rows.append({
            "step": step,
            "x": round(tf.location.x, 3),
            "y": round(tf.location.y, 3),
            "yaw": round(tf.rotation.yaw, 2),
            "speed": round(speed_kmh, 2),
            "steer": round(steer, 4),
            "unfam1": round(u1, 5),
            "unfam2": round(u2, 5),
            "lambda_diff": round(lam_diff, 5),
            "lambda_max": round(lam_max, 5),
            "lambda_min": round(lam_min, 5),
            "target_kmh": round(target_kmh, 2),
            "throttle": round(throttle, 3),
            "xte": round(xte, 4),
            "he": round(he, 2),
            "nearest_idx": nearest_idx,
        })

        if step % 50 == 0:
            print(f"  step={step:4d}  spd={speed_kmh:5.1f}/{target_kmh:4.1f}km/h  "
                  f"steer={steer:+.3f}  lam_min={lam_min:.3f}  "
                  f"xte={xte:+.3f}m  he={he:+.1f}deg")

    # freiner
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    world.tick()

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    _plot_nav_log(log_rows, outdir)
    print(f"[Nav] Log : {log_path}")


def _plot_nav_log(log_rows, outdir):
    df = pd.DataFrame(log_rows)
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(df["step"], df["steer"], color="steelblue")
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("Steer")
    axes[0].grid(True)

    axes[1].plot(df["step"], df["unfam1"], label="MB1", color="tomato")
    axes[1].plot(df["step"], df["unfam2"], label="MB2", color="mediumseagreen")
    axes[1].set_ylabel("Unfamiliarity")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["step"], df["speed"], color="darkorange")
    axes[2].set_ylabel("Vitesse (km/h)")
    axes[2].grid(True)

    # XTE
    axes[3].plot(df["step"], df["xte"], color="purple")
    axes[3].axhline(0, color="k", lw=0.8, ls="--")
    xte_abs = df["xte"].abs()
    med = xte_abs.median()
    mad = (xte_abs - med).abs().median()
    axes[3].set_ylabel("XTE (m)")
    axes[3].set_title(f"Cross-Track Error  -  mediane |XTE| = {med:.3f} +/- {mad:.3f} m",
                      fontsize=10, loc="left")
    axes[3].grid(True)

    # HE
    axes[4].plot(df["step"], df["he"], color="teal")
    axes[4].axhline(0, color="k", lw=0.8, ls="--")
    he_abs = df["he"].abs()
    med_he = he_abs.median()
    mad_he = (he_abs - med_he).abs().median()
    axes[4].set_ylabel("HE (deg)")
    axes[4].set_title(f"Heading Error  -  mediane |HE| = {med_he:.1f} +/- {mad_he:.1f} deg",
                      fontsize=10, loc="left")
    axes[4].set_xlabel("Step")
    axes[4].grid(True)

    fig.suptitle("AntCar - Navigation dynamique", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "results_navigation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Plot] {path}")


# ---- Main ----

def parse_args():
    p = argparse.ArgumentParser(description="AntCar pipeline CARLA")
    p.add_argument("--mode", choices=["offline", "navigate", "all"],
                   default="offline",
                   help="offline=phases 1-4 | navigate=phase 5 | all=tout")
    p.add_argument("--model_dir", default=CFG.OUTDIR,
                   help="Dossier de sortie / chargement modele")
    p.add_argument("--lateral_offset", type=float, default=CFG.TEST_LATERAL_OFFSET,
                   help="Decalage lateral (m) route de test")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG()
    cfg.OUTDIR = args.model_dir

    learn_dir = os.path.join(cfg.OUTDIR, "images/raw_learn/")
    test_dir = os.path.join(cfg.OUTDIR, "images/raw_test/")
    nav_dir = os.path.join(cfg.OUTDIR, "navigation/")
    os.makedirs(cfg.OUTDIR, exist_ok=True)

    # phases offline (1 a 4)
    if args.mode in ("offline", "all"):
        print("\n---- Phase 1 : Capture des routes ----")

        client, world, cmap = connect_carla(cfg)
        orig_settings = world.get_settings()

        try:
            vehicle, spawn_tf = spawn_vehicle(world, cmap, cfg)

            route_learn = build_route(cmap, spawn_tf, cfg.N_POINTS_LEARN,
                                       cfg.STEP_M, lateral_offset=0.0)
            route_test = build_route(cmap, spawn_tf, cfg.N_POINTS_TEST,
                                      cfg.STEP_M, lateral_offset=args.lateral_offset)

            rig = FisheyeRig(world, vehicle, cfg)

            print("\n[1a] Capture route apprentissage...")
            capture_static_route(world, vehicle, rig, route_learn, learn_dir, cfg)

            print("\n[1b] Capture route test...")
            capture_static_route(world, vehicle, rig, route_test, test_dir, cfg)

        finally:
            if args.mode == "offline":
                rig.destroy()
                try: vehicle.destroy()
                except Exception: pass
                restore_carla(world, orig_settings)

        sim = run_training(learn_dir, test_dir, cfg.OUTDIR, cfg)

        if args.mode == "offline":
            print(f"\nPipeline offline terminé. Resultats : {cfg.OUTDIR}")
            return

    # phase 5 : navigation
    if args.mode in ("navigate", "all"):
        print("\n---- Phase 5 : Navigation ----")

        if args.mode == "navigate":
            learn_dir = os.path.join(cfg.OUTDIR, "images/raw_learn/")
            client, world, cmap = connect_carla(cfg)
            orig_settings = world.get_settings()
            vehicle, spawn_tf = spawn_vehicle(world, cmap, cfg)
            rig = FisheyeRig(world, vehicle, cfg)
            sim = load_model(learn_dir, cfg.OUTDIR, cfg)
            route_learn = build_route(cmap, spawn_tf, cfg.N_POINTS_LEARN, cfg.STEP_M, 0.0)

        navigator = AntNavigator(sim, cfg)

        try:
            run_navigation(world, vehicle, rig, navigator, route_learn, cfg, nav_dir)
        finally:
            rig.destroy()
            try: vehicle.destroy()
            except Exception: pass
            restore_carla(world, orig_settings)

        print(f"\nNavigation terminée. Log : {nav_dir}")


if __name__ == "__main__":
    main()
