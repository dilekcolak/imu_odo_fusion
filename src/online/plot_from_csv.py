# -*- coding: utf-8 -*-
import os, argparse, csv, math, datetime
import numpy as np
import matplotlib.pyplot as plt

REQ_COLS = ["t",
    "gt_x","gt_y","gt_yaw",
    "imu_gz","odo_v","odo_w",
    "naive_x","naive_y","naive_yaw",
    "ekf_x","ekf_y","ekf_yaw","ekf_bg","ekf_v"
]

def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for c in REQ_COLS:
            if c not in cols:
                raise ValueError(f"CSV kolon eksik: {c}")
        rows = [r for r in reader]
    def col(name): return np.array([float(r[name]) for r in rows], dtype=float)
    data = {k: col(k) for k in REQ_COLS}
    return data

def rmse(ax, ay, bx, by):
    return math.sqrt(np.mean((ax-bx)**2 + (ay-by)**2))

def default_csv_path():
    # Bu dosyanın konumundan proje kökünü bul → data/runs/run_latest.csv
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "data", "runs", "run_latest.csv")

def default_figs_dir():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "data", "figs")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV yolu (vermezsen run_latest.csv kullanılır)")
    ap.add_argument("--latest", action="store_true", help="Zorla run_latest.csv kullan")
    ap.add_argument("--save_dir", default=None, help="PNG'lerin kaydedileceği klasör (varsayılan: ../data/figs)")
    ap.add_argument("--show", action="store_true", help="Grafikleri ekranda göster")
    args = ap.parse_args()

    # CSV yolunu belirle
    csv_path = args.csv
    if args.latest or not csv_path:
        csv_path = default_csv_path()

    if not os.path.exists(csv_path):
        runs_dir = os.path.dirname(csv_path)
        raise SystemExit(
            f"CSV bulunamadı: {csv_path}\n"
            f"- Önce live_stream.py çalıştırıp 'c' veya 'q' ile kaydedin.\n"
            f"- Beklenen klasör: {runs_dir}"
        )

    # Kayıt klasörü
    save_dir = args.save_dir or default_figs_dir()
    os.makedirs(save_dir, exist_ok=True)

    data = read_csv(csv_path)
    t = data["t"]
    gt_x, gt_y = data["gt_x"], data["gt_y"]
    nv_x, nv_y = data["naive_x"], data["naive_y"]
    ek_x, ek_y = data["ekf_x"], data["ekf_y"]

    # Hata serileri
    err_naive = np.hypot(nv_x - gt_x, nv_y - gt_y)
    err_ekf   = np.hypot(ek_x - gt_x, ek_y - gt_y)

    # Metrikler
    rmse_n = rmse(nv_x, nv_y, gt_x, gt_y)
    rmse_e = rmse(ek_x, ek_y, gt_x, gt_y)
    loop_gt   = np.hypot(gt_x[-1]-gt_x[0], gt_y[-1]-gt_y[0])
    loop_nv   = np.hypot(nv_x[-1]-nv_x[0], nv_y[-1]-nv_y[0])
    loop_ek   = np.hypot(ek_x[-1]-ek_x[0], ek_y[-1]-ek_y[0])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_traj = os.path.join(save_dir, f"{base}_traj_{ts}.png")
    out_err  = os.path.join(save_dir, f"{base}_errors_{ts}.png")
    out_txt  = os.path.join(save_dir, f"{base}_summary_{ts}.txt")

    # 1) Trajectory
    plt.figure(figsize=(6,6))
    plt.plot(gt_x, gt_y, color="black", label="Ground Truth")
    plt.plot(nv_x, nv_y, "--", color="red",  label="Filtresiz Füzyon")
    plt.plot(ek_x, ek_y, "-.", color="teal", label="Kalman (IMU+Odo)")
    plt.scatter(gt_x[0], gt_y[0], c="green", marker="o", label="Başlangıç")
    plt.scatter(gt_x[-1], gt_y[-1], c="blue", marker="x", label="GT Bitiş")
    plt.scatter(nv_x[-1], nv_y[-1], c="red", marker="x", label="Naive Bitiş")
    plt.scatter(ek_x[-1], ek_y[-1], c="teal", marker="x", label="EKF Bitiş")
    plt.axis("equal"); plt.grid(True)
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.title("Yörüngeler: GT vs Naive vs EKF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_traj, dpi=150)

    # 2) Error vs time
    plt.figure(figsize=(8,3.5))
    plt.plot(t, err_naive, "--", label="Naive konum hatası [m]")
    plt.plot(t, err_ekf, "-.", label="EKF konum hatası [m]")
    plt.grid(True); plt.xlabel("Zaman [s]"); plt.ylabel("Hata [m]")
    plt.title("Konum Hatası (GT referans)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_err, dpi=150)

    # 3) Summary metin
    with open(out_txt, "w") as f:
        f.write("Özet metrikler\n")
        f.write("================\n")
        f.write(f"Örnek sayısı: {len(t)}\n")
        f.write(f"RMSE Naive [m]: {rmse_n:.3f}\n")
        f.write(f"RMSE EKF   [m]: {rmse_e:.3f}\n")
        f.write(f"Loop-closure GT [m]: {loop_gt:.3f}\n")
        f.write(f"Loop-closure Naive [m]: {loop_nv:.3f}\n")
        f.write(f"Loop-closure EKF [m]: {loop_ek:.3f}\n")

    print("✅ Kaydedildi:")
    print(" -", out_traj)
    print(" -", out_err)
    print(" -", out_txt)

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
