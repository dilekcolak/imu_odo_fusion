import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from simulate_trajectory_curvy import make_curvy_path
from simulate_trajectory import make_cornered_path
from simulate_imu import simulate_imu
from simulate_odometry import simulate_odometry
from fuse_naive import fuse_naive
from fuse_ekf import fuse_ekf
import matplotlib.pyplot as plt
import numpy as np

DT = 0.05
T_TOTAL = 120.0

# 1) Ground Truth
#t, pos, heading = make_cornered_path(total_time=T_TOTAL, dt=DT)
t, pos, heading = make_curvy_path(total_time=T_TOTAL, dt=DT, seed=None)

# 2) Sensörler
imu_ax, imu_ay, imu_gz, _ = simulate_imu(t, pos, heading, dt=DT, seed=None)
v_odo, w_odo, _ = simulate_odometry(t, pos, heading, dt=DT, seed=None)

# 3) Naive füzyon
traj_naive = fuse_naive(t, v_odo, imu_gz, dt=DT)

# 4)  EKF füzyon (hız ve yaw-rate ölçümleriyle)
X_ekf = fuse_ekf(
    t, v_odo, imu_gz, w_odo, dt=DT,
    q_v=0.30, q_bg=np.deg2rad(0.01),
    r_v=0.05, r_w=np.deg2rad(0.08),
    slip_innov_thresh_v=0.6,  slip_R_scale_v=80.0,
    slip_innov_thresh_w=0.4,  slip_R_scale_w=80.0,
    x0=(0.0, 0.0, 0.0, 0.0, v_odo[0])
)


# 5) Çizim
plt.figure(figsize=(6,6))
plt.plot(pos[:,0], pos[:,1], label="Ground Truth", color="black")
plt.plot(traj_naive[:,0], traj_naive[:,1], "--", label="Filtresiz Füzyon", color="red")
plt.plot(X_ekf[:,0], X_ekf[:,1], "-.", label="Kalman (IMU+Odo)", color="teal")

plt.scatter(pos[0,0], pos[0,1], c="green", marker="o", label="Başlangıç")
plt.scatter(pos[-1,0], pos[-1,1], c="blue", marker="x", label="GT Bitiş")
plt.scatter(traj_naive[-1,0], traj_naive[-1,1], c="red", marker="x", label="Naive Bitiş")
plt.scatter(X_ekf[-1,0], X_ekf[-1,1], c="teal", marker="x", label="EKF Bitiş")

plt.xlabel("X [m]"); plt.ylabel("Y [m]")
plt.legend(); plt.axis("equal"); plt.grid(True)
plt.title("GT vs Filtresiz Füzyon vs Kalman (IMU + Odo)")
plt.show()

# (İsteğe bağlı) zamanla konum hatası kıyası
err_naive = np.linalg.norm(traj_naive[:,:2] - pos, axis=1)
err_ekf   = np.linalg.norm(X_ekf[:,:2] - pos, axis=1)

plt.figure(figsize=(7,3.5))
plt.plot(t, err_naive, "--", label="Naive konum hatası")
plt.plot(t, err_ekf, "-.", label="EKF konum hatası")
plt.xlabel("Zaman [s]"); plt.ylabel("Hata [m]")
plt.grid(True); plt.legend(); plt.title("Konum Hatası (GT referans)")
plt.show()

# (İsteğe bağlı) loop-closure (başlangıç-bitiş ofseti)
loop_naive = np.linalg.norm(traj_naive[-1,:2] - traj_naive[0,:2])
loop_ekf   = np.linalg.norm(X_ekf[-1,:2]       - X_ekf[0,:2])
print(f"Loop-closure hata [m]  Naive: {loop_naive:.3f}   EKF: {loop_ekf:.3f}")
