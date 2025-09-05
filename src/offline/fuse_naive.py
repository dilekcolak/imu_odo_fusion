import numpy as np

def fuse_naive(t, v_odo, imu_gz, dt, x0=(0.0, 0.0, 0.0)):
    """
    IMU + Odometrinin filtresiz (naive) füzyonu.
    
    Parametreler
    -----------
    t : (N,)        zaman [s]
    v_odo : (N,)    odometriden hız ölçümü [m/s]
    imu_gz: (N,)    IMU gyro ölçümü (yaw hızı) [rad/s]
    dt : float      zaman adımı [s]
    x0 : tuple      başlangıç (x, y, yaw)
    
    Dönenler
    --------
    traj : (N,3)    [x, y, yaw] zaman serisi
    """
    N = len(t)
    traj = np.zeros((N, 3))
    traj[0] = np.array(x0)

    for k in range(1, N):
        x, y, yaw = traj[k-1]

        # yaw IMU’dan
        yaw = yaw + imu_gz[k] * dt

        # hız odometriden
        dx = v_odo[k] * dt * np.cos(yaw)
        dy = v_odo[k] * dt * np.sin(yaw)

        x += dx
        y += dy

        traj[k] = [x, y, yaw]

    return traj


# Küçük test
if __name__ == "__main__":
    from simulate_trajectory import make_cornered_path
    from simulate_imu import simulate_imu
    from simulate_odometry import simulate_odometry

    # ground truth rota
    t, pos, heading = make_cornered_path(total_time=20.0, dt=0.05)

    # IMU ve odometri simülasyonu
    imu_ax, imu_ay, imu_gz, imu_truth = simulate_imu(t, pos, heading, dt=0.05, seed=0)
    v_odo, odo_truth = simulate_odometry(t, pos, heading, dt=0.05, seed=0)

    # Filtresiz füzyon
    traj_naive = fuse_naive(t, v_odo, imu_gz, dt=0.05)

    print("Naive trajectory shape:", traj_naive.shape)
    print("Son konum (naive):", traj_naive[-1, :2])
    print("Son yaw (naive):", traj_naive[-1, 2])
