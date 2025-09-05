import numpy as np

def simulate_imu(t, pos_xy, heading_rad, dt, accel_noise = 0.15, gyro_noise=np.deg2rad(0.30), accel_bias_rw=0.001, gyro_bias_rw=np.deg2rad(0.05), seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = len(t)
    pos_xy = np.asarray(pos_xy)
    heading_rad = np.asarray(heading_rad)

    vel_xy = np.zeros_like(pos_xy)
    vel_xy[1:] = (pos_xy[1:] - pos_xy[:-1]) / dt

    acc_world = np.zeros_like(pos_xy)
    acc_world[1:] = (vel_xy[1:] - vel_xy[:-1]) / dt

    c, s = np.cos(heading_rad), np.sin(heading_rad)
    acc_body = np.zeros_like(acc_world)
    for k in range(N):
        acc_body[k, 0] =  c[k]*acc_world[k,0] + s[k]*acc_world[k,1]
        acc_body[k, 1] = -s[k]*acc_world[k,0] + c[k]*acc_world[k,1]

    yaw_rate = np.zeros(N)
    yaw_rate[1:] = (heading_rad[1:] - heading_rad[:-1]) / dt

    bias_ax = np.zeros(N)
    bias_ay = np.zeros(N)
    bias_gz = np.zeros(N)
    for k in range(1, N):
        bias_ax[k] = bias_ax[k-1] + accel_bias_rw * np.random.randn()
        bias_ay[k] = bias_ay[k-1] + accel_bias_rw * np.random.randn()
        bias_gz[k] = bias_gz[k-1] + gyro_bias_rw  * np.random.randn()

    imu_ax = acc_body[:,0] + bias_ax + accel_noise*np.random.randn(N)
    imu_ay = acc_body[:,1] + bias_ay + accel_noise*np.random.randn(N)
    imu_gz = yaw_rate + bias_gz + gyro_noise*np.random.randn(N)


    truth = {
        "acc_body": acc_body,
        "yaw_rate": yaw_rate,
        "bias_ax": bias_ax,
        "bias_ay": bias_ay,
        "bias_gz": bias_gz,
    }
    return imu_ax, imu_ay, imu_gz, truth


# Küçük yerel test (isteğe bağlı): direkt çalıştırıldığında sadece boyutları basar
if __name__ == "__main__":
    from simulate_trajectory import make_cornered_path
    t, pos, heading = make_cornered_path(total_time=10.0, dt=0.05)
    imu_ax, imu_ay, imu_gz, truth = simulate_imu(t, pos, heading, dt=0.05, seed=0)
    print("örnek:", imu_ax.shape, imu_ay.shape, imu_gz.shape)

