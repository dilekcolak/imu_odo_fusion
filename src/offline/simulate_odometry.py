import numpy as np

def simulate_odometry(
    t, pos_xy, heading_rad, dt,
    odo_noise=0.02,                    # m/s hız gürültüsü
    odo_omega_noise=np.deg2rad(0.03),  # rad/s yaw-rate gürültüsü
    slip_prob_lin=0.20, slip_scale_lin=(0.30, 0.70),
    slip_prob_ang=0.15, slip_scale_ang=(0.30, 0.70),
    seed=None,
):
    """
    Tekerlek odometrisi simülasyonu: lineer hız v_odo ve açısal hız ω_odo üretir.
    Slip olayları hem lineer hem açısal ölçümlere uygulanır.
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(t)
    pos_xy = np.asarray(pos_xy)
    heading_rad = np.asarray(heading_rad)

    # Gerçek hız ve yaw-rate
    vel_xy = np.zeros_like(pos_xy)
    vel_xy[1:] = (pos_xy[1:] - pos_xy[:-1]) / dt
    v_true = np.linalg.norm(vel_xy, axis=1)

    yaw_rate_true = np.zeros(N)
    yaw_rate_true[1:] = (heading_rad[1:] - heading_rad[:-1]) / dt

    # Ölçümler: gürültü
    v_meas = v_true + odo_noise * np.random.randn(N)
    w_meas = yaw_rate_true + odo_omega_noise * np.random.randn(N)

    # Slip olayları
    slip_mask_lin = np.random.rand(N) < slip_prob_lin
    slip_mask_ang = np.random.rand(N) < slip_prob_ang
    for k in range(N):
        if slip_mask_lin[k]:
            v_meas[k] *= (1.0 + np.random.uniform(*slip_scale_lin) * np.random.choice([-1, 1]))
        if slip_mask_ang[k]:
            w_meas[k] *= (1.0 + np.random.uniform(*slip_scale_ang) * np.random.choice([-1, 1]))

    truth = {
        "v_true": v_true,
        "yaw_rate_true": yaw_rate_true,
        "slip_lin": slip_mask_lin,
        "slip_ang": slip_mask_ang,
    }
    return v_meas, w_meas, truth


if __name__ == "__main__":
    from simulate_trajectory import make_cornered_path
    t, pos, heading = make_cornered_path(total_time=10.0, dt=0.05)
    v_odo, w_odo, tr = simulate_odometry(t, pos, heading, 0.05, seed=0)
    print("örnek:", v_odo[:5], w_odo[:5], "slip_lin:", np.sum(tr["slip_lin"]), "slip_ang:", np.sum(tr["slip_ang"]))

