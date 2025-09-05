import numpy as np

def fuse_ekf(
    t,                    # (N,) zaman [s]
    v_odo,                # (N,) odometri hız ölçümü [m/s]
    imu_gz,               # (N,) IMU gyro (yaw-rate) [rad/s]
    omega_odo,            # (N,) odometri yaw-rate ölçümü [rad/s]
    dt,                   # zaman adımı [s]

    # süreç gürültüleri
    q_v=0.50,             # hız için random-walk std
    q_bg=np.deg2rad(0.02),# gyro bias random-walk std

    # ölçüm gürültüleri (nominal)
    r_v=0.08,             # v_odo ölçüm std [m/s]
    r_w=np.deg2rad(0.12), # omega_odo ölçüm std [rad/s]

    # adaptif R (slip tespiti)
    slip_innov_thresh_v=0.30,
    slip_R_scale_v=200.0,
    slip_innov_thresh_w=0.20,
    slip_R_scale_w=120.0,

    # başlangıç durumu: [x, y, psi, b_g, v]
    x0=(0.0, 0.0, 0.0, 0.0, 0.0),

    # başlangıç kovaryansı ayarları
    yaw_init_std_deg=10.0,  # psi için başlangıç std (derece)
    v_init_var=1.0          # hız diyagonal kovaryansı (büyük olsun)
):
    """
    EKF durum: X = [x, y, psi, b_g, v]
      - Süreç modeli:
          psi' = psi + (imu_gz - b_g)*dt
          v'   = v        (random-walk Q ile)
          x'   = x + v'*dt*cos(psi')
          y'   = y + v'*dt*sin(psi')
          b_g' = b_g      (random-walk Q ile)
      - Ölçümler:
          z_v = v_odo                 (H_v: [0,0,0,0,1])
          z_w = omega_odo ≈ imu_gz - b_g   (H_w: db_g = -1)
        Not: z_w, psi' yerine yaw-rate inovasyonunda bias’ı kalibre eder.
      - Adaptif R:
          |innov| eşikleri aşarsa ilgili R ölçeklenir (slip etkisini bastırır).

    Dönen:
      X : (N,5) zaman serisi [x, y, psi, b_g, v]
    """
    t = np.asarray(t)
    v_odo = np.asarray(v_odo)
    imu_gz = np.asarray(imu_gz)
    omega_odo = np.asarray(omega_odo)

    N = len(t)
    X = np.zeros((N, 5), dtype=float)
    X[0] = np.array(x0, dtype=float)

    # Başlangıç kovaryansı
    P = np.eye(5) * 1e-3
    # yaw belirsizliği (daha gerçekçi başlangıç)
    P[2, 2] = (np.deg2rad(yaw_init_std_deg))**2
    # hız belirsizliği büyük: ölçümleri daha kolay kabul etsin
    P[4, 4] = v_init_var

    # Süreç gürültüsü matrisi Q
    Q = np.zeros((5, 5))
    # psi süreç gürültüsünü doğrudan koymuyoruz; psi imu_gz üzerinden ilerliyor
    Q[3, 3] = (q_bg)**2   # b_g random walk
    Q[4, 4] = (q_v)**2    # v random walk

    # Ölçüm gürültüleri (nominal)
    Rv = np.array([[r_v**2]])           # v_odo
    Rw = np.array([[r_w**2]])           # omega_odo

    I = np.eye(5)

    for k in range(1, N):
        x, y, psi, bg, v = X[k-1]

        # --------- PREDICT ---------
        gz = imu_gz[k] - bg           # bias düzeltilmiş gyro
        psi_p = psi + gz * dt
        v_p   = v
        x_p   = x + v_p * dt * np.cos(psi_p)
        y_p   = y + v_p * dt * np.sin(psi_p)
        bg_p  = bg

        Xp = np.array([x_p, y_p, psi_p, bg_p, v_p], dtype=float)

        # Jacobian F = d f / d X
        F = np.eye(5)
        F[0, 2] = -v_p * dt * np.sin(psi_p)  # dx/dpsi
        F[0, 4] =  dt * np.cos(psi_p)        # dx/dv
        F[1, 2] =  v_p * dt * np.cos(psi_p)  # dy/dpsi
        F[1, 4] =  dt * np.sin(psi_p)        # dy/dv
        F[2, 3] = -dt                         # dpsi/db_g = -dt

        P = F @ P @ F.T + Q

        # --------- UPDATE #1: v_odo ---------
        # z_v = v  ;  H_v = [0,0,0,0,1]
        z_v = np.array([[v_odo[k]]], dtype=float)
        H_v = np.zeros((1, 5), dtype=float)
        H_v[0, 4] = 1.0

        innov_v = z_v - (H_v @ Xp)
        R_use_v = Rv.copy()
        if abs(innov_v[0, 0]) > slip_innov_thresh_v:
            R_use_v *= slip_R_scale_v

        S_v = H_v @ P @ H_v.T + R_use_v
        K_v = P @ H_v.T @ np.linalg.inv(S_v)
        Xk = Xp + (K_v @ innov_v).reshape(-1)
        P  = (I - K_v @ H_v) @ P

        # --------- UPDATE #2: omega_odo ---------
        # Model: z_w ≈ imu_gz[k] - b_g  => inovasyon: (z_w - (imu_gz[k] - b_g))
        z_w = np.array([[omega_odo[k]]], dtype=float)
        innov_w = z_w - (imu_gz[k] - Xk[3])  # skaler inovasyon

        # h_w'nin X'e türevi: dh/db_g = -1  => H_w = [0,0,0,-1,0]
        H_w = np.zeros((1, 5), dtype=float)
        H_w[0, 3] = -1.0

        R_use_w = Rw.copy()
        if abs(innov_w[0, 0]) > slip_innov_thresh_w:
            R_use_w *= slip_R_scale_w

        S_w = H_w @ P @ H_w.T + R_use_w
        K_w = P @ H_w.T @ np.linalg.inv(S_w)
        Xk = Xk + (K_w @ innov_w).reshape(-1)
        P  = (I - K_w @ H_w) @ P

        # kayıt
        X[k] = Xk

    return X

