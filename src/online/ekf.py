# -*- coding: utf-8 -*-
import numpy as np

class EKF:
    """
    X=[x, y, psi, b_g, v] durumu, kovaryans ve ekf_step().
    Parametreler, önceki live_stream.py ile birebir.
    """
    def __init__(self):
        self.X = np.zeros(5)
        self.P = np.eye(5)*1e-3
        self.P[2,2] = (np.deg2rad(12.0))**2
        self.P[4,4] = 1.0

        # Süreç ve ölçüm parametreleri
        self.q_v  = 0.70
        self.q_bg = np.deg2rad(0.03)

        # Odo'ya güven azaltıldı (gerçekçilik)
        self.r_v  = 0.30                    # m/s
        self.r_w  = np.deg2rad(0.60)        # rad/s

        # Adaptif R eşikleri
        self.th_v, self.scale_v = 0.20, 30.0
        self.th_w, self.scale_w = 0.15, 25.0

    def step(self, dt, gz, v_odo, w_odo):
        X = self.X; P = self.P
        x,y,psi,bg,v = X

        # ---- PREDICT ----
        psi_p = psi + (gz - bg)*dt
        v_p   = v
        x_p   = x + v_p*dt*np.cos(psi_p)
        y_p   = y + v_p*dt*np.sin(psi_p)
        bg_p  = bg
        Xp = np.array([x_p,y_p,psi_p,bg_p,v_p])

        F = np.eye(5)
        F[0,2] = -v_p*dt*np.sin(psi_p)
        F[0,4] =  dt*np.cos(psi_p)
        F[1,2] =  v_p*dt*np.cos(psi_p)
        F[1,4] =  dt*np.sin(psi_p)
        F[2,3] = -dt

        Q = np.zeros((5,5))
        Q[3,3] = (self.q_bg)**2
        Q[4,4] = (self.q_v)**2
        P = F@P@F.T + Q

        # ---- UPDATE #1: v_odo ----
        H_v = np.zeros((1,5)); H_v[0,4] = 1.0
        innov_v = np.array([[v_odo]]) - H_v@Xp
        Rv = np.array([[self.r_v**2]])
        if abs(float(innov_v)) > self.th_v:
            Rv *= self.scale_v
        S = H_v@P@H_v.T + Rv
        K = P@H_v.T@np.linalg.inv(S)
        Xk = Xp + (K@innov_v).reshape(-1)
        P  = (np.eye(5) - K@H_v)@P

        # ---- UPDATE #2: w_odo  ~ (imu_gz - b_g) ----
        H_w = np.zeros((1,5)); H_w[0,3] = -1.0
        innov_w = np.array([[w_odo]]) - (gz - Xk[3])
        Rw = np.array([[self.r_w**2]])
        if abs(float(innov_w)) > self.th_w:
            Rw *= self.scale_w
        S = H_w@P@H_w.T + Rw
        K = P@H_w.T@np.linalg.inv(S)
        Xk = Xk + (K@innov_w).reshape(-1)
        P  = (np.eye(5) - K@H_w)@P

        self.X, self.P = Xk, P
        return self.X, self.P
