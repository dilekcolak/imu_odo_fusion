
# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
from utils import wrap_pi, LowPass

class SensorSim:
    """
    Komut üretimi (v,w) + IMU/Odo ölçümleri.
    'Realism pack' burada. Davranış, eski live_stream.py ile aynıdır.
    """
    def __init__(self, dt, keep, v_mean=1.0):
        self.dt = float(dt)
        self.keep = int(keep)

        # Komut üretici (pürüzsüz v & w)
        self.v_mean = v_mean
        self.lp_v = LowPass(0.94, self.v_mean)
        self.lp_w = LowPass(0.90, 0.0)
        self.seg_left = 0.0

        # IMU (gyro)
        self.bg = 0.0
        self.gyro_bias_rw = np.deg2rad(0.05)   # rad/s * sqrt(dt)
        self.gyro_noise   = np.deg2rad(0.35)   # rad/s beyaz gürültü

        # Odometri (gürültü + slip)
        self.v_noise   = 0.10                  # m/s
        self.w_noise   = np.deg2rad(0.20)      # rad/s
        self.slip_p_v  = 0.18
        self.slip_s_v  = (0.30, 0.70)
        self.slip_p_w  = 0.14
        self.slip_s_w  = (0.25, 0.70)

        # Realism pack: kalibrasyon hataları + gecikmeler
        self.kv_scale = 1.03
        self.kw_scale = 0.97
        self.bv = 0.02
        self.bw = np.deg2rad(0.10)
        self.bv_rw = 0.010
        self.bw_rw = np.deg2rad(0.02)

        self.gyro_sf = 1.002

        self.delay_v = 0.06
        self.delay_w = 0.08
        self.buf_v = deque([0.0] * (int(self.delay_v/self.dt)+1),
                           maxlen=int(self.delay_v/self.dt)+1)
        self.buf_w = deque([0.0] * (int(self.delay_w/self.dt)+1),
                           maxlen=int(self.delay_w/self.dt)+1)

        # dıştan okunacak: GT’ye “eve dönüş” kararında süre izlemesi için
        self.t = []

    def command(self, x, y, psi):
        """Pürüzsüz komutlar. Eve dönüş eğilimini korur (aynı mantık)."""
        if self.seg_left <= 0.0:
            self.seg_left = np.random.uniform(1.0, 3.0)
            v_cmd = self.v_mean * np.random.uniform(0.7, 1.3)
            w_cmd = np.random.uniform(-0.8, 0.8)  # rad/s
        else:
            v_cmd = self.v_mean
            w_cmd = 0.0
        self.seg_left -= self.dt

        v = self.lp_v.step(v_cmd)
        w = self.lp_w.step(w_cmd)

        # “eve dönüş” eğilimi (loop-closure testi)
        if len(self.t) > 0.8*self.keep:
            to_home = np.array([-x, -y])
            desired = np.arctan2(to_home[1], to_home[0])
            err = wrap_pi(desired - psi)
            w = 0.7*err
        return v, w

    def measure(self, v_true, w_true):
        """IMU ve Odo ölçümlerini üretir (realism pack dahil)."""
        # IMU gyro
        self.bg += self.gyro_bias_rw*np.random.randn()
        gz = (self.gyro_sf * w_true) + self.bg + self.gyro_noise*np.random.randn()

        # Odo bias/drift güncelle
        self.bv += self.bv_rw * np.sqrt(self.dt) * np.random.randn()
        self.bw += self.bw_rw * np.sqrt(self.dt) * np.random.randn()

        # Odo v
        v_meas = (self.kv_scale * v_true + self.bv) + self.v_noise*np.random.randn()
        if np.random.rand() < self.slip_p_v:
            v_meas *= (1.0 + np.random.uniform(*self.slip_s_v)*np.random.choice([-1,1]))

        # Odo w
        w_meas = (self.kw_scale * w_true + self.bw) + self.w_noise*np.random.randn()
        if np.random.rand() < self.slip_p_w:
            w_meas *= (1.0 + np.random.uniform(*self.slip_s_w)*np.random.choice([-1,1]))

        # Gecikmeler
        self.buf_v.append(v_meas)
        self.buf_w.append(w_meas)
        return gz, self.buf_v[0], self.buf_w[0]
