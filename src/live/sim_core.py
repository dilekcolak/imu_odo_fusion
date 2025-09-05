#coding: utf-8 -*-

import os, csv, time

import numpy as np

from utils import wrap_pi

from sensors import SensorSim

from ekf import EKF



class LiveSim:

    """

    GT üret + sensör ölç + Naive + EKF + log/CSV.

    Çizim yok; onu live_stream.py yapıyor.

    """

    def __init__(self, dt=0.05, total_keep=3000):

        self.dt = float(dt)

        self.keep = int(total_keep)



        # GT durumu

        self.x = 0.0; self.y = 0.0; self.psi = 0.0



        # Naive durumu

        self.nx = 0.0; self.ny = 0.0; self.npsi = 0.0



        # Bileşenler

        self.sens = SensorSim(dt=self.dt, keep=self.keep, v_mean=1.0)

        self.ekf  = EKF()



        # tarihçe (ring buffer)

        self.gt = []

        self.nv = []

        self.ek = []

        self.t  = []

        self.err_naive = []

        self.err_ekf   = []



        # kayıt

        self.log_rows = []



        # kontrol

        self.paused = False

        self._last_print_s = -1.0



    def step(self):

        dt = self.dt



        # Komutlar (sensörden; eve dönüş mantığı için GT pozisyonunu ver)

        v_cmd, w_cmd = self.sens.command(self.x, self.y, self.psi)



        # GT entegrasyon

        self.psi = wrap_pi(self.psi + w_cmd*dt)

        self.x   = self.x + v_cmd*dt*np.cos(self.psi)

        self.y   = self.y + v_cmd*dt*np.sin(self.psi)



        # Sensör ölçümleri

        gz, v_odo, w_odo = self.sens.measure(v_cmd, w_cmd)



        # Naive entegrasyon

        self.npsi = wrap_pi(self.npsi + gz*dt)

        self.nx   = self.nx + v_odo*dt*np.cos(self.npsi)

        self.ny   = self.ny + v_odo*dt*np.sin(self.npsi)



        # EKF

        Xk, _ = self.ekf.step(dt, gz, v_odo, w_odo)



        # tarihçe + zaman

        self.gt.append([self.x,self.y])

        self.nv.append([self.nx,self.ny])

        self.ek.append([Xk[0],Xk[1]])

        self.t.append(self.t[-1]+dt if self.t else 0.0)

        self.sens.t = self.t  # eve dönüş mantığı için sensöre aktar



        # hatalar

        err_n = np.hypot(self.nx - self.x, self.ny - self.y)

        err_e = np.hypot(Xk[0]-self.x, Xk[1]-self.y)

        self.err_naive.append(err_n)

        self.err_ekf.append(err_e)



        # terminal log ~1 Hz

        if int(self.t[-1]) != int(self._last_print_s):

            self._last_print_s = self.t[-1]

            rmse_nv = np.sqrt(np.mean(np.square(self.err_naive)))

            rmse_ek = np.sqrt(np.mean(np.square(self.err_ekf)))

            import numpy as _np

            print(f"t={self.t[-1]:5.1f}s  |  v_odo={v_odo:+.2f} m/s  "

                  f"w_odo={_np.rad2deg(w_odo):+.1f} deg/s  "

                  f"|  err(N/E)={err_n:.2f}/{err_e:.2f} m  "

                  f"|  RMSE(N/E)={rmse_nv:.2f}/{rmse_ek:.2f} m")



        # CSV ham log

        self.log_rows.append([

            self.t[-1],

            self.x, self.y, self.psi,

            gz, v_odo, w_odo,

            self.nx, self.ny, self.npsi,

            self.ekf.X[0], self.ekf.X[1], self.ekf.X[2], self.ekf.X[3], self.ekf.X[4]

        ])



        # ring buffer sınırı

        for arr in (self.gt, self.nv, self.ek, self.t, self.err_naive, self.err_ekf):

            if len(arr) > self.keep:

                del arr[0]



    def save_csv(self, outpath=None):

        # Proje kökü: .../imu_odo_fusion

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if outpath is None:

            outdir = os.path.join(base_dir, "data", "runs")

            os.makedirs(outdir, exist_ok=True)

            outpath = os.path.join(outdir, "run_latest.csv")



        header = [

            "t",

            "gt_x","gt_y","gt_yaw",

            "imu_gz","odo_v","odo_w",

            "naive_x","naive_y","naive_yaw",

            "ekf_x","ekf_y","ekf_yaw","ekf_bg","ekf_v"

        ]

        tmp = outpath + ".tmp"

        with open(tmp, "w", newline="") as f:

            w = csv.writer(f)

            w.writerow(header)

            w.writerows(self.log_rows)

        os.replace(tmp, outpath)

        print(f"✅ CSV güncellendi: {outpath}")



    def reset(self):

        print("↺ Resetlendi (yeni rastgele akış).")

        self.__init__(dt=self.dt, total_keep=self.keep)

