# -*- coding: utf-8 -*-
"""
live_1553_bridge.py — 1553 RT (IMU+EKF) ↔ BC köprüsü (callback + ayrı CSV akışları)

Bu sürüm:
- IMU ve EKF stream'lerini AYRI CSV dosyalarına yazar (data/streams/imu_stream.csv, ekf_stream.csv)
- Temiz kapanış için Event tabanlı stop() ve join() uygular
- İsteğe bağlı on_sample(dict) callback'i çağırır (kaynak: "src" alanı "imu" veya "ekf")
"""

import os
import csv
import time
import threading
from typing import Callable, Optional

from bus1553 import Bus1553
from bc1553 import BC1553

class Live1553Bridge:
    def __init__(self,
                 bus: Bus1553,
                 bc: Optional[BC1553] = None,
                 rt_addr: int = 1,
                 period_s: float = 0.02,
                 on_sample: Optional[Callable[[dict], None]] = None,
                 outdir: str = 'data/streams'):
        self.bus = bus
        self.bc = bc or BC1553(self.bus)
        self.rt_addr = int(rt_addr)
        self.period_s = float(period_s)
        self.on_sample = on_sample
        self.outdir = outdir

        self._stop_evt = threading.Event()
        self._threads = []

        self._imu_fp = None
        self._imu_w  = None
        self._ekf_fp = None
        self._ekf_w  = None

    def start(self):
        os.makedirs(self.outdir, exist_ok=True)
        # IMU stream CSV
        self._imu_fp = open(os.path.join(self.outdir, 'imu_stream.csv'), 'w', newline='')
        self._imu_w = csv.writer(self._imu_fp)
        self._imu_w.writerow(['t','seq','yaw','p','q','r','ax','ay','az','temp_c'])
        # EKF stream CSV
        self._ekf_fp = open(os.path.join(self.outdir, 'ekf_stream.csv'), 'w', newline='')
        self._ekf_w = csv.writer(self._ekf_fp)
        self._ekf_w.writerow(['t','seq','x','y','z','vx','vy','vz','roll','pitch','yaw'])

        t1 = threading.Thread(target=self._loop_poll_imu, daemon=True)
        t2 = threading.Thread(target=self._loop_poll_ekf, daemon=True)
        t1.start(); t2.start()
        self._threads = [t1, t2]
        return self

    def stop(self):
        self._stop_evt.set()
        for t in self._threads:
            t.join(timeout=2.0)
        for fp in (self._imu_fp, self._ekf_fp):
            if fp:
                fp.flush()
                fp.close()

    # ---------- internal loops ----------
    def _loop_poll_imu(self):
        t0 = time.time()
        while not self._stop_evt.is_set():
            try:
                sample = self.bc.poll_imu(self.rt_addr, timeout=0.5)  # dict bekler
                if sample:
                    now = time.time() - t0
                    row = [f'{now:.3f}', sample.get('seq',0),
                           f"{sample.get('yaw',0.0):.6f}", f"{sample.get('p',0.0):.6f}", f"{sample.get('q',0.0):.6f}",
                           f"{sample.get('r',0.0):.6f}", f"{sample.get('ax',0.0):.6f}", f"{sample.get('ay',0.0):.6f}",
                           f"{sample.get('az',0.0):.6f}", f"{sample.get('temp_c',0.0):.2f}"]
                    self._imu_w.writerow(row)
                    if self.on_sample:
                        d = dict(sample); d['src']='imu'; d['t']=now
                        self.on_sample(d)
            except Exception:
                pass
            time.sleep(self.period_s)

    def _loop_poll_ekf(self):
        t0 = time.time()
        while not self._stop_evt.is_set():
            try:
                sample = self.bc.poll_ekf(self.rt_addr, timeout=0.5)  # dict bekler
                if sample:
                    now = time.time() - t0
                    row = [f'{now:.3f}', sample.get('seq',0),
                           f"{sample.get('x',0.0):.6f}", f"{sample.get('y',0.0):.6f}", f"{sample.get('z',0.0):.6f}",
                           f"{sample.get('vx',0.0):.6f}", f"{sample.get('vy',0.0):.6f}", f"{sample.get('vz',0.0):.6f}",
                           f"{sample.get('roll',0.0):.6f}", f"{sample.get('pitch',0.0):.6f}", f"{sample.get('yaw',0.0):.6f}"]
                    self._ekf_w.writerow(row)
                    if self.on_sample:
                        d = dict(sample); d['src']='ekf'; d['t']=now
                        self.on_sample(d)
            except Exception:
                pass
            time.sleep(self.period_s)
