# -*- coding: utf-8 -*-
"""
live_bc_demo.py — Basit demo: RT'yi başlat, BC ile sonsuz IMU+EKF çek, Ctrl+C ile temiz kapan.

Bu sürüm:
- Sonsuz akış (while True) ve KeyboardInterrupt ile çıkış
- rt.stop() + join() ile thread'leri temiz kapatır
- görselleştirme olmadan sadece logları görmek için kullanılır
"""

import threading
import time

from bus1553 import Bus1553
from bc1553 import BC1553
from rt1553 import RT1553
from imu_sim import ImuSim
from ekf import EKF

def main():
    bus = Bus1553()
    bc = BC1553(bus)

    # RT tarafı: IMU sim + EKF ile çalışsın
    sim = ImuSim()
    ekf = EKF()
    rt = RT1553(bus, rt_addr=1, sim=sim, ekf=ekf)
    t = threading.Thread(target=rt.run_forever, daemon=True)
    t.start()

    print('[Demo] IMU(EKF) okuma başlıyor — SA_IMU=2 SA_EKF=3')
    try:
        while True:
            # IMU oku
            imu = bc.poll_imu(rt=1, timeout=0.5)
            if imu:
                print(f"[IMU] seq={imu.get('seq'):5d} yaw={imu.get('yaw',0.0):+6.3f} rad | "
                      f"pqr=({imu.get('p',0.0):+5.3f},{imu.get('q',0.0):+5.3f},{imu.get('r',0.0):+5.3f}) rad/s | "
                      f"acc=({imu.get('ax',0.0):+5.2f},{imu.get('ay',0.0):+5.2f},{imu.get('az',0.0):+5.2f}) m/s^2")
            # EKF oku
            ek = bc.poll_ekf(rt=1, timeout=0.5)
            if ek:
                print(f"[EKF] seq={ek.get('seq'):5d} pos=({ek.get('x',0.0):+6.3f},{ek.get('y',0.0):+6.3f}) m "
                      f"vel=({ek.get('vx',0.0):+5.3f}, {ek.get('vy',0.0):+5.3f}) m/s yaw={ek.get('yaw',0.0):+6.3f} rad")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping…")
    finally:
        rt.stop()
        t.join(timeout=2.0)
        print('Bye.')

if __name__ == '__main__':
    main()

