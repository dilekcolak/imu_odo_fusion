# -*- coding: utf-8 -*-

"""

Live yörünge görselleştirici + MIL-STD-1553 IMU frame HUD


Kısayollar:

  p: durdur/devam

  r: reset

  c: CSV kaydet (LiveSim içindeki save_csv çağrılır)

  q: pencereyi kapat

"""



import time

import numpy as np

import matplotlib.pyplot as plt

from sim_core import LiveSim



# --- Opsiyonel 1553 entegrasyonu ---

_HAS_1553 = True

try:

    from live_1553_bridge import Live1553Bridge

except Exception:

    _HAS_1553 = False

    Live1553Bridge = None  # type: ignore





def run_live(dt: float = 0.05, use_1553: bool = True):

    sim = LiveSim(dt=dt)



    # --- 1553 köprüsü: IMU sample callback & başlat ---

    imu_latest = {"ok": False}  # HUD için son frame

    bridge = None
    rt = None
    _rt_thread = None



    if use_1553:
        try:
            import threading
            from bus1553 import Bus1553
            from rt1553 import RT1553
            from imu_sim import ImuSim
            from ekf import EKF
            from live_1553_bridge import Live1553Bridge

            # IMU HUD’a veri akıtmak için callback
            def _on_sample(d):
                if d.get("src") == "imu":
                    imu_latest.clear()
                    imu_latest.update(d)
                    imu_latest["ok"] = True

            # Bus + RT (IMU sim + EKF) başlat
            bus = Bus1553()
            rt = RT1553(bus, rt_addr=1, sim=ImuSim(), ekf=EKF())
            _rt_thread = threading.Thread(target=rt.run_forever, daemon=True)
            _rt_thread.start()

            # Köprü: csv_path YOK — yerine outdir var
            bridge = Live1553Bridge(
                bus,
                rt_addr=1,
                period_s=max(0.01, dt),
                on_sample=_on_sample,
                outdir="data/streams",
            ).start()

            print("✅ 1553 köprüsü aktif. IMU HUD açık.")
        except Exception as e:
            print(f"⚠️ 1553 köprüsü başlatılamadı, HUD devre dışı. Hata: {e}")
    else:
        print("ℹ️ 1553 köprüsü devre dışı (use_1553=False).")



    # --- Matplotlib setup ---

    plt.ion()

    fig, ax = plt.subplots(figsize=(7, 4.2))



    ln_gt, = ax.plot([], [], color='black', label='Ground Truth')

    ln_nv, = ax.plot([], [], '--', color='red', label='Filtresiz Füzyon')

    ln_ek, = ax.plot([], [], '-.', color='teal', label='Kalman (IMU+Odo)')

    pt_cur, = ax.plot([], [], 'ko', ms=4)



    ax.set_aspect('auto')

    ax.grid(True)

    ax.legend(loc='upper left')

    ax.set_xlabel('X [m]')

    ax.set_ylabel('Y [m]')



    # Üstte metin HUD (RMSE + IMU özet)

    txt = ax.text(0.01, 0.98, '', transform=ax.transAxes, va='top', family='monospace')



    # --- Kısayol davranışları ---

    def on_key(event):

        if event.key == 'p':

            sim.paused = not sim.paused

            print("⏸️  Pause" if sim.paused else "▶️  Devam")

        elif event.key == 'r':

            sim.reset()

            print("🔄 Reset")

        elif event.key == 'c':

            try:

                sim.save_csv()

                print("💾 CSV kaydedildi (LiveSim).")

            except Exception as e:

                print("⚠️ CSV kaydı başarısız:", e)

        elif event.key == 'q':

            plt.close(fig)



    fig.canvas.mpl_connect('key_press_event', on_key)



    # --- Ana döngü ---

    last_draw = time.time()

    try:

        while plt.fignum_exists(fig.number):

            if not sim.paused:

                sim.step()



            # ~20 Hz çizim

            now = time.time()

            if now - last_draw > 0.05:

                if sim.gt:

                    # Trajectory dizileri

                    gt = np.array(sim.gt)

                    nv = np.array(sim.nv) if sim.nv else np.empty((0, 2))

                    ek = np.array(sim.ek) if sim.ek else np.empty((0, 2))



                    # Çizgiler

                    ln_gt.set_data(gt[:, 0], gt[:, 1])

                    if nv.size:

                        ln_nv.set_data(nv[:, 0], nv[:, 1])

                    if ek.size:

                        ln_ek.set_data(ek[:, 0], ek[:, 1])



                    # Son nokta

                    pt_cur.set_data([gt[-1, 0]], [gt[-1, 1]])



                    # Otomatik limit

                    pts_x = [gt[:, 0]]

                    pts_y = [gt[:, 1]]

                    if nv.size:

                        pts_x.append(nv[:, 0])

                        pts_y.append(nv[:, 1])

                    if ek.size:

                        pts_x.append(ek[:, 0])

                        pts_y.append(ek[:, 1])



                    allx = np.concatenate(pts_x)

                    ally = np.concatenate(pts_y)

                    pad = 3.0

                    ax.set_xlim(allx.min() - pad, allx.max() + pad)

                    ax.set_ylim(ally.min() - pad, ally.max() + pad)



                    # RMSE hesapları (liste boşsa 0.0)

                    try:

                        rmse_nv = float(np.sqrt(np.mean(np.square(sim.err_naive)))) if sim.err_naive else 0.0

                    except Exception:

                        rmse_nv = 0.0

                    try:

                        rmse_ek = float(np.sqrt(np.mean(np.square(sim.err_ekf)))) if sim.err_ekf else 0.0

                    except Exception:

                        rmse_ek = 0.0



                    # HUD metni

                    hud = f"RMSE Naive: {rmse_nv:.2f} m  |  RMSE EKF: {rmse_ek:.2f} m"



                    # (Varsa) 1553 IMU özeti

                    if imu_latest.get("ok", False):

                        try:

                            hud += (

                                f"\nIMU (1553): seq={int(imu_latest['seq']):5d}  "

                                f"roll={imu_latest['roll']:+.3f}  pitch={imu_latest['pitch']:+.3f}  yaw={imu_latest['yaw']:+.3f} rad  |  "

                                f"pqr=({imu_latest['p']:+.3f},{imu_latest['q']:+.3f},{imu_latest['r']:+.3f}) rad/s  |  "

                                f"ax,ay,az=({imu_latest['ax']:+.2f},{imu_latest['ay']:+.2f},{imu_latest['az']:+.2f}) m/s²  |  "

                                f"T={imu_latest['temp_c']:+.2f}°C"

                            )

                        except Exception:
                            pass



                    txt.set_text(hud)



                fig.canvas.draw_idle()

                fig.canvas.flush_events()

                last_draw = now



            time.sleep(dt * 0.6)



    except KeyboardInterrupt:

        pass

    finally:

        # Köprüyü temiz kapat

        if bridge is not None:

            try:
                if bridge:
                    bridge.stop()
                if rt:
                    rt.stop()
                if _rt_thread:
                    _rt_thread.join(timeout=2.0)
            except Exception:
                pass





if __name__ == "__main__":

    # use_1553=True ise ve live_1553_bridge.py mevcutsa IMU HUD görünür

    run_live(dt=0.05, use_1553=True)

