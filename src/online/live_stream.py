# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
from sim_core import LiveSim

def run_live(dt=0.05):
    sim = LiveSim(dt=dt)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7,4.2))
    ln_gt,   = ax.plot([], [], color='black', label='Ground Truth')
    ln_nv,   = ax.plot([], [], '--', color='red',   label='Filtresiz Füzyon')
    ln_ek,   = ax.plot([], [], '-.', color='teal',  label='Kalman (IMU+Odo)')
    pt_cur,  = ax.plot([], [], 'ko', ms=4)

    ax.set_aspect('auto')
    ax.grid(True); ax.legend(loc='upper left')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')

    txt = ax.text(0.01, 0.98, '', transform=ax.transAxes, va='top')

    def on_key(event):
        if event.key == 'p':
            sim.paused = not sim.paused
            print("⏸️  Pause" if sim.paused else "▶️  Devam")
        elif event.key == 'r':
            sim.reset()
        elif event.key == 'c':
            sim.save_csv()
        elif event.key == 'q':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)

    last_draw = time.time()
    try:
        while plt.fignum_exists(fig.number):
            if not sim.paused:
                sim.step()

            # ~20 Hz çizim
            now = time.time()
            if now - last_draw > 0.05:
                if sim.gt:
                    gt = np.array(sim.gt); nv = np.array(sim.nv); ek = np.array(sim.ek)
                    ln_gt.set_data(gt[:,0], gt[:,1])
                    ln_nv.set_data(nv[:,0], nv[:,1])
                    ln_ek.set_data(ek[:,0], ek[:,1])
                    pt_cur.set_data([gt[-1,0]],[gt[-1,1]])

                    allx = np.concatenate([gt[:,0], nv[:,0], ek[:,0]])
                    ally = np.concatenate([gt[:,1], nv[:,1], ek[:,1]])
                    pad = 3.0
                    ax.set_xlim(allx.min()-pad, allx.max()+pad)
                    ax.set_ylim(ally.min()-pad, ally.max()+pad)

                    import numpy as _np
                    rmse_nv = np.sqrt(np.mean(np.square(sim.err_naive))) if sim.err_naive else 0.0
                    rmse_ek = np.sqrt(np.mean(np.square(sim.err_ekf)))   if sim.err_ekf   else 0.0
                    txt.set_text(f"RMSE Naive: {rmse_nv:.2f} m  |  RMSE EKF: {rmse_ek:.2f} m  "
                                 f"|  Anlık hata (N/E): {sim.err_naive[-1]:.2f}/{sim.err_ekf[-1]:.2f} m")

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_draw = now

            time.sleep(dt*0.6)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_live(dt=0.05)
