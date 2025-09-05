# -*- coding: utf-8 -*-

# imu_sim.py — basit IMU/INS simülatörü



import math

import time



def wrap_pi(a):

    return (a + math.pi) % (2*math.pi) - math.pi



class ImuSim:

    def __init__(self, seed_t=None):

        self.t0 = time.time() if seed_t is None else seed_t

        self.last_t = self.t0



    def step(self, dt=0.02):

        """

        dt: saniye. Çağıran periyodik çağırırsa dt sabit kalır; değilse otomatik hesaplarız.

        """

        now = time.time()

        if dt is None:

            dt = max(1e-3, now - self.last_t)

        self.last_t = now



        t = now - self.t0



        # Basit profil: küçük açılar, yumuşak manevra

        roll  = 10.0 * math.pi/180.0 * math.sin(0.3 * t)     # rad

        pitch =  5.0 * math.pi/180.0 * math.sin(0.2 * t+1.0) # rad

        yaw   = (0.1 * t) % (2*math.pi)                      # yavaş heading artışı

        yaw   = wrap_pi(yaw)



        # Açısal hızlar (p,q,r) ~ roll/pitch türevlerinden türetilmiş kaba yaklaşım

        p = 10.0 * math.pi/180.0 * 0.3 * math.cos(0.3*t)   # rad/s

        q =  5.0 * math.pi/180.0 * 0.2 * math.cos(0.2*t+1) # rad/s

        r =  0.1                                           # sabit z yaw rate (rad/s)



        # İvmeler (uçuşta küçük lateral ivmeler, hafif sinus)

        g = 9.80665

        ax = 0.1 * g * math.sin(0.5*t)   # m/s^2

        ay = 0.1 * g * math.cos(0.4*t)   # m/s^2

        az = g - 0.05 * g * math.sin(0.6*t)  # m/s^2 (aşağı doğru pozitif kabul edilirse offsetli)



        temp_c = 30.0 + 3.0 * math.sin(0.1*t)



        return {

            "roll": roll, "pitch": pitch, "yaw": yaw,

            "p": p, "q": q, "r": r,

            "ax": ax, "ay": ay, "az": az,

            "temp_c": temp_c,

        }

