# -*- coding: utf-8 -*-
"""
rt1553.py — MIL-STD-1553 Remote Terminal (RT)

- T/R=0 (RT Receive): BC -> DATA (wc), RT -> STATUS
- T/R=1 (RT Transmit): RT -> STATUS, RT -> DATA (wc)

IMU ve EKF akışları 14-word sabit frame üretir.
"""

import time
from typing import Callable, List, Optional, Tuple, Union

from bus1553 import (
    Bus1553, make_word, unpack_word,
    SYNC_CMD, SYNC_DATA, SYNC_STATUS
)
from sensor1553 import (
    FRAME_WORDS,
    pack_imu_words, unpack_imu_words,
    pack_ekf_words, unpack_ekf_words,
    SA_IMU, SA_EKF,
)

# ---------- Command/Status helpers ----------

def make_command_field(rt: int, tr: int, sa: int, wc: int) -> int:
    rt &= 0x1F; tr &= 0x01; sa &= 0x1F; wc &= 0x1F
    return (rt << 11) | (tr << 10) | (sa << 5) | wc

def parse_command_field(field: int) -> Tuple[int, int, int, int]:
    rt = (field >> 11) & 0x1F
    tr = (field >> 10) & 0x01
    sa = (field >> 5) & 0x1F
    wc = field & 0x1F
    return rt, tr, sa, wc

def make_status_field(rt: int, bits: int = 0) -> int:
    rt &= 0x1F; bits &= 0xFF
    return (rt << 11) | bits

def parse_status_field(field: int) -> Tuple[int, int]:
    rt = (field >> 11) & 0x1F
    bits = field & 0xFF
    return rt, bits


# ============================ RT ============================

class RT1553:
    def __init__(
        self,
        bus: Bus1553,
        rt_addr: int = 1,
        sensor_cb: Optional[Callable[[int], List[int]]] = None,
        sim=None,
        ekf=None,
    ):
        self.bus = bus
        self.addr = rt_addr & 0x1F

        # dış komponentler
        self.sensor_cb = sensor_cb  # istenirse dış IMU üreticisi
        self.sim = sim              # dahili IMU sim
        self.ekf = ekf              # dahili EKF

        # çalışma durumu
        self._seq = {"imu": 0, "ekf": 0}
        self._counter = 0

        # son imu/odo değerleri
        self.last_gz = 0.0
        self.last_v_odo = 0.0
        self.last_w_odo = 0.0

        # stop bayrağı
        self._stop = False

    # ---- default dummy ----
    def _default_sensor_cb(self, wc: int) -> List[int]:
        out: List[int] = []
        for _ in range(max(0, wc)):
            self._counter = (self._counter + 1) & 0xFFFF
            out.append(self._counter)
        return out

    # ---- scalar converter ----
    @staticmethod
    def _to_scalar(v) -> float:
        """Her türlü sayıyı (numpy array/tensor/list/tuple) güvenli float'a indirger."""
        try:
            import numpy as _np  # type: ignore
            if isinstance(v, _np.ndarray):
                if v.size == 0:
                    return 0.0
                return float(v.reshape(-1)[0])
        except Exception:
            pass
        try:
            if isinstance(v, (list, tuple)):
                return float(v[0]) if v else 0.0
            return float(v)
        except Exception:
            return 0.0

    @classmethod
    def _coerce_state_dict(cls, state: Union[dict, tuple, list, None]) -> dict:
        keys = ["x","y","z","vx","vy","vz","roll","pitch","yaw"]
        if isinstance(state, dict):
            return {k: cls._to_scalar(state.get(k, 0.0)) for k in keys}
        elif isinstance(state, (tuple, list)):
            vals = list(state)[:9] + [0.0] * max(0, 9 - len(state))
            return {k: cls._to_scalar(v) for k, v in zip(keys, vals)}
        else:
            return {k: 0.0 for k in keys}

    # ---- payload generators ----
    def _payload_imu(self) -> List[int]:
        sample = (
            self.sim.step(dt=0.02)  # sim varsa
            if self.sim and hasattr(self.sim, "step")
            else {
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "p": 0.0, "q": 0.0, "r": 0.0,
                "ax": 0.0, "ay": 0.0, "az": 0.0,
                "temp_c": 25.0,
            }
        )
        # son değerleri EKF için sakla (gerekirse)
        self.last_gz = float(sample.get("r", 0.0))   # varsayım: gz = r
        self.last_v_odo = self._to_scalar(sample.get("v_odo", 0.0))
        self.last_w_odo = self._to_scalar(sample.get("w_odo", 0.0))

        self._seq["imu"] = (self._seq["imu"] + 1) & 0xFFFF
        return pack_imu_words(sample, self._seq["imu"])

    def _payload_ekf(self) -> List[int]:
        """
        EKF durum vektörü: X=[x, y, psi, b_g, v]
        1553 EKF frame alanları: (x,y,z, vx,vy,vz, roll,pitch,yaw)
        Eşleme: z=0; vx=v; vy=vz=0; roll=pitch=0; yaw=psi
        """
        if self.ekf is not None and hasattr(self.ekf, "X"):
            X = self.ekf.X
        else:
            X = [0.0, 0.0, 0.0, 0.0, 0.0]

        def _sc(v):
            try:
                return float(v)
            except Exception:
                return 0.0

        state = {
            "x":    _sc(X[0]),
            "y":    _sc(X[1]),
            "z":    0.0,
            "vx":   _sc(X[4]),
            "vy":   0.0,
            "vz":   0.0,
            "roll": 0.0,
            "pitch":0.0,
            "yaw":  _sc(X[2]),
        }
        self._seq["ekf"] = (self._seq["ekf"] + 1) & 0xFFFF
        return pack_ekf_words(state, self._seq["ekf"])

    # ---- send/recv helpers ----
    def _send_status(self, bits: int = 0):
        self.bus.rt_send([make_word(SYNC_STATUS, make_status_field(self.addr, bits))])

    def _send_data_block(self, data_words: List[int]):
        packet = [make_word(SYNC_DATA, (w & 0xFFFF)) for w in data_words]
        if packet:
            self.bus.rt_send(packet)

    # ---- receive helpers ----
    def _normalize_recv_object(self, obj) -> List[int]:
        """bus.rt_recv(...) bazen tek kelime (int), bazen list[int] döndürebilir."""
        if not obj:
            return []
        if isinstance(obj, list):
            return obj
        return [obj]

    def _recv_n_data(self, n: int, timeout: float = 1.0) -> Optional[List[int]]:
        """BC -> DATA akışında n kelime topla; bus rt_recv list de döndürebilir."""
        deadline = time.time() + timeout
        collected: List[int] = []
        while time.time() < deadline and len(collected) < n:
            obj = self.bus.rt_recv(timeout=timeout)
            words = self._normalize_recv_object(obj)
            for w in words:
                sync, field = unpack_word(w)
                if sync != SYNC_DATA:
                    continue
                collected.append(field & 0xFFFF)
                if len(collected) >= n:
                    return collected
        return collected if len(collected) == n else None

    # ---- ana döngü ----
    def run_forever(self, sleep: float = 0.01):
        print(f"[RT] up. addr=0x{self.addr:02X}")
        self._stop = getattr(self, "_stop", False)
        while not self._stop:
            obj = self.bus.rt_recv(timeout=0.5)
            if not obj:
                time.sleep(sleep); continue

            # Bus bazı durumlarda bir seferde 'list[int]' döndürebilir.
            words = self._normalize_recv_object(obj)

            for w in words:
                sync, field = unpack_word(w)
                if sync != SYNC_CMD:
                    continue

                rt, tr, sa, wc = parse_command_field(field)
                if rt != self.addr:
                    continue

                # Transmit: RT -> STATUS, RT -> DATA
                if tr == 1:
                    self._send_status(0)
                    if sa == SA_IMU:
                        payload = self._payload_imu()
                    elif sa == SA_EKF:
                        payload = self._payload_ekf()
                    else:
                        payload = []
                    count = wc if wc > 0 else FRAME_WORDS
                    self._send_data_block(payload[:count])

                # Receive: BC -> DATA (wc), RT -> STATUS
                else:
                    count = wc if wc > 0 else FRAME_WORDS
                    rx = self._recv_n_data(count, timeout=1.0)
                    self._send_status(0 if rx is not None else 0x01)

            if sleep > 0:
                time.sleep(sleep)

    def stop(self):
        """Temiz kapanış için durdurma bayrağı."""
        self._stop = True
