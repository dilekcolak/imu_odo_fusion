# -*- coding: utf-8 -*-

"""

-bc1553.py — MIL-STD-1553 Bus Controller (BC) simülasyonu 
- IMU (SA_IMU) ve EKF (SA_EKF) için sabit 14-word (FRAME_WORDS) okuma akışları kullanılır.
"""



from time import time

from typing import List, Optional, Tuple



from bus1553 import (

    Bus1553, make_word, unpack_word,

    SYNC_CMD, SYNC_DATA, SYNC_STATUS,

)

from sensor1553 import (

    SA_IMU, SA_EKF,

    FRAME_WORDS,

    unpack_imu_words, unpack_ekf_words,

)


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





class BC1553:

    def __init__(self, bus: Bus1553):

        self.bus = bus



    def _send_command(self, rt: int, tr: int, sa: int, wc: int):

        cmd_field = make_command_field(rt, tr, sa, wc)

        self.bus.bc_send([make_word(SYNC_CMD, cmd_field)])



    def _send_data_block(self, data_words: List[int]):

        packet = [make_word(SYNC_DATA, (w & 0xFFFF)) for w in data_words]

        if packet:

            self.bus.bc_send(packet)



    def _recv_until(self, want_sync: int, n: int, timeout: float) -> Optional[List[int]]:

        deadline = time() + timeout

        collected: List[int] = []

        while time() < deadline:

            words = self.bus.bc_recv(timeout=0.05)

            if not words:

                continue

            for w in words:

                sync, field = unpack_word(w)

                if sync == want_sync:

                    collected.append(field)

                    if len(collected) >= n:

                        return collected

        return None



    def _recv_one_status(self, timeout: float) -> Optional[int]:

        fields = self._recv_until(SYNC_STATUS, 1, timeout)

        return fields[0] if fields else None



    def _recv_n_data(self, n: int, timeout: float) -> Optional[List[int]]:

        if n <= 0:

            return []

        return self._recv_until(SYNC_DATA, n, timeout)



    def tx_to_rt(self, rt: int, sa: int, data_words: List[int], timeout: float = 1.0) -> bool:

        #BC -> RT yazma: Command + Data, ardından Status bekle.

        wc = len(data_words) & 0x1F

        self._send_command(rt, tr=0, sa=sa, wc=wc)

        self._send_data_block(data_words)

        status = self._recv_one_status(timeout)

        if status is None:

            return False

        s_rt, s_bits = parse_status_field(status)

        return (s_rt == rt) and (s_bits == 0)



    def rx_from_rt(self, rt: int, sa: int, wc: int, timeout: float = 1.0) -> Optional[List[int]]:

        #RT -> BC okuma (mevcut API korunur).

        wc &= 0x1F

        self._send_command(rt, tr=1, sa=sa, wc=wc)

        status = self._recv_one_status(timeout)

        if status is None:

            return None

        s_rt, s_bits = parse_status_field(status)

        if (s_rt != rt) or (s_bits != 0):

            return None

        data = self._recv_n_data(n=wc if wc > 0 else 0, timeout=timeout)

        return data if (data is not None and len(data) == (wc if wc > 0 else 0)) else None




    def poll_imu(self, rt: int = 1, timeout: float = 0.05) -> Optional[dict]:

        words = self.rx_from_rt(rt=rt, sa=SA_IMU, wc=FRAME_WORDS, timeout=timeout)

        if not words:

            return None

        try:

            return unpack_imu_words(words)

        except Exception:

            return None



    def poll_ekf(self, rt: int = 1, timeout: float = 0.05) -> Optional[dict]:

        words = self.rx_from_rt(rt=rt, sa=SA_EKF, wc=FRAME_WORDS, timeout=timeout)

        if not words:

            return None

        try:

            return unpack_ekf_words(words)

        except Exception:

            return None



    def poll_once(self, rt: int = 1,

                  imu_timeout: float = 0.02,

                  ekf_timeout: float = 0.05) -> Tuple[Optional[dict], Optional[dict]]:

        #Bir turda IMU ve sonra EKF oku. Yoksa None döner.


        imu = self.poll_imu(rt=rt, timeout=imu_timeout)

        ekf = self.poll_ekf(rt=rt, timeout=ekf_timeout)

        return imu, ekf





if __name__ == "__main__":

    # Basit self-test (RT yoksa okuma başarısız döner)

    bus = Bus1553()

    bc = BC1553(bus)



    imu = bc.poll_imu(rt=1)

    ekf = bc.poll_ekf(rt=1)

    print("[SelfTest] IMU:", imu)

    print("[SelfTest] EKF:", ekf)

