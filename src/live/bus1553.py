# -*- coding: utf-8 -*-

"""

bus1553.py — Çift yönlü MIL-STD-1553 “bus” simülasyonu + ayrıntılı logging



Env değişkenleri:

  BUS1553_LOG=1/0           -> paket loglarını aç/kapat (varsayılan: 1)

  BUS1553_LOG_LEVEL=DEBUG   -> log seviyesi (varsayılan: INFO)

  BUS1553_LOG_FILE=/path.log-> dosyaya da log yaz

"""



import os

import queue

import logging



# 1553 Word tipleri

SYNC_CMD    = 0b00

SYNC_DATA   = 0b01

SYNC_STATUS = 0b10



def _mk_logger():

    lvl_name = os.getenv("BUS1553_LOG_LEVEL", "INFO").upper()

    level = getattr(logging, lvl_name, logging.INFO)

    logger = logging.getLogger("bus1553")

    logger.setLevel(level)

    if not logger.handlers:

        sh = logging.StreamHandler()

        sh.setLevel(level)

        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

        logger.addHandler(sh)

        log_file = os.getenv("BUS1553_LOG_FILE")

        if log_file:

            fh = logging.FileHandler(log_file)

            fh.setLevel(level)

            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

            logger.addHandler(fh)

    return logger



LOGGER = _mk_logger()

VERBOSE = os.getenv("BUS1553_LOG", "1").lower() in ("1", "true", "on", "yes")




def make_word(sync, field):

    """16-bit field + 2 bit sync'i üstte taşıyan basit kelime paketi."""

    return (sync << 16) | (field & 0xFFFF)



def unpack_word(word):

    """Paketten sync ve 16-bit field'ı ayır."""

    sync = (word >> 16) & 0b11

    field = word & 0xFFFF

    return sync, field



def _sync_name(sync):

    return {SYNC_CMD: "CMD", SYNC_DATA: "DATA", SYNC_STATUS: "STATUS"}.get(sync, f"S{sync}")



# (Sadece LOG için) Command/Status alanlarını kaba çöz

def _parse_command_field(field):

    # [RT(5) | T/R(1) | SA(5) | WC(5)]

    rt  = (field >> 11) & 0x1F

    tr  = (field >> 10) & 0x01

    sa  = (field >> 5)  & 0x1F

    wc  = field & 0x1F

    return rt, tr, sa, wc



def _parse_status_field(field):

    # [RT(5) | Reserved(3)=0 | Bits(8)]  (basitleştirilmiş model)

    rt   = (field >> 11) & 0x1F

    bits = field & 0xFF

    return rt, bits



def _fmt_words(words):

    return "[" + ", ".join(hex(w) for w in words) + "]"



def _decode_words(words, direction):

    """

    direction: 'BC->RT' ya da 'RT->BC'

    Her kelime için anlamlı kısa bir açıklama üretir.

    """

    out = []

    for w in words:

        sync, field = unpack_word(w)

        sname = _sync_name(sync)

        if sync == SYNC_CMD:

            rt, tr, sa, wc = _parse_command_field(field)

            out.append(f"{sname}(rt={rt}, tr={tr}, sa={sa}, wc={wc})")

        elif sync == SYNC_STATUS:

            rt, bits = _parse_status_field(field)

            out.append(f"{sname}(rt={rt}, bits=0b{bits:08b})")

        else:

            out.append(f"{sname}(0x{field:04X})")

    return f"[BUS][{direction}] " + "  ".join(out)



# -------------------- Bus Sınıfı --------------------

class Bus1553:

    """

    Çift yönlü kuyruklar:

      - q_to_rt : BC'nin gönderdiği, RT'nin okuyacağı

      - q_to_bc : RT'nin gönderdiği, BC'nin okuyacağı

    """

    def __init__(self):

        self.q_to_rt = queue.Queue()

        self.q_to_bc = queue.Queue()



    # ---- BC perspektifi ----

    def bc_send(self, words):

        self.q_to_rt.put(words)



    def bc_recv(self, timeout=0.1):

        """BC'nin RT'den gelen kelimeleri alması"""

        try:

            words = self.q_to_bc.get(timeout=timeout)

            if VERBOSE:

                LOGGER.debug(_decode_words(words, "RT->BC"))

            return words

        except queue.Empty:

            return None



    # ---- RT perspektifi ----

    def rt_send(self, words):

        self.q_to_bc.put(words)



    def rt_recv(self, timeout=0.1):

        """RT'nin BC'den gelen kelimeleri alması"""

        try:

            words = self.q_to_rt.get(timeout=timeout)

            if VERBOSE:

                LOGGER.debug(_decode_words(words, "BC->RT"))

            return words

        except queue.Empty:

            return None


