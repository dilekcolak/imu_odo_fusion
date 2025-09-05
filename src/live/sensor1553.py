# sensor1553.py — IMU/INS ve EKF verilerini 1553 DATA kelimelerine paketleme/çözme



SA_IMU = 2

SA_EKF = 3



FRAME_ID_IMU = 0xA55A

FRAME_ID_EKF = 0xA55B



# 1(header) + 1(seq) + 3(angles) + 3(rates) + 3(acc) + 1(temp) + 1(resv) + 1(cksum)

FRAME_WORDS = 14





def csum16(words):

    """Basit 16‑bit toplam checksum (mod 65536)."""

    return sum(words) & 0xFFFF





def _sat_i16(x):

    if x > 32767:

        return 32767

    if x < -32768:

        return -32768

    return x





def _u16(x):

    return x & 0xFFFF





def _i16_to_signed(u):

    """16-bit two's complement -> Python int"""

    if u & 0x8000:

        return -((~u & 0xFFFF) + 1)

    return u





# ---------------- IMU frame ---------------- #



def pack_imu_words(sample: dict, seq: int) -> list:

    """

    Dönüş: 14 kelimelik çerçeve (16-bit)

    Ölçekler:

      angles [rad]   -> millirad (1e-3 rad)

      rates  [rad/s] -> millirad/s

      acc    [m/s^2] -> mg

      temp   [°C]    -> centi-degree (0.01°C)

    """

    roll_mrad = _sat_i16(int(round(sample["roll"] * 1000)))

    pitch_mrad = _sat_i16(int(round(sample["pitch"] * 1000)))

    yaw_mrad = _sat_i16(int(round(sample["yaw"] * 1000)))



    p_mrads = _sat_i16(int(round(sample["p"] * 1000)))

    q_mrads = _sat_i16(int(round(sample["q"] * 1000)))

    r_mrads = _sat_i16(int(round(sample["r"] * 1000)))



    g = 9.80665

    ax_mg = _sat_i16(int(round(sample["ax"] / g * 1000)))

    ay_mg = _sat_i16(int(round(sample["ay"] / g * 1000)))

    az_mg = _sat_i16(int(round(sample["az"] / g * 1000)))



    temp_centi = _sat_i16(int(round(sample["temp_c"] * 100)))



    words = [

        _u16(FRAME_ID_IMU),  # 0: header

        _u16(seq),  # 1: seq

        _u16(roll_mrad & 0xFFFF),  # 2

        _u16(pitch_mrad & 0xFFFF),  # 3

        _u16(yaw_mrad & 0xFFFF),  # 4

        _u16(p_mrads & 0xFFFF),  # 5

        _u16(q_mrads & 0xFFFF),  # 6

        _u16(r_mrads & 0xFFFF),  # 7

        _u16(ax_mg & 0xFFFF),  # 8

        _u16(ay_mg & 0xFFFF),  # 9

        _u16(az_mg & 0xFFFF),  # 10

        _u16(temp_centi & 0xFFFF),  # 11

        0x0000,  # 12: reserved

        0x0000,  # 13: checksum (doldurulacak)

    ]

    words[-1] = _u16(csum16(words[:-1]))

    return words





def unpack_imu_words(words: list) -> dict:

    """14 kelimelik IMU çerçeveyi çözer, mühendislik birimlerine geri çevirir."""

    if len(words) != FRAME_WORDS:

        raise ValueError("Frame length mismatch")

    if words[0] != FRAME_ID_IMU:

        raise ValueError("Bad IMU frame header")

    if words[-1] != csum16(words[:-1]):

        raise ValueError("Checksum mismatch")



    seq = words[1]



    roll = _i16_to_signed(words[2]) / 1000.0

    pitch = _i16_to_signed(words[3]) / 1000.0

    yaw = _i16_to_signed(words[4]) / 1000.0



    p = _i16_to_signed(words[5]) / 1000.0

    q = _i16_to_signed(words[6]) / 1000.0

    r = _i16_to_signed(words[7]) / 1000.0



    g = 9.80665

    ax = _i16_to_signed(words[8]) / 1000.0 * g

    ay = _i16_to_signed(words[9]) / 1000.0 * g

    az = _i16_to_signed(words[10]) / 1000.0 * g



    temp_c = _i16_to_signed(words[11]) / 100.0



    return {

        "seq": seq,

        "roll": roll,

        "pitch": pitch,

        "yaw": yaw,

        "p": p,

        "q": q,

        "r": r,

        "ax": ax,

        "ay": ay,

        "az": az,

        "temp_c": temp_c,

    }





# ---------------- EKF frame ---------------- #



def pack_ekf_words(state: dict, seq: int) -> list:

    """

    state ör: {

      "x": m, "y": m, "z": m,

      "vx": m/s, "vy": m/s, "vz": m/s,

      "roll": rad, "pitch": rad, "yaw": rad

    }

    Ölçekler: m, m/s ve rad için 1e-3 (milimetre, mm/s, millirad)

    """



    def fx(v, s):

        return _sat_i16(int(round(v / s))) & 0xFFFF



    w = [0] * FRAME_WORDS

    w[0] = _u16(FRAME_ID_EKF)  # header

    w[1] = _u16(seq)  # seq

    w[2] = fx(state["x"], 1e-3)

    w[3] = fx(state["y"], 1e-3)

    w[4] = fx(state["z"], 1e-3)

    w[5] = fx(state["vx"], 1e-3)

    w[6] = fx(state["vy"], 1e-3)

    w[7] = fx(state["vz"], 1e-3)

    w[8] = fx(state["roll"], 1e-3)

    w[9] = fx(state["pitch"], 1e-3)

    w[10] = fx(state["yaw"], 1e-3)

    w[11] = 0x0000  # reserved (flags/quality vb. için)

    w[12] = 0x0000  # reserved

    w[13] = 0x0000  # checksum placeholder

    w[13] = _u16(csum16(w[:-1]))

    return w





def unpack_ekf_words(words: list) -> dict:

    if len(words) != FRAME_WORDS:

        raise ValueError("Frame length mismatch")

    if words[0] != FRAME_ID_EKF:

        raise ValueError("Bad EKF frame header")

    if words[-1] != csum16(words[:-1]):

        raise ValueError("Checksum mismatch")



    seq = words[1]

    x = _i16_to_signed(words[2]) * 1e-3

    y = _i16_to_signed(words[3]) * 1e-3

    z = _i16_to_signed(words[4]) * 1e-3

    vx = _i16_to_signed(words[5]) * 1e-3

    vy = _i16_to_signed(words[6]) * 1e-3

    vz = _i16_to_signed(words[7]) * 1e-3

    roll = _i16_to_signed(words[8]) * 1e-3

    pitch = _i16_to_signed(words[9]) * 1e-3

    yaw = _i16_to_signed(words[10]) * 1e-3



    return {

        "seq": seq,

        "x": x,

        "y": y,

        "z": z,

        "vx": vx,

        "vy": vy,

        "vz": vz,

        "roll": roll,

        "pitch": pitch,

        "yaw": yaw,

    }

