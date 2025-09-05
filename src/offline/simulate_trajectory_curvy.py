import numpy as np

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def make_curvy_path(total_time=40.0, dt=0.05, v_mean=1.0, seed=1):
    """
    Unicycle kinematik ile organik, kıvrımlı rota üretir.
    Dönen: t, pos(N,2), heading(N,)
    """
    rng = np.random.default_rng(seed)
    N = int(total_time/dt)
    t = np.arange(N)*dt
    pos = np.zeros((N,2))
    psi = np.zeros(N)

    v = v_mean
    w = 0.0
    seg_left = 0.0

    for k in range(1, N):
        if seg_left <= 0.0:
            seg_left = rng.uniform(1.0, 3.0)   # yeni komut süresi
            v = v_mean * rng.uniform(0.8, 1.2)
            w = rng.uniform(-0.6, 0.6)        # rad/s dönüş hızı

        # son %20’de eve dönüş kontrolü
        if k > 0.8*N:
            to_home = -pos[k-1]
            desired = np.arctan2(to_home[1], to_home[0])
            err = wrap_pi(desired - psi[k-1])
            w = 0.8 * err

        psi[k] = wrap_pi(psi[k-1] + w*dt)
        pos[k,0] = pos[k-1,0] + v*dt*np.cos(psi[k])
        pos[k,1] = pos[k-1,1] + v*dt*np.sin(psi[k])

        seg_left -= dt

    return t, pos, psi

