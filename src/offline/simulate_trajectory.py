import numpy as np
import matplotlib.pyplot as plt

def make_cornered_path(total_time=60.0, dt=0.05):
    N = int(total_time/dt)
    pos = np.zeros((N, 2))
    heading = np.zeros(N)

    dirs = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    d = 0
    segment_length = 8.0
    steps = int(segment_length / (1.0*dt))

    for k in range(1, N):
        if k % steps == 0:
            turn = np.random.choice([-1, 0, +1])
            d = (d + turn) % 4
        pos[k] = pos[k-1] + dirs[d] * 1.0 * dt
        heading[k] = np.arctan2(dirs[d][1], dirs[d][0])

    t = np.arange(N) * dt
    return t, pos, heading

if __name__ == "__main__":
    t, pos, heading = make_concerned_path()
    print("Üretilen adım sayısı:", len(t))
    print("Son konum:", pos[-1])
    # Rota çizimi
    plt.figure(figsize=(6,6))
    plt.plot(pos[:,0], pos[:,1], label="Ground Truth Rota")
    plt.scatter(pos[0,0], pos[0,1], c="green", marker="o", label="Başlangıç")
    plt.scatter(pos[-1,0], pos[-1,1], c="red", marker="x", label="Bitiş")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.title("Köşeli Ground Truth Rota")
    plt.show()
