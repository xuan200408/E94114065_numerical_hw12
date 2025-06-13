import numpy as np

def solve_forward():
    K = 0.1
    dr = 0.1
    dt = 0.5
    r = np.arange(0.5, 1.0 + dr, dr)
    nt = int(10 / dt) + 1
    nr = len(r)

    T = np.zeros((nr, nt))
    T[:, 0] = 200 * (r - 0.5)  # 初始條件

    for j in range(nt - 1):
        # r = 1: Dirichlet BC
        T[-1, j+1] = 100 + 40 * dt * (j + 1)

        # r = 0.5: Neumann BC: ∂T/∂r + 3T = 0 ⇒ T0 = T1 / (1 + 3*dr)
        T[0, j+1] = T[1, j] / (1 + 3 * dr)

        for i in range(1, nr - 1):
            term1 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dr**2
            term2 = (T[i+1, j] - T[i-1, j]) / (2 * r[i] * dr)
            T[i, j+1] = T[i, j] + dt * K * (term1 + term2)

    return T, r, dt


def solve_backward():
    K = 0.1
    dr = 0.1
    dt = 0.5
    r = np.arange(0.5, 1.0 + dr, dr)
    nt = int(10 / dt) + 1
    nr = len(r)
    lam = K * dt / dr**2

    T = np.zeros((nr, nt))
    T[:, 0] = 200 * (r - 0.5)

    for j in range(nt - 1):
        A = np.zeros((nr, nr))
        b = np.zeros(nr)

        # r = 1: Dirichlet BC
        A[-1, -1] = 1
        b[-1] = 100 + 40 * dt * (j + 1)

        # r = 0.5: Neumann BC
        A[0, 0] = -1 / dr + 3
        A[0, 1] = 1 / dr
        b[0] = 0

        for i in range(1, nr - 1):
            ri = r[i]
            A[i, i-1] = -lam * (1 - 0.5 * dr / ri)
            A[i, i]   = 1 + 2 * lam
            A[i, i+1] = -lam * (1 + 0.5 * dr / ri)
            b[i] = T[i, j]

        T[:, j+1] = np.linalg.solve(A, b)

    return T, r, dt


def solve_crank_nicolson():
    K = 0.1
    dr = 0.1
    dt = 0.5
    r = np.arange(0.5, 1.0 + dr, dr)
    nt = int(10 / dt) + 1
    nr = len(r)
    lam = K * dt / dr**2

    T = np.zeros((nr, nt))
    T[:, 0] = 200 * (r - 0.5)

    for j in range(nt - 1):
        A = np.zeros((nr, nr))
        B = np.zeros(nr)

        # r = 1: Dirichlet BC
        A[-1, -1] = 1
        B[-1] = 100 + 40 * dt * (j + 1)

        # r = 0.5: Neumann BC
        A[0, 0] = -1 / dr + 3
        A[0, 1] = 1 / dr
        B[0] = 0

        for i in range(1, nr - 1):
            ri = r[i]

            # A matrix (implicit)
            A[i, i-1] = -lam/2 * (1 - 0.5 * dr / ri)
            A[i, i]   = 1 + lam
            A[i, i+1] = -lam/2 * (1 + 0.5 * dr / ri)

            # B vector (explicit)
            term1 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dr**2
            term2 = (T[i+1, j] - T[i-1, j]) / (2 * ri * dr)
            B[i] = T[i, j] + dt/2 * K * (term1 + term2)

        T[:, j+1] = np.linalg.solve(A, B)

    return T, r, dt


def print_sample(T, r, dt, title=""):
    print(f"\n{title}")
    print(f"{'time':>5} {'r=0.5':>10} {'r=1.0':>10}")
    for j in range(T.shape[1]):
        print(f"{j*dt:5.2f} {T[0, j]:10.3f} {T[-1, j]:10.3f}")


# 執行三種方法並印出結果
T_fwd, r, dt = solve_forward()
T_bwd, _, _ = solve_backward()
T_cn,  _, _ = solve_crank_nicolson()

print_sample(T_fwd, r, dt, "Forward Difference")
print_sample(T_bwd, r, dt, "Backward Difference")
print_sample(T_cn,  r, dt, "Crank-Nicolson")
