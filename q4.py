import numpy as np

dx = 0.1
dt = 0.1
x = np.arange(0, 1+dx, dx)
nx = len(x)
nt = 101  # up to t = 10

p = np.zeros((nx, nt))
c = dt / dx

# Initial condition
p[:, 0] = np.cos(2 * np.pi * x)
p[0, :] = 1
p[-1, :] = 2

# First time step using initial velocity
dpdt = 2 * np.pi * np.sin(2 * np.pi * x)
p[1:-1, 1] = p[1:-1, 0] + dt * dpdt[1:-1] + 0.5 * c**2 * (p[2:, 0] - 2*p[1:-1, 0] + p[:-2, 0])

# Time stepping
for j in range(1, nt-1):
    p[1:-1, j+1] = 2*p[1:-1, j] - p[1:-1, j-1] + c**2 * (p[2:, j] - 2*p[1:-1, j] + p[:-2, j])

# Output samples
for j in range(0, nt, 10):
    print(f"t={j*dt:.1f}, p(x=0.5) = {p[int(nx/2), j]:.3f}")
