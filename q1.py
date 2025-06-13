import numpy as np
import matplotlib.pyplot as plt

# Grid setup
pi = np.pi
h = k = pi / 10  # Step size (π/10)
nx, ny = 11, 6    # number of grid points

u = np.zeros((nx, ny))

# Boundary conditions
x = np.linspace(0, pi, nx)
y = np.linspace(0, pi / 2, ny)

u[0, :] = np.cos(y)           # u(0, y)
u[-1, :] = -np.cos(y)         # u(π, y)
u[:, 0] = np.cos(x)           # u(x, 0)
u[:, -1] = 0                  # u(x, π/2)

# Iterative solution using Gauss-Seidel
for _ in range(5000):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])

# Print results
for j in range(ny):
    for i in range(nx):
        print(f"u({x[i]:.2f}, {y[j]:.2f}) = {u[i,j]:.5f}")
