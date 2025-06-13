import numpy as np

dr = 0.1
dtheta = np.pi / 30
r_vals = np.arange(0.5, 1.0 + dr, dr)
theta_vals = np.arange(0, np.pi/3 + dtheta, dtheta)
nr = len(r_vals)
ntheta = len(theta_vals)

T = np.zeros((nr, ntheta))

# Boundary conditions
T[0, :] = 50
T[-1, :] = 100
T[:, 0] = 0
T[:, -1] = 0

# Iterative solution
for _ in range(5000):
    for i in range(1, nr - 1):
        for j in range(1, ntheta - 1):
            r = r_vals[i]
            T_rr = (T[i+1,j] - 2*T[i,j] + T[i-1,j]) / dr**2
            T_r = (T[i+1,j] - T[i-1,j]) / (2*r*dr)
            T_theta = (T[i,j+1] - 2*T[i,j] + T[i,j-1]) / (r**2 * dtheta**2)
            T[i,j] = (T_rr + T_r + T_theta + T[i,j]) / 2  # Over-relaxation optional

# Print a sample result
for j in range(ntheta):
    print(f"theta = {theta_vals[j]:.2f}, T(r=0.5) = {T[0,j]:.2f}, T(r=1.0) = {T[-1,j]:.2f}")
