#visualizeparticule

import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

import numpy as np
import matplotlib.pyplot as plt

from tokamaksim.integrators import rk4_step
from tokamaksim.particle import particle_deriv
from tokamaksim.fields import  R0

position0 = np.array([R0, 0.0, 0.0])
v_perp = 10**5
y0 = np.concatenate([position0, np.array([v_perp, 0.0, 0.0])])

# Test de convergence du rayon de Larmor

T = 2e-6  # durée totale de simulation

for h in [1e-9, 5e-10, 2.5e-10]:
    n_steps = int(T / h)

    t = 0.0
    y = y0.copy()
    positions = [y[:3].copy()]
    velocities = [y[3:6].copy()]
    
    for _ in range(n_steps):
        y = rk4_step(particle_deriv, y, t, h)
        t += h
        positions.append(y[:3].copy())
        velocities.append(y[3:6].copy())
   
    
    positions = np.array(positions)
    velocities = np.array(velocities)

    speeds = np.linalg.norm(velocities, axis=1)
    error_v = np.max(np.abs(speeds - speeds[0])) / speeds[0]

    print(f"Erreur relative maximale sur la vitesse : {error_v:.3e}")
    xz = positions[:, [0, 2]]

    centre = xz.mean(axis=0)
    distances = np.linalg.norm(xz - centre, axis=1)

    rayon_moyen = distances.mean()
    ecart = (distances.max() - distances.min()) / rayon_moyen

    print(f"\nh = {h:.2e} s")
    print(f"Rayon moyen : {rayon_moyen:.8e} m")
    print(f"Écart relatif : {100*ecart:.4f} %")
