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

h = 10**-9
n_steps = 2000

t = 0.0 
y = y0
positions = [y[0:3].copy()]

for i in range (n_steps):
    y = rk4_step(particle_deriv, y, t, h)
    t+=h
    positions.append(y[0:3].copy())

positions = np.array(positions)

centre = positions[:, [0,2]].mean(axis=0)  # centre approximatif (colonnes x et z)
distances = np.linalg.norm(positions[:, [0,2]] - centre, axis=1)
print("rayon min :", distances.min())
print("rayon max :", distances.max())
print("écart relatif :", (distances.max()-distances.min())/distances.mean())

plt.figure(figsize=(6,6))
plt.plot(positions[:, 0],positions[:, 2])
plt.xlabel("x")
plt.ylabel("z")
plt.title("orbite de larmor vue dans le plan x-z")
plt.axis("equal")
plt.savefig(os.path.join(os.path.dirname(__file__),"orbite_larmor.png"))
print("Figure enregisstrée :orbite_larmor.png ")
