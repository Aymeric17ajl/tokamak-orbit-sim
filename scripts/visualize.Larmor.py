# Test du mouvement de Larmor dans un champ magnetique uniforme

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from tokamaksim.integrators import rk4_step
from tokamaksim.particle import q, m

B = np.array([0.0, 0.0, 2.0])
v_perp = 10**5

# Rayon et periode theoriques
rayon_theorique = m * v_perp / (abs(q) * np.linalg.norm(B))
omega = abs(q) * np.linalg.norm(B) / m
periode = 2 * np.pi / omega

print("Rayon theorique :", rayon_theorique, "m")
print("Periode cyclotron :", periode, "s")

# Equation du mouvement dans le champ uniforme
def deriv_uniforme(t, y):
    position = y[0:3]
    velocity = y[3:6]
    acceleration = (q / m) * np.cross(velocity, B)
    return np.concatenate([velocity, acceleration])

# Conditions initiales
position0 = np.array([0.0, 0.0, 0.0])
velocity0 = np.array([v_perp, 0.0, 0.0])
y0 = np.concatenate([position0, velocity0])

# Integration avec plusieurs pas 
for division in [50, 100, 200]:
    h = periode / division
    n_steps = 5 * division
    t = 0.0
    y = y0.copy()

    positions = [y[:3].copy()]
    vitesses = [y[3:6].copy()]

    for i in range(n_steps):
       y = rk4_step(deriv_uniforme, y, t, h)
       t += h
       positions.append(y[:3].copy())
       vitesses.append(y[3:6].copy())

    positions = np.array(positions)
    vitesses = np.array(vitesses)

# Verification de la vitesse
    speeds = np.linalg.norm(vitesses, axis=1)
    erreur_vitesse = np.max(np.abs(speeds - v_perp)) / v_perp

#Vérification du rayon 
    centre = np.array([0.0, -rayon_theorique, 0.0])

    distances = np.linalg.norm(positions - centre, axis=1)

    rayon_numerique = distances.mean()

    erreur_rayon = abs(rayon_numerique - rayon_theorique) / rayon_theorique

    print("\nDivision :", division)
    print("Pas h :", h, "s")
    print("Rayon numerique :", rayon_numerique, "m")
    print("Erreur relative sur le rayon :", erreur_rayon)
    print("Erreur relative maximale sur la vitesse :", erreur_vitesse)

    # Visualisation de la trajectoire
    plt.plot(positions[:, 0], positions[:, 1],
             label="h = periode / " + str(division))

# Cercle theorique
angle = np.linspace(0, 2 * np.pi, 200)
plt.plot(
    rayon_theorique * np.cos(angle),
    -rayon_theorique + rayon_theorique * np.sin(angle),
    "--",
    label="Cercle theorique"
)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.legend()
plt.grid()
plt.show()

