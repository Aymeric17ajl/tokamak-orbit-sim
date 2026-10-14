#testparticule

import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

import numpy as np
from tokamaksim.integrators import rk4_step
from tokamaksim.particle import particle_deriv, q, m
from tokamaksim.fields import B_field, R0, B0

#Position de départ : sur l'axe magnétique ou' B vaut B0

position0 = np.array([R0, 0.0, 0.0])
v_perp = 10**5     

#Sur l'axe, B pointe selon y (vérifié à la brique 2)->on met la vitesse selon x, perpendiculaire
y0 = np.concatenate([position0, np.array([v_perp, 0.0, 0.0 ])])

h = 10**-9
n_steps = 2000

t=0.0
y = y0
positions = [y[0:3].copy()]

for i in range (n_steps):
    y = rk4_step(particle_deriv, y, t, h)
    t+=h
    positions.append(y[0:3].copy())

positions = np.array(positions)

# Rayon de Larmor théorique 
r_larmor_theorique = m*v_perp / ( abs(q) * B0 )

# Rayon observé : à partir de l'écart max en z (déplacement hors du plan de départ) 
r_larmor_observe = (positions[:,2].max() - positions[:,2].min())/2

print ("Rayon de Larmor théorique :", r_larmor_theorique)
print ("Rayon de Larmor observé :", r_larmor_observe)
print ("Erreur relative : ", abs(r_larmor_theorique - r_larmor_observe)/r_larmor_theorique)