#diagnosis_fields

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tokamaksim.fields import B_field, R0, a, B0, q 

# Un point sur l'axe magnétique : r=0, donc x=R0, y=0, z=0

position_axe = np.array([R0 , 0.0 ,0.0])
B_axe = B_field (position_axe) 

print ("Champ B sur l'axe magnétique :", B_axe)
print ("Norme de B:",np.linalg.norm(B_axe))
print ("B0 attentdu:", B0)


# Verification du champ toroidal
print("\nVerification du champ toroidal :")

for R in [R0 - a, R0, R0 + a]:
    position = np.array([R, 0.0, 0.0])
    B = B_field(position)
    B_attendu = B0 * R0 / R

    print("R =", R, "m")
    print("B toroidal attendu :", B_attendu, "T")
    print("B toroidal calcule :", B[1], "T")

# Verification du champ poloidal
print("\nVerification du champ poloidal :")

for r in [0.0, 0.5*a, a]:
    position = np.array([R0 + r, 0.0, 0.0])
    B = B_field(position)
    R_local = R0 + r
    B_attendu = (r/q) * (B0/R_local)

    print("r =", r, "m")
    print("B poloidal attendu :", B_attendu, "T")
    print("B poloidal calcule :", B[2], "T")