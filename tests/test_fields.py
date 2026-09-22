#test_fields

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tokamaksim.fields import B_field, R0, B0 

# Un point sur l'axe magnétique : r=0, donc x=R0, y=0, z=0

position_axe = np.array([R0 , 0.0 ,0.0])
B_axe = B_field (position_axe) 

print ("Champ B sur l'axe magnétique :", B_axe)
print ("Norme de B:",np.linalg.norm(B_axe))
print ("B0 attentdu:", B0)