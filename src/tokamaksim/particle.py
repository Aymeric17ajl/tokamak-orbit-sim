#particle
import numpy as np 
from.fields import B_field

q = 1.602 * 10**-19     # charge proton
m = 1.673 *10**-27      # masse proton 

def particle_deriv(t,y) :
    position = y[0:3]
    velocity = y[3:6]


    B = B_field(position)
    acceleration = (q/m) * np.cross (velocity, B)

    
     
    return (np.concatenate([velocity, acceleration]))