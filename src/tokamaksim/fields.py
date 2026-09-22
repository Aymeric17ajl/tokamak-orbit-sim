#fields
import numpy as np 

R0=3.0    # grand rayon

a=1.0     # petit rayon

B0=2.0    # champ toroidal au centre

q=2.0     # facteur de sécurité


def cartesian_to_toroidal(x, y, z):
    phi = np.arctan2 (y, x)
    R = np.sqrt (x**2 + y**2)
    r = np.sqrt ((R-R0)**2 + z**2)
    theta = np.arctan2 (z , (R-R0))
    return ( r , theta , phi )


def B_field(position):
    x , y ,z = position
    r , theta , phi = cartesian_to_toroidal(x, y, z)


    R = R0 + r*np.cos(theta)
    B_phi = B0 * (R0)/R 
    B_theta = (r/q) * (B0/R0)

    direction_phi = np.array([-np.sin (phi), np.cos(phi), 0])
    direction_theta = np.array([-np.sin(theta)*np.cos(phi) , -np.sin(theta)*np.sin(phi) , np.cos(theta)])

    return B_phi * direction_phi + B_theta * direction_theta