#particle
import numpy as np 
from .fields import B_field


# Physical constants for a proton 

q : float = 1.602 * 10**-19     # Proton charge (Coulombs)
m : float = 1.673 * 10**-27     # Proton mass (kilograms)

def particle_deriv(t: float, y: np.ndarray) -> np.ndarray:
    """
      Calculate the time derivates of the particle state vector (equation of motion).
      Implements the 3D Lorentz force equation :m * dv/dt = q * (v x B).

      Parameters : 
      -----------
      t : float
        Current simulation time (seconds). Unused as the B-field is stationary.
       y : np.ndarray
        6D particle state vector [x, y, z, vx, vy, vz], where:
        - [x, y, z]: Cartesian position coordinates (meters)
        - [vx, vy, vz]: Cartesian velocity components (m/s)

     Returns:
    --------
    np.ndarray
        6D derivative state vector [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt], representing:
        - [vx, vy, vz]: Velocity vector components (m/s)
        - [ax, ay, az]: Acceleration vector components from the Lorentz force (m/s^2)
    """
    
    
    
    
    position = y[0:3]
    velocity = y[3:6]


    B = B_field(position)
    acceleration = (q/m) * np.cross (velocity, B)

    
    return (np.concatenate([velocity, acceleration]))