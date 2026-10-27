#fields
from typing import Tuple
import numpy as np 

# Global geometric and magnetic constants for the tokamak

R0: float = 3.0    # Major radius (meters)

a: float = 1.0     # Minor radius (meters)

B0: float = 2.0    # Toroidal magnetic field at the center axis (Teslas)

q: float = 2.0     # Safety factor


def cartesian_to_toroidal(x: float, y: float, z: float) -> Tuple[float, float, float]:
    
    """
    Convert 3D Cartesian coordinates to local toroidal coordinates.

    Parameters:
    -----------
    x : float
        Cartesian X coordinate (meters).
    y : float
        Cartesian Y coordinate (meters).
    z : float
        Cartesian Z coordinate (meters).

    Returns:
    --------
    Tuple[float, float, float]
        Local coordinates (r, theta, phi) where:
        - r: local minor radius (meters)
        - theta: poloidal angle (radians)
        - phi: toroidal angle (radians)
    """
    
    
    phi = np.arctan2 (y, x)
    R = np.sqrt (x**2 + y**2)
    r = np.sqrt ((R-R0)**2 + z**2)
    theta = np.arctan2 (z , (R-R0))
    return ( r , theta , phi )


def B_field(position: np.ndarray) -> np.ndarray:
    """
    Calculate the  magnetic field vector at a specific spatial point.

    Parameters:
    -----------
    position : np.ndarray
        A 3D Cartesian position vector [x, y, z] (meters).

    Returns:
    --------
    np.ndarray
        The 3D magnetic field vector [Bx, By, Bz] in Cartesian components (Teslas).
    """
   
   
   
    x , y ,z = position
    r , theta , phi = cartesian_to_toroidal(x, y, z)


    R = R0 + r*np.cos(theta)
    B_phi = B0 * (R0)/R 
    B_theta = (r/q) * (B0/R)

    direction_phi = np.array([-np.sin (phi), np.cos(phi), 0])
    direction_theta = np.array([-np.sin(theta)*np.cos(phi) , -np.sin(theta)*np.sin(phi) , np.cos(theta)])

    return B_phi * direction_phi + B_theta * direction_theta