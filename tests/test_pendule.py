#test_pendule

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tokamaksim.integrators import rk4_step

# Physical constants for the linear pendulum
g: float = 9.81
L: float = 1.0
omega0: float = np.sqrt(g / L)  # Natural frequency

def pendulum_deriv(t: float, y: np.ndarray) -> np.ndarray:
    """Linear pendulum equations of motion: dtheta/dt = omega, domega/dt = -(g/L)*theta."""
    theta = y[0]
    omega = y[1]
     
    return np.array([omega, -(g / L) * theta])

def test_pendulum_small_oscillations_validation():
    """
    Validate the RK4 integrator by comparing the linear numerical solution
    against the analytical solution (cos).
    """
    theta0 = 0.1  # Initial displacement
    omega0_start = 0.0
    y = np.array([theta0, omega0_start])
    
    t = 0.0
    h = 0.01
    n_steps = 100
    
    # Evolve the linear system using my generic RK4 solver
    for _ in range(n_steps):
        y = rk4_step(pendulum_deriv, y, t, h)
        t += h
        
    # Analytical solution: theta(t) = theta0 * cos(omega0 * t)
    theta_analytical = theta0 * np.cos(omega0 * t)
    
    # Compare the numerical theta with the theoretical analytical theta
    absolute_error = abs(y[0] - theta_analytical)
    
    # Strict tolerance boundary for a perfectly linear model
    tolerance = 1e-6
    
    assert absolute_error < tolerance, f"Pendulum drift against analytical solution: error is {absolute_error}"

