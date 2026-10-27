#test_integrators

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tokamaksim.integrators import rk4_step

def f_exponential_decay(t, y):
    """
    Definition of the ODE system: dy/dt = -y
    Analytical solution is y(t) = y0 * exp(-t)
    """
    return -y

def test_rk4_mathematical_precision():
    """
    Verify the strict mathematical accuracy of the generic RK4 integrator.
    Tests if the numerical solution matches the analytical solution.
    """
    t0 = 0.0
    y0 = np.array([1.0])  # Initial condition: exp(0) = 1
    h = 0.1               # Time step
    
    # 1. Compute numerical solution using my generic RK4 step solver
    y_numerical = rk4_step(f_exponential_decay, y0, t0, h)
    
    # 2. Compute the exact analytical solution at the next time step (t0 + h)
    y_exact = np.exp(-(t0 + h))
    
    # 3. Calculate the absolute error between the two results
    absolute_error = abs(y_numerical - y_exact)
    
    # Strict mathematical tolerance for a 4th-order scheme with h=0.1
    # Local truncation error should be of order O(h^5)
    rk4_tolerance = 1e-6
    
    # Assert checks if the condition is true. If false, the automated test fails.
    assert absolute_error < rk4_tolerance, f"RK4 integrator mathematical drift detected: error is {absolute_error}"
