#test_fields

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


import numpy as np
from tokamaksim.integrators import rk4_step
from tokamaksim.particle import particle_deriv, q, m
from tokamaksim.fields import B_field, R0, B0

def test_larmor_radius_quantitative_validation():
    """Quantitavely validate  the Lorentz force calculation by tracking a particule's gyro orbit.
        compares the observed numerical Larmor radius against the analytical theoretical formula."""

    # Initial position: on the magnetic axis where B magnitude equals B0
    
    position0 = np.array ([R0, 0.0, 0.0])
    v_perp = 10 ** 5 

    #Velocity vector chosen perpendiculqr to the magnetic field direction 
    
    y0 = np.concatenate([position0, np.array([v_perp, 0.0, 0.0])])
    h = 10** - 9 
    n_steps = 2000
    t = 0.0 
    y = y0 
    positions = [y[0:3].copy()]

    # Numerical integrations Loop
    for _ in range (n_steps):
        y = rk4_step(particle_deriv, y, t, h )
        t+=h
        positions.append(y[0:3].copy())

    positions = np.array(positions)

    #Theoretical Larmor radius calculation 

    r_larmor_theoretical = (m * v_perp )/ ( abs (q) * B0)

     # Observed numerical Larmor radius calculated from the maximum vertical displacement amplitude
    r_larmor_observed = (positions[:, 2].max() - positions[:, 2].min()) / 2

    # Relative error evaluation
    relative_error = abs(r_larmor_theoretical - r_larmor_observed) / r_larmor_theoretical

    # Scientific tolerance threshold: numerical drift must remain under 0.5%
    tolerance_threshold = 2 * 10 ** -2
    
    assert relative_error < tolerance_threshold, f"Larmor radius validation drift: relative error is {relative_error}"