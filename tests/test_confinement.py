#Testconfinement

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from tokamaksim.integrators import rk4_step
from tokamaksim.confinement import confinement_deriv

def test_test_plasma_confinement_steady_state ():
    """ Verify yhat yhe 0D plasma power balance solver converges to a stable , physically reallistic steady-state temperature and density near 5 keV.
    """
    #Initial conditions (Ignition start )
    n0 = 10**19
    T0 = 0.5 
    y = np.array ([n0, T0])

    t = 0.0
    h = 0.01
    n_steps = 300 #Simulate 3 seconds to match steady-state 

    #Envolve the 0D coupled system using the generic RK4 engine
    
    for _ in range(n_steps):
        y = rk4_step(confinement_deriv, y, t, h)
        t += h
    final_density, final_temperature = y 

    #1. Assert physical consistency (no negative density or temperature)
    assert final_density > 0, f"Density became non-physical: {final_density}"
    assert final_temperature > 0, f"Temperature dropped below zero: {final_temperature}" 
    
    
    #2. Assert convergence to your observed physical steady-state value
    #Cheks that temperature matches our verified 5.18 keV run within a small tolerance
    assert abs(final_temperature - 5.18) < 10 ** -1 , f"Abnormal steady-state temperature: {final_temperature} keV"