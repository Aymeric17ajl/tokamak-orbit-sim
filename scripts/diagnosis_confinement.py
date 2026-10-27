
#diagnosis_confinement

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from tokamaksim.integrators import rk4_step
from tokamaksim.confinement import confinement_deriv, tau_E

def run_0D_plasma_simulation ():
    """
    Simulate the time evolution of plasma density and temperature using the RK4 engine
    and evaluate the Lawson triple product criteria.
    """
    print("Starting 0D Plasma Confinement Simulation...")

    # Initial conditions: low density and nearly cold plasma (ignition start)
    n0 = 1.0e19  # particles/m3
    T0 = 0.5     # keV (~ 5 million degrees)
    y = np.array([n0, T0])

    # Time stepping parameters
    t = 0.0
    t_max = 3.0  # Simulate 3 seconds of plasma discharge
    h = 0.01     # Time step (10 milliseconds)
    
    # History buffers for visualization (isolating components to avoid numpy index mixing)
    time_history = [t]
    density_history = [y[0]]     
    temperature_history = [y[1]] 

    # Simulation loop using the generic RK4 step solver
    while t < t_max:
        y = rk4_step(confinement_deriv, y, t, h)
        t += h
        
        time_history.append(t)
        density_history.append(y[0])      # Store only density n
        temperature_history.append(y[1])  # Store only temperature T

    # Convert histories to numpy arrays for plotting
    time_history = np.array(time_history)
    density_history = np.array(density_history)
    temperature_history = np.array(temperature_history)

    # --- LAWSON CRITERION CALCULATIONS & REPORT ---
    final_density = density_history[-1]
    final_temperature = temperature_history[-1]
    
    # Calculate the Triple Product: n * T * tau_E
    triple_product = final_density * final_temperature * tau_E

    print("\n" + "="*40)
    print("       LAWSON CRITERION REPORT")
    print("="*40)
    print(f"Final Density (n)      : {final_density:.2e} m^-3")
    print(f"Final Temperature (T)  : {final_temperature:.2f} keV")
    print(f"Confinement Time (tau) : {tau_E:.2f} s")
    print(f"Calculated n*T*tau     : {triple_product:.2e} m^-3*keV*s")
    
    # Fusion ignition threshold benchmark (~ 3e21 m^-3*keV*s)
    ignition_threshold = 3.0e21
    if triple_product >= ignition_threshold:
        print("STATUS                 : IGNITION ACHIEVED! 🎉")
    else:
        print(f"STATUS                 : SUB-IGNITION (Needs {ignition_threshold/triple_product:.1f}x higher performance)")
    print("="*40 + "\n")

    # --- PLOTTING RESULTS ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Top plot: Density evolution
    ax1.plot(time_history, density_history / 1.0e19, color='blue', lw=2)
    ax1.set_ylabel('Density ($10^{19}$ m$^{-3}$)', color='blue')
    ax1.grid(True, linestyle='--')
    ax1.set_title('0D Tokamak Plasma Core Evolution (Ignition to Steady-State)', fontsize=12, fontweight='bold')

    # Bottom plot: Temperature evolution
    ax2.plot(time_history, temperature_history, color='red', lw=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Temperature (keV)', color='red')
    ax2.grid(True, linestyle='--')

    plt.tight_layout()
    
    # Save the figure to show it off in the README later
    output_path = os.path.join(os.path.dirname(__file__), 'plasma_0d_evolution.png')
    plt.savefig(output_path, dpi=150)
    print(f"Graph successfully saved at: {output_path}")
    plt.show()

if __name__ == "__main__":
    run_0D_plasma_simulation()