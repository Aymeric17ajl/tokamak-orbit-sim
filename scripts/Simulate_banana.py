#Simulate_banana
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np 
import matplotlib.pyplot as plt
from tokamaksim.integrators import rk4_step
from tokamaksim.particle import particle_deriv
from tokamaksim.fields import B_field, R0, a


q , m = 1.602 * 10 ** - 19 , 1.67 * 10 ** -27
v_total = 2*10**6  # vitesse de la particule

#position de depart
x0 = R0 + 0.3*a
y0 = 0.0
z0 = 0.0
position0 = np.array([x0,y0,z0])

#Champ B au point de depart 
B_local = B_field(position0)
B_norm = np.linalg.norm(B_local)
b_hat = B_local / B_norm

#Constrution de la vitesse (pitch-angle)
def make_velocity (xi) :
    v_par = xi * v_total
    v_perp = np.sqrt (1- xi ** 2 )* v_total 

    #vecteur perpendiculaire a B
    perp = np.cross (b_hat , np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(perp) < 10 ** -10 :
        perp = np.cross (b_hat , np.array([1.0, 0.0, 0.0]))
    perp = perp / np.linalg.norm(perp)

    return v_par * b_hat + v_perp * perp

#Simulation
#On teste deux pitch-angles : 0.9(passing) et 0.3 (trapped/banana)

for xi in  [0.9 , 0.3] :
    print("\n" + "="*50)
    print (f"Simulation en cours avec xi = {xi}...")
    print("="*50)


    
    v0 = make_velocity(xi)
    y_init = np.concatenate ([position0, v0])

    h = 3 * 10**-10
    n_steps = 320000
    
    t = 0.0
    positions = [y_init[0:3].copy()]

    y = y_init.copy()
    
    for _ in range(n_steps):
        y = rk4_step(particle_deriv, y, t, h)
        t += h
        positions.append(y[0:3].copy())

    print("Position initiale :", y_init[0:3])
    print("Position finale :", y[0:3])
    print("Déplacement total :", np.linalg.norm(y[0:3] - y_init[0:3]))
    
    v_final = y[3:6]
    erreur_vitesse = abs(np.linalg.norm(v_final) - v_total) / v_total

    print("Vitesse initiale :", v_total, "m/s")
    print("Vitesse finale :", np.linalg.norm(v_final), "m/s")
    print("Erreur relative sur la vitesse :", erreur_vitesse)

#Conversion en poloidal (R,Z)
    positions = np.array(positions)
    R_traj = np.sqrt(positions[:,0]**2 + positions[:,1]**2)
    Z_traj = positions[:,2]

    # Tracé du graphique
    plt.figure(figsize=(8,8))
    plt.plot(R_traj, Z_traj, linewidth=1, label=f'Trajectoire (xi = {xi})', color='blue')

    # Tracé du bord du plasma
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(R0 + a*np.cos(theta), a*np.sin(theta), 'r--', linewidth=1.5, label='Bord plasma')


    plt.xlabel('R [m]' , fontsize=12)
    plt.ylabel('Z [m]', fontsize=12)
    plt.title(f'Orbite poloïdale — Pitch-angle xi = {xi}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', framealpha=0.9)

    
    plt.axis('equal')
    plt.xlim(R0 - a - 0.1, R0 + a + 0.1)
    plt.ylim(-a -0.2, a + 0.1)
    
    
    # Sauvegarde
    filepath = os.path.join(os.path.dirname(__file__), f'orbit_xi_{xi}.png')
    plt.savefig(filepath, dpi=150,  bbox_inches='tight')
    plt.show()

print("Simulation terminée !")



