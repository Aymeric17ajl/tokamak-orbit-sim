#visualize_field
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np 
import matplotlib.pyplot as plt
from tokamaksim.fields import B_field, R0, a

x_vals = np.linspace(R0 - a, R0 + a, 20 )
z_vals = np.linspace(-a, a, 20 )
X,Z = np.meshgrid (x_vals, z_vals)

Bx = np.zeros_like(X)
Bz = np.zeros_like(X)

for i in range (X.shape[0]):
    for j in range (X.shape[1]):
        x,z = X[i,j], Z[i,j]
        r = np.sqrt(((x-R0)**2 + z**2))
        if (r <= a) :
            B = B_field (np.array([x, 0.0 , z]))  
            Bx[i, j] = B[0]
            Bz[i, j] = B[2]


plt.figure (figsize=(6,6))
plt.quiver (X, Z, Bx, Bz)
theta_circle = np.linspace (0, 2*np.pi, 100)
plt.plot(R0 + a*np.cos(theta_circle), a*np.sin(theta_circle), 'r--' )
plt.xlabel ("X(=R à phi = 0 )")
plt.ylabel ("z")
plt.title ("Champ poloidal, coupe à phi = 0 ")
plt.axis('equal')
plt.savefig(os.path.join(os.path.dirname(__file__), "Champ_poloidal.png"))
print("Figure sauvegardée : Champ_poloidal.png")