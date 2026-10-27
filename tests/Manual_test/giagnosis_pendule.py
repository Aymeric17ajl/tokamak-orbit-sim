#test-pendule
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


import numpy as np
from tokamaksim.integrators import rk4_step

#Comparaison des erreurs

g=9.81
L=1

def pendulum_deriv(t,y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g/L) * theta
    return np.array([dtheta_dt, domega_dt])


#conditions de départ
y0 = np.array([0.02,0.0])
T = 5.0
t0 = 0.0

erreurs = []

for h in [0.08, 0.04, 0.02, 0.01]:
   n_steps = int(T/h)

   t = t0
   y = y0.copy()
   
   for i in range (n_steps):
      y = rk4_step(pendulum_deriv , y , t , h)
      t = t + h 




#validation 
   omega = np.sqrt(g/L)
   theta_exact = y0[0]*np.cos(n_steps *h *omega)
   erreur = abs (theta_exact - y[0])
   erreurs.append(erreur)


   erreur_relative = erreur / abs(theta_exact)
   
   print("\nh=", h)
   print ("Angle final simulé :",y[0])
   print("Angle exact : " , theta_exact )
   print("Erreur relative :", erreur_relative)
 
