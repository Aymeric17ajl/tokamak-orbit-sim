#test-pendule
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import numpy as np
from tokamaksim.integrators import rk4_step


g=9.81
L=1

def pendulum_deriv(t,y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g/L) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])


#conditions de départ
y0=np.array([0.02,0.0])
t0 = 0.0
h = 0.01
n_steps = 500

t = t0
y = y0
for i in range (n_steps):
    y = rk4_step(pendulum_deriv , y , t , h)
    t = t + h 

print ("Angle final simulé :",y[0])


#validation 
omega = np.sqrt(g/L)
theta_exact = y0[0]*np.cos(n_steps *h *omega)
erreur_relative = abs ((theta_exact - y[0]))/abs(theta_exact)
print("Angle exact : " , theta_exact )
print("Erreur relative : ", erreur_relative)