#RK4
import numpy as np
def rk4_step(f,y,t,h):
    k1=f(t,y)
    k2=f(t+h/2,y+(h/2)*k1)
    k3=f(t+h/2,y+(h/2)*k2)
    k4= f(t+h,y+h*k3)
    return (y+(h/6)*(k1 + 2*k2 + 2*k3 + k4))