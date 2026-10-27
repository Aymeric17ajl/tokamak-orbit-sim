#RK4
from typing import Callable
import numpy as np





def rk4_step(f: Callable[[float, np.ndarray], np.ndarray], y: np.ndarray, t: float, h: float) -> np.ndarray:
    """
    Perform a single time step integration using the explicit 4th-order Runge-Kutta (RK4) method.

    Parameters:
    -----------
    f : Callable[[float, np.ndarray], np.ndarray]
        The right-hand side vector function of the ODE system, dy/dt = f(t, y).
        Must take a float (time) and a NumPy array (state vector) as inputs and return a NumPy array.
    y : np.ndarray
        The current state vector at time t.
    t : float
        The current time simulation value.
    h : float
        The integration time step size.

    Returns:
    --------
    np.ndarray
        The updated state vector at time t + h.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)