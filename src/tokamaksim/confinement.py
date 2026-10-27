#confinement

import numpy as np 

# Global physical constant parameters fornthe 0D power balance

P_ext : float = 5 * 10 ** 4    #External heating power in Watts/m^3
tau_E : float = 0.5            #energy confinement time in seconds
S_source : float = 10 ** 19    #Particle fueling source rate (particules/m^3/second)
tau_p : float = 1.0            #Particle confinment time in seconds

def confinement_deriv (t: float, y: np.ndarray)-> np.ndarray :

    """Calculate the time derivates for the 0D plasma power balance .
       Tracks the coupled evolution of plasma density (n) and temperature (T).
       
       Parameters : 
       -------------
       t: float 
          current simulation time (seconds).
        y: np.ndarray
           2nd plasma state vector [n,T], where : 
             - n : Plasma density (particules / m^3)
             
             - T : Plasma (keV)   
        returns : 
        ---------
         np.ndarray 
              2D derivate vector [dn/dt, dT/dt]  
    """
    n, T = y 
    
    # 1.Particule Balance Equation: dn/dt = Source - Losses
    # Safety floor to avoid unphysical negative densities or divisions by zero  
    n = max (n, 10*10 )
    dn_dt = S_source - (n / tau_p)
    
    
    #2.Energy Power Balance Equation: dT/dt
    # Conversion factor: 1 keV = 1.602 * 10 **-16 Joules
    
    keV_to_joules = 1.602 * 10 ** - 16

    #Bremsstrahlung raduation loss (simplified theoretical scaling proportional to n^2 * sqrt(T)) 

    T_safe = max(T, 1.0e-3)
    C_brems = 5.35e-37 
    P_brems = C_brems * (n**2) * np.sqrt(T_safe)

    # Conduction/convection heat loss leakage through the magnetic cage
    P_transport = (3 * n * T_safe * keV_to_joules) / tau_E

    # Isolating dT/dt from the net power flow: dE/dt = P_ext - P_brems - P_transport
    dT_dt = (P_ext - P_brems) / (3 * n * keV_to_joules) - (T_safe / tau_E)

    return np.array([dn_dt, dT_dt])         


  