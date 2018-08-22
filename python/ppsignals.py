import numpy as np

def sig_to_phase(sig, radians=False):
    """
    """
    phase = np.angle(sig)
    if radians:
        return phase
    else:
        return phase/(2*np.pi)
    
    

def phase_to_sig(phase, radians=False):
    if radians:
        sig = np.exp(1j*phase)
    else:
        sig = np.exp(2j*np.pi*phase)
    return sig

def poly_to_sig(poly, x, radians=False):
    phase = 10

def poly_to_phase(poly, x, radians=False):
    
    
    
