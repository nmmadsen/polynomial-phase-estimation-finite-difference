import numpy as np
import ppsignals

def unwrap(phase, num_differences=1, center_phase=True):
    cur_phase = phase.copy()

    for ii in range(num_differences):
        cur_phase[ii+1:] = ppsignals.phase_diff(cur_phase[ii:])

    if center_phase:
        cur_phase[num_differences:] = ppsignals.centered_phase(
            cur_phase[num_differences:])

    for ii in reversed(range(num_differences)):
        cur_phase[ii:] = np.cumsum(cur_phase[ii:])
        
    return cur_phase
        

def poly_est(phase, order, num_differences=1, center_phase=True):
    unwrapped = unwrap(phase, num_differences, center_phase)
    poly_est = ppsignals
    
