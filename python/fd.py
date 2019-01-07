import ppsignals


def polyfit(phase, order, recursions=0, center_phase=True):
    phase_d = ppsignals.phase_diff(phase)
    if recursions > 0:
        poly_est = polyfit(phase_d, order-1, recursions-1)
    else:
        poly_est = poly


    return poly_est
    
    
