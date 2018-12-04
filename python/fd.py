import ppsignals


def polyfit(phase, order, recursions=0, center_phase=True):
    phase_d = ppsignals.phase_diff(sig)
    if recursions > 0:
        poly_est = polyfit(phase_d)
    
