import numpy as np
import ppsignals

def poly_refine(in_data, poly_est):

    N = len(data)

    if N > 200:
        filt_len = 9
    elif N < 70:
        filt_len = 5
    else:
        filt_len = 7

    if not np.iscomplexobj(in_data):
        sig = ppsignals.phase_to_sig(in_data)
    else:
        sig = in_data
    sig_est = ppsignals.poly_to_sig(poly_est, N)
    sig_diff = sig * np.conj(sig_est)

    if N >= filt_len:
        sig_diff = np.convolve(sig_diff, np.ones(filt_len), mode='same')

    
    

