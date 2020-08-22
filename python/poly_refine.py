import numpy as np
import ppsignals
import basic

#pylint: disable=C0103

def poly_refine(sig, poly_est, filt_len=None, mag_weighting=1):
    order = len(poly_est)-1
    N = len(sig)
    n = ppsignals.sample_times(N)
    if filt_len is None:
        filt_len = int(np.log2(N)-4)*2+1

    sig_est = ppsignals.poly_to_sig(poly_est, N)
    sig_dechirp = sig * np.conj(sig_est)

    sig_dechirp = np.convolve(sig_dechirp, np.ones(filt_len), mode='valid')

    half_width = int(filt_len/2)
    if half_width >= 1:
        cur_n = n[half_width:-half_width]
    else:
        cur_n = n

    phase_dechirp = ppsignals.sig_to_phase(sig_dechirp)

    phase_dechirp_unwrapped = basic.unwrap(phase_dechirp)

    poly = np.polyfit(cur_n, phase_dechirp_unwrapped, order,
                      w=abs(sig_dechirp)**mag_weighting)
    return poly+poly_est


def poly_refine_iter(sig, poly_est, num_iters=3, mag_weighting=1):
    N = len(sig)
    filt_lens = int(np.log2(N)-4)*2+1 + np.arange(num_iters)*2
    for filt_len in filt_lens:
        poly_est = poly_refine(sig, poly_est, filt_len, mag_weighting)
    return poly_est
