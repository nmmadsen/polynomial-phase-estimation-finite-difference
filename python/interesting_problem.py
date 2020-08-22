"""
A demonstration of why we need integer indices for our times, otherwise
frequency aliasing will result in distortions to the phase
"""

import numpy as np


#pylint: disable=C0103
N = 256
poly_est1 = np.array([-5.62730152e-07, 2.86880785e-04,
                      5.00110996e-01, 0.00000000e+00])
poly_est2 = np.array([-5.62730152e-07, 2.86880785e-04,
                      5.00110996e-01-1, 0.00000000e+00])
# n = np.arange(N)-int((N-1)/2)
n = np.arange(N)-(N-1)/2
phase1 = np.polyval(poly_est1, n)
phase2 = np.polyval(poly_est2, n)
sig_est1 = np.exp(2j*np.pi*phase1)
sig_est2 = np.exp(2j*np.pi*phase2)
# sig_dechirp = sig * np.conj(sig_est)
# print(np.angle(sig_dechirp)/2/np.pi)
print(np.max(np.abs(sig_est1-sig_est2)))
