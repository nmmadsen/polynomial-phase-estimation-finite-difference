import numpy as np

import poly_refine
import ppsignals

#pylint: disable=C0103


def fin_diff_mat(order):
    """
    Matrix that will take the top <order> coefficients of a polynomial, and
    return the coefficients of the finite difference, f(x) -> f(x+1)-f(x),
    In the paper it is defined as f(x)-f(x-1) which results in a matrix
    with checkerboard negatives, and shifts the indices by 1

    >>> fin_diff_mat(4)
    array([[4., 0., 0., 0.],
           [6., 3., 0., 0.],
           [4., 3., 2., 0.],
           [1., 1., 1., 1.]])
    """
    from scipy.special import binom
    out_mat = np.zeros((order, order))
    for ii in range(order):
        for jj in range(ii+1):
            out_mat[ii, jj] = binom(order-jj, order-ii-1)
    return out_mat

def disc_polyder(poly):
    """
    Take the discrete derivative of a polynomial

    >>> import numpy as np
    >>> N = 100
    >>> n = np.arange(N) - (N-1)/2.
    >>> n_d = n[:-1]
    >>> poly = [0.005, 0.1, 0.3]
    >>> x = np.polyval(poly, n)
    >>> x_d = np.diff(x)
    >>> np.polyfit(n_d, x_d, 1)
    array([0.01 , 0.105])
    >>> disc_polyder(poly)
    array([0.01 , 0.105])
    
    """
    order = len(poly)-1
    polyder = np.dot(fin_diff_mat(order), poly[:order])
    return polyder


def disc_polyint(poly):
    """
    Take the discrete integral of a polynomial, puts in zero for constant term

    >>> import numpy as np
    >>> poly = [0.005, 0.1, 0.3]
    >>> poly_d = disc_polyder(poly)
    >>> disc_polyint(poly_d)
    array([0.005, 0.1  , 0.   ])
    """
    from scipy.linalg import solve_triangular
    order = len(poly)
    # it is important to use solve_triangular, other solvers don't give very
    # good answers and it is triangular after all, so solving is straightforward
    polyint = solve_triangular(fin_diff_mat(order), poly, lower=True)
    polyint = np.hstack((polyint, [0]))
    return polyint

def polyfit(sig, order, phase_differences=1, center_phase=True,
            filt_len=5, mag_weighting=1, poly_refine_iters=3):
    """
    Main routine for finite difference algorithm as detailed in
    "Finite-Difference Algorithm for Polynomial Phase Signal Parameter
    Estimation" by Madsen and Cao.  It also includes some modifications
    that will be explained in a paper that has been submitted for publication.
    Assumes unit spacing, from -(N-1)/2 to (N-1)/2

    It is a good algorithm for the following situation:
       1. Computational complexity is a concern
       2. The signal is long (>100 samples, I've done up to a million)
       3. The order of the signal is large (>2)
       4. The signal is Nyquist sampled

    Basic flow of the algorithm is as follows:
       1. Conjugate multiply adjacent samples to get phase differences
       2. Optional: Run a smoothing filter over phase differences
       3. Optional: Center the phase
       4. Perform polynomial fit on phase differences
       5. Convert the finite difference polynomial coefficients to estimates
          of the original coefficients
       6. Refine the estimate using O'Sheas method
      

    Inputs:
    sig: numpy array complex
        Complex samples of polynomial phase signal to be estimated
    order: int (>0)
        The order of the polynomial phase signal
    phase_differences: int (>0)
        How many phase differences to perform.  1 is generally recommended.
        If the signal spans almost the entire Nyquist range 2 might perform
        better, but it will also hurt your SNR threshold.
    center_phase: bool
        Whether to perform a rough frequency estimate and remove the frequency
        component prior to polynomial fit, then add it back in.  Generally,
        reduces the likelihood of wrapping problems.
    filt_len: int (odd)
        Length of rectangular filter used to smooth the phase
        differences.  To keep things a bit simpler it needs to be odd.
    mag_weighting: float (>= 0)
        Power to raise magnitudes to when doing a weighted polynomial fit.
        A mag weighting of 1 would weight each phase sample by the magnitude
        of the signal, 2 would be the magnitude squared etc.
    poly_refine_iters: int (>0)
        Number of times to repeat the polynomial refinement step.

    >>> N = 50
    >>> snr = 100
    >>> noise_var = 10.**(-0.1*snr)
    >>> the_poly = np.array([0.005, 0.2, 0.3])
    >>> order = len(the_poly)-1
    >>> sig = ppsignals.sig_gen(the_poly, N, noise_var=noise_var)
    >>> est_poly = polyfit(sig, order)
    >>> diff_poly = ppsignals.unalias_poly(the_poly - est_poly)
    >>> np.all(abs(diff_poly) < 1e-5)
    True
        
    """
    N = len(sig)
    n = ppsignals.sample_times(N)


    # Keep each phase difference of the signal
    sigs = [sig]
    for ii in range(phase_differences):
        sigs.append(sigs[-1][1:]*np.conj(sigs[-1][:-1]))
    # update indices
    cur_n = n[:-phase_differences]

    # filter signal
    filt_sig = np.convolve(sigs[-1], np.ones(filt_len), 'valid')
    half_width = int(filt_len/2)
    if half_width >= 1:
        cur_n = cur_n[half_width:-half_width]

    # get phase
    filt_phase = ppsignals.sig_to_phase(filt_sig)

    # center phase if you want to
    if center_phase:
        filt_phase = ppsignals.centered_phase(filt_phase)

    # fit the polynomial
    poly_est = np.polyfit(cur_n, filt_phase, order-phase_differences,
                          w=abs(filt_sig)**mag_weighting)


    # integrate the estimate refining along the way
    for ii in range(phase_differences):
        poly_est = disc_polyint(poly_est)
        poly_est = poly_refine.poly_refine_iter(
            sigs[-2-ii], poly_est, poly_refine_iters)
    

    poly_est = ppsignals.unalias_poly(poly_est)
    return poly_est
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
