import numpy as np
import ppsignals

#pylint: disable=C0103
def unwrap(phase, num_differences=1, center_phase=True):
    """
    Unwraps wrapped phase. Takes the difference and then sums it up.
    Can perform multiple phase differences to handle higher frequency
    signals, or can use center_phase to remove estimate and remove
    the average frequency and then add it back in.

    Tries to ensure that the signal is between -0.5 and 0.5 in the center of
    the signal, but naturally there is a little ambiguity when there is
    an even number of samples.

    >>> N = 50
    >>> poly = np.array([0.005, 0.2, 0.3])
    >>> n = ppsignals.sample_times(N)
    >>> x = np.polyval(poly, n)
    >>> x_wrapped = ppsignals.wrap_phase(x)
    >>> x_unwrapped = unwrap(x_wrapped)
    >>> np.allclose(x_unwrapped, x)
    True
    """
    cur_phase = phase.copy()

    for ii in range(num_differences):
        cur_phase[ii+1:] = ppsignals.phase_diff(cur_phase[ii:])

    if center_phase:
        cur_phase[num_differences:] = ppsignals.centered_phase(
            cur_phase[num_differences:])

    for ii in reversed(range(num_differences)):
        cur_phase[ii:] = np.cumsum(cur_phase[ii:])

    tstart = int(-ppsignals.default_tstart(len(phase)))
    cur_phase -= np.round(cur_phase[tstart])

    return cur_phase

def poly_est(phase, order, num_differences=1, center_phase=True):
    N = len(phase)
    unwrapped = unwrap(phase, num_differences, center_phase)
    tt = ppsignals.sample_times(N)
    poly = np.polyfit(tt, unwrapped, order)
    return poly

if __name__ == "__main__":
    import doctest
    doctest.testmod()
