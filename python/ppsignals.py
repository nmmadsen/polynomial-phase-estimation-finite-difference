import numpy as np
#pylint: disable=C0103

def default_tstart(N, dt=1.):
    return -(N-1)/2. * dt

def sample_times(N, tstart=None, dt=1.):
    if tstart is None:
        tstart = default_tstart(N, dt)
    return np.arange(N)*dt+tstart

def sig_to_phase(sig):
    phase = np.angle(sig)
    return phase/(2*np.pi)

def phase_to_sig(phase):
    return np.exp(2j*np.pi*phase)

def poly_to_sig(poly, N, tstart=None, dt=1.):
    phase = poly_to_phase(poly, N, tstart, dt)
    return  phase_to_sig(phase)

def poly_to_phase(poly, N, tstart=None, dt=1.):
    tt = sample_times(N, tstart, dt)
    phase = np.polyval(poly, tt)
    phase = wrap_phase(phase)
    return phase

def unwrapped_phase_to_poly(phase, N, order, tstart=None, dt=1.):
    tt = sample_times(N, tstart, dt)
    poly = np.polyfit(tt, phase, order)
    return poly

def wrap_phase(phase):
    return phase - np.round(phase)

def phase_diff(phase, delay=1):
    phase_d = phase[delay:] - phase[:-delay] 
    return wrap_phase(phase_d)

def centered_phase(in_data):
    if np.iscomplexobj(in_data):
        sig = in_data
        phase = sig_to_phase(in_data)
    else:
        sig = phase_to_sig(in_data)
        phase = in_data.copy()
    mean_phase = np.angle(np.mean(sig))/(2.*np.pi)
    phase -= mean_phase
    phase = wrap_phase(phase)
    phase += mean_phase
    return phase


def unalias_poly(poly):
    """
    When fitting polynomials to wrapped data there are infinitely
    many polynomials to fit the same data points.  For example,
    adding integer cycles to the constant term will produce polynomials
    that look identical when wrapped.  This aliasing can be addressed by
    mapping the polynomial to some polynomial within a tessalating region
    for the lattice. This maps it to the hyper prism:
    intersection of -0.5/m! to 0.5/m! where m is the order of the coefficient.

    For a discussion see Chapter 7 of:
    McKilliam, Lattice theory, circular statistics and polynomial phase
    signals, 2010
    The algorithm is specified in Algorithm 2.2 on page 31

    inputs:
    poly - numpy array
      the polyomial we don't want to be aliased anymore
    dt - float
       the sampling interval for this polynomial
    radians - boolean
       whether this polynomial is in radians

    outputs:
    poly - numpy array
       the polynomial with the same units and sampling interval just unaliased

    """
    polylen = len(poly)

    # set up the generator matrix for the integer valued polynomials
    P = np.zeros((polylen, polylen))
    P[-1, 0] = 1.
    for ii in range(1, polylen):
        P[-ii-1:, ii] = np.polymul(P[:, ii-1], [1., ii-1])/ii

    # not quite sure if I understand what these lines of code do but
    # they do end up giving good results
    Q, R = np.linalg.qr(P)
    ystar = np.dot(Q.T, poly)
    u = np.zeros(polylen)
    for kk in reversed(range(polylen)):
        r = np.sum(R[kk, kk+1:]*u[kk+1:])
        u[kk] = np.round((ystar[kk]-r)/R[kk, kk])
    poly -= np.dot(P, u)
    return poly
