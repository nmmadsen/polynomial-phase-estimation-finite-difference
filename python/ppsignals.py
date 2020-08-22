import numpy as np
#pylint: disable=C0103

def default_tstart(N, dt=1.):
    """
    Having it centered gives the lowest CRB, but on the other hand, it needs
    to be an integer, otherwise the math for aliasing doesn't quite work since
    it is based on the integer polynomials. This makes the CRB quite a bit more
    painful to estimate, but it is pretty close, and for analysis we could just
    use odd signal lengths
    """
    return int(-(N-1)/2.) * dt

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

def sig_gen(poly, N, magnitude=1, noise_var=0, tstart=None, dt=1.):
    sig = poly_to_sig(poly, N, tstart, dt)*magnitude
    noise = np.random.normal(0.0, np.sqrt(noise_var/2), sig.shape) + \
            1j * np.random.normal(0.0, np.sqrt(noise_var/2), sig.shape)
    sig += noise
    return sig

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

def cramer_rao_bound(order, length, snr, radians=False, dt=1.):
    """
    Cramer Rao bound, only works if zero time is centered in the middle of
    the signal

    Taken from O'Shea "On Refining Polynomial Phase
    Signal Parameter Estimates"

    Also not too bad to derive on your own
    """
    D_inv = np.diag(1./(dt**np.arange(order+1)))
    H_p1 = np.zeros((order+1, order+1))
    for ii in range(order+1):
        for jj in range(order+1):
            if (ii+jj)%2 == 0:
                H_p1[ii, jj] = 2*(length/2)**(ii+jj+1)/(ii+jj+1)

    crb = 1./(2*(10.0**(0.1*snr)))*np.diag((D_inv.dot(np.linalg.solve(H_p1, D_inv))))
    if not radians:
        crb /= (2*np.pi)**2
    crb = crb[::-1]
    return crb
