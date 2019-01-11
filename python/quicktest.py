
import numpy as np
import ppsignals
import basic

#pylint: disable=C0103
N = 50
the_poly = np.array([0.05, 0.2, 0.3])
order = len(the_poly)-1
# the_poly = np.array([0.2, 0.3])
tt = ppsignals.sample_times(N)
unwrapped_phase = np.polyval(the_poly, tt)

print(unwrapped_phase)

wrapped_phase = ppsignals.wrap_phase(unwrapped_phase)

unwrapped_phase2 = basic.unwrap(wrapped_phase, 2, center_phase=True)

print(unwrapped_phase - unwrapped_phase2)

poly_est = np.polyfit(tt, unwrapped_phase2, order)

poly_est_unalias = ppsignals.unalias_poly(poly_est)

print(poly_est_unalias)
