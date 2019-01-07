
import numpy as np
import ppsignals
import basic

#pylint: disable=C0103
N = 11
the_poly = np.array([0.01, 0.2, 0.3])
the_poly = np.array([0.2, 0.3])
tt = ppsignal.sample_times(N)
unwrapped_phase = np.polyval(tt, the_poly, N)

print(unwrapped_phase)

wrapped_phase = ppsignals.wrap_phase(unwrapped_phase)

unwrapped_phase2 = basic.unwrap(wrapped_phase)

print(unwrapped_phase - unwrapped_phase2)
