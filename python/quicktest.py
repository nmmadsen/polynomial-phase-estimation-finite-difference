
import numpy as np
import ppsignals
import fd

#pylint: disable=C0103
np.random.seed(1)
trials = 1000
N = 256
snr = 0
noise_var = 10.**(-0.1*snr)
order = 3
# the_poly = np.array([0.2, 0.3])

ord_range = np.arange(order, 1, -1)
# act    -5.62730545e-07  2.86880851e-04 -4.99885625e-01 -3.95334855e-02]
# array([-5.62730152e-07,  2.86880785e-04,  5.00110996e-01,  0.00000000e+00])
poly_range = 0.5/(order*ord_range*(N/2)**(ord_range-1))
poly_range = np.hstack((poly_range, [0.5, 0.1]))
error_sum = np.zeros(order+1)
for _ in range(trials):
    the_poly = np.random.uniform(-1, 1, order+1)*poly_range

    sig = ppsignals.sig_gen(the_poly, N, noise_var=noise_var)




    # import pdb; pdb.set_trace()
    est_poly = fd.polyfit(sig, order)

    diff_poly = ppsignals.unalias_poly(the_poly - est_poly)

    error_sum += diff_poly**2

crb = ppsignals.cramer_rao_bound(order, N, snr)
print(error_sum/crb/trials)
