# from scvi.inference.posterior import _p_wa_higher_wb
# import numpy as np


# def test_beta_cdf():
#     np.random.seed(42)
#     k1 = 0.5
#     k2 = 1.0
#     theta_1 = 1.2
#     theta_2 = 0.7
#     gt_p = _p_wa_higher_wb(k1, k2, theta_1, theta_2)
#     samp1 = np.random.gamma(k1, theta_1, size=10000)
#     samp2 = np.random.gamma(k2, theta_2, size=10000)

#     inf_p = (samp1 >= samp2).astype(float).mean()
#     assert ((inf_p - gt_p) / inf_p) <= 1e-2
#     print(((inf_p - gt_p) / inf_p))
