import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

result_elbo = pickle.load(open("figures/simulations_scRNA/results.dic", "rb"))
result_iwae = pickle.load(open("figures/simulations_scRNA/results_IWAE.dic", "rb"))
result_cubo = pickle.load(open("figures/simulations_scRNA/resultsCUBO.dic", "rb"))
result_kl = pickle.load(open("figures/simulations_scRNA/resultsKL.dic", "rb"))


for k in result_elbo.keys():
    for res_type in result_elbo[k].keys():
        print(k, res_type)
        print("ELBO", np.mean(result_elbo[k][res_type][:10]), "+-", np.std(result_elbo[k][res_type][:10]))
        print("IWAE", np.mean(result_iwae[k][res_type][:10]), "+-", np.std(result_iwae[k][res_type][:10]))
        print("CUBO", np.mean(result_cubo[k][res_type][:10]), "+-", np.std(result_cubo[k][res_type][:10]))
        print("KL", np.mean(result_kl[k][res_type][:10]), "+-", np.std(result_kl[k][res_type][:10]))

