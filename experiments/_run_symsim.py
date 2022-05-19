#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd

from experiments_utils import (
    get_vals,
    compute_predictions,
)
from scvi.dataset import GeneExpressionDataset
from scvi.utils import make_dir_if_necessary
from scvi_utils import filename_formatter


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--n_cells_rare", type=int)
parser.add_argument("--n_cells_ref", type=int, default=None)
parser.add_argument("--n_cells_other", type=int, default=2000)
parser.add_argument("--delta", type=float, default=0.3)
parser.add_argument("--offset", type=float, default=1e-10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--folder_dir", type=str, default="lfc_estimates3")
args = parser.parse_args()
N_CELLS_QUERY = int(args.n_cells_rare)
FOLDER_NAME = str(args.folder_dir)
DELTA = float(args.delta)
SEED = int(args.seed)
print("SEED:", SEED)
N_CELLS_REF = args.n_cells_ref
N_OTHERS = args.n_cells_other
OFFSET = args.offset

N_EPOCHS = 1000
N_PICKS = 1
N_SAMPLES = 10000

print("USING OFFSET = {}".format(OFFSET))

RECONSTRUCTION_LOSS = "nb"
N_PICKS = 2
ADVANCED_PROPS = True
N_SAMPLES = 5000
# N_EPOCHS = 500
N_LATENT = 50
PIN_MEMORY = False
USE_BATCH_NORM = False
PATH_TO_SCRIPTS = "/home/pierre/lvm-DE-reproducibility/conquer_comparison/scripts"
DIR_PATH = "{FOLDER_NAME}/symsim_bimod_{N_CELLS_QUERY}_{N_CELLS_REF}_{OFFSET}_{DELTA}_{N_OTHERS}_{SEED}".format(
    FOLDER_NAME=FOLDER_NAME,
    N_CELLS_QUERY=N_CELLS_QUERY,
    N_CELLS_REF=N_CELLS_REF,
    OFFSET=OFFSET,
    DELTA=DELTA,
    N_OTHERS=N_OTHERS,
    SEED=SEED,
)


Q0 = 1e-1
# np.random.seed(SEED)
make_dir_if_necessary(DIR_PATH)
print(os.listdir(DIR_PATH))

#####################################################
#####################################################
# # Generate Datase
# dataset
# gene_means
# n_genes
# n_examples_ct
# label_a
# label_b
# lfc_gt
# is_significant_de
symsim_data_path = (
    "./data/symsim_nevf30.n_de_evf_18.sigma_0.rep_1/DE"
)
x_obs_all = pd.read_csv(
    os.path.join(symsim_data_path, "DE_med.obsv.3.csv"), index_col=0
).T
batch_info = (
    pd.read_csv(os.path.join(symsim_data_path, "DE_med.batchid.csv"), index_col=0) - 1
)
metadata = pd.read_csv(
    os.path.join(symsim_data_path, "DE_med.cell_meta.csv"), index_col=0
)

x_obs = x_obs_all

# OLD
select_gene = np.arange(x_obs_all.values.shape[1])
# non_null_genes = x_obs_all.sum(0) > 0
# _lfc_info = pd.read_csv(
#     os.path.join(symsim_data_path, "med_theoreticalFC.csv"), index_col=0
# )
# bimod_select = (_lfc_info["23"].abs() <= 0.2) | (_lfc_info["23"].abs() >= 1.5)

# NEW
label_a = 1
label_b = 2

k_on = pd.read_csv(os.path.join(symsim_data_path, "DE_med.kon_mat.csv"))
k_off = pd.read_csv(os.path.join(symsim_data_path, "DE_med.koff_mat.csv"))
s_mat = pd.read_csv(os.path.join(symsim_data_path, "DE_med.s_mat.csv"))

k_on = k_on.values.T
k_off = k_off.values.T
s_mat = s_mat.values.T

# labels = dataset.labels.squeeze()
labels = metadata["pop"].values - 1
where_a = np.where(labels == 1)[0]  # [:1000]
where_b = np.where(labels == 2)[0]  # [:1000]
print(label_a, label_b, where_a.shape, where_b.shape)
means_a = s_mat[where_a] * k_on[where_a] / (k_on[where_a] + k_off[where_a])
means_b = s_mat[where_b] * k_on[where_b] / (k_on[where_b] + k_off[where_b])
lfc_dist_gt = (np.log2(means_a) - np.log2(means_b))[:, select_gene]
lfc_gt = lfc_dist_gt.mean(0)
lfc_gt_alt = np.log2(means_a.mean(0)) - np.log2(means_b.mean(0))


select_gene = np.arange(x_obs_all.values.shape[1])
non_null_genes = x_obs_all.sum(0) > 0
bimod_select = (np.abs(lfc_gt) <= 0.2) | (np.abs(lfc_gt) >= 1.0)

select_gene = non_null_genes * bimod_select
# x_obs_all = x_obs_all.loc[:, non_null_genes]

x_obs = x_obs_all.loc[:, select_gene]
true_ = pd.read_csv(
    os.path.join(symsim_data_path, "DE_med.true.csv"), index_col=0
).T.loc[:, select_gene]
lfc_info = pd.read_csv(
    os.path.join(symsim_data_path, "med_theoreticalFC.csv"), index_col=0
).loc[select_gene, :]
# lfc_info[d_key].values
print("Original data distrib: ", metadata["pop"].groupby(metadata["pop"]).size())

# Data only
# n_examples_ct = np.array([2000, 2000, N_CELLS_QUERY, 2000, 2000])
n_examples_ct = np.array([N_OTHERS, N_CELLS_REF, N_CELLS_QUERY, N_OTHERS, N_OTHERS])

selected_indices = (
    metadata["pop"]
    .sample(frac=1.0, random_state=SEED)
    .groupby(metadata["pop"])
    .apply(lambda x: x.iloc[: n_examples_ct[x.name - 1]].index.values)
)
indices = np.concatenate(selected_indices.values) - 1
print(selected_indices.apply(lambda x: len(x)))

dataset = GeneExpressionDataset()
dataset.populate_from_data(
    X=x_obs.values[indices],
    batch_indices=batch_info["x"].values[indices],
    labels=metadata["pop"].values[indices],
    cell_types=metadata["pop"].values[indices],
)
print(dataset.X.shape)
print(np.unique(dataset.labels, return_counts=True))


n_examples = len(dataset)
TEST_INDICES = np.arange(len(dataset))
n_genes = dataset.nb_genes
x_test, y_test = dataset.X[TEST_INDICES, :], dataset.labels[TEST_INDICES, :].squeeze()
batch_test = dataset.batch_indices[TEST_INDICES, :].squeeze()
x_test[:, -1] = 1.0 + x_test[:, -1]
y_all = dataset.labels.squeeze()
data_path = os.path.join(DIR_PATH, "data.npy")
labels_path = os.path.join(DIR_PATH, "labels.npy")
batches_path = os.path.join(DIR_PATH, "batch_indices.npy")
np.save(data_path, x_test.squeeze().astype(int))
np.savetxt(labels_path, y_test.squeeze())
np.savetxt(batches_path, batch_test.squeeze())

# ## Params
label_a = 1
label_b = 2
n_genes = dataset.nb_genes

d_key = "{}{}".format(label_a + 1, label_b + 1)
print(d_key)

is_significant_de = (lfc_info[d_key].abs() >= DELTA).values
lfc_gt = lfc_info[d_key].values

print("label_a: {} examples".format(n_examples_ct[label_a]))
print("label_b: {} examples".format(n_examples_ct[label_b]))


idx_a = np.where(y_all == label_a)[0]
gene_means = np.log10(dataset.X[idx_a].mean(0))

sset_genes = (np.abs(lfc_gt) <= 0.1) | (np.abs(lfc_gt) >= DELTA + 0.1)
where_de_ee = np.arange(n_genes)
where_ = np.arange(n_genes)

# where_ = where_de_ee & sset_genes

#####################################################
#####################################################
# # Algorithms

# - scVI
# - observed library and softmax: mf_obs
# - observed library without softmax: de1
# - observed library, unnormalized scale and gene means de2

# ## Competitors

SIZES_A = [n_examples_ct[label_a] - 1]
SIZE_B = n_examples_ct[label_b] - 1

filename = os.path.join(
    DIR_PATH, "other_predictions_{}_{}_delta{}.pickle".format(label_a, label_b, DELTA)
)
other_kwargs = dict(
    filename=filename,
    PATH_TO_SCRIPTS=PATH_TO_SCRIPTS,
    folder_path=DIR_PATH,
)


##################################
# OFFSET = 0.1 / np.median(dataset.X.sum(-1))
OFFSET = 1e-12
##################################


scvi_kwargs = dict(
    USE_BATCH_NORM=USE_BATCH_NORM,
    RECONSTRUCTION_LOSS=RECONSTRUCTION_LOSS,
    N_EPOCHS=N_EPOCHS,
    N_SAMPLES=N_SAMPLES,
    offset=OFFSET,
)

savepath = os.path.join(DIR_PATH, "all_res.pkl")
idx_a = np.where(y_all == 1)[0]
idx_b = np.where(y_all == 2)[0]
if not os.path.exists(savepath):
    all_predictions_res = compute_predictions(
        idx_a=idx_a,
        idx_b=idx_b,
        scvi_kwargs=scvi_kwargs,
        others_kwargs=other_kwargs,
        dataset=dataset,
        q0=Q0,
        delta=DELTA,
        q_gmm=0.95,
        ncs=5,
        n_samples_c=100,
    )
else:
    all_predictions_res = dict()
vals = get_vals(savepath=savepath, **all_predictions_res)
df_vals = pd.DataFrame(vals).assign(
    N_CELLS_QUERY=N_CELLS_QUERY,
    FOLDER_NAME=FOLDER_NAME,
    DELTA=DELTA,
    N_CELLS_REF=N_CELLS_REF,
    N_OTHERS=N_OTHERS,
    OFFSET=OFFSET,
    SEED=SEED,
)
dataset_properties = pd.DataFrame(
    dict(
        is_significant_de=is_significant_de,
        lfc_gt=lfc_gt,
        # lfc_gt_alt=lfc_gt_alt,
        gene_means=gene_means,
        mean_a=dataset.X[idx_a].mean(0),
        mean_b=dataset.X[idx_b].mean(0),
    )
)
dataset_properties.to_pickle(
    os.path.join(DIR_PATH, filename_formatter("dataset_properties.pickle"))
)

repeated_arr = np.repeat(
    dataset_properties.is_significant_de.values[None, :], len(df_vals), axis=0
)
_is_significant_de = pd.Series(repeated_arr.tolist(), index=(np.arange(len(df_vals))))
df_vals.loc[:, "is_significant_de"] = _is_significant_de
repeated_arr = np.repeat(
    dataset_properties.lfc_gt.values[None, :], len(df_vals), axis=0
)
_lfc_gt = pd.Series(repeated_arr.tolist(), index=(np.arange(len(df_vals))))
df_vals.loc[:, "lfc_gt"] = _lfc_gt
df_vals.to_pickle(os.path.join(DIR_PATH, filename_formatter("results_save.pickle")))
