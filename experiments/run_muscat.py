#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import pandas as pd

from experiments_utils import (
    compute_predictions,
    get_vals,
)
from scvi.dataset import GeneExpressionDataset
from scvi.utils import make_dir_if_necessary
from scvi_utils import (
    filename_formatter,
)


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--n_cells_rare", type=int)
parser.add_argument("--delta", type=float, default=0.3)
parser.add_argument("--offset", type=float, default=1e-10)
parser.add_argument("--n_cells_ref", type=int)
parser.add_argument("--n_cells_other", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--folder_name", type=str)
args = parser.parse_args()
N_CELLS_REF = int(args.n_cells_ref)
N_CELLS_QUERY = int(args.n_cells_rare)
DELTA = float(args.delta)
SEED = args.seed
FOLDER_NAME = args.folder_name
N_OTHERS = args.n_cells_other
OFFSET = args.offset

N_EPOCHS = 500
N_PICKS = 1
N_SAMPLES = 10000


np.random.seed(SEED)
print("USING OFFSET = {}".format(OFFSET))


RECONSTRUCTION_LOSS = "nb"
N_PICKS = 2
ADVANCED_PROPS = True
N_SAMPLES = 10000
# N_EPOCHS = 500
N_LATENT = 50
USE_BATCH_NORM = False
PATH_TO_SCRIPTS = "/home/pierre/lvm-DE-reproducibility/conquer_comparison/scripts"
DIR_PATH = "{}/muscat_{}_{}_{}_{}_{}_{}".format(
    FOLDER_NAME, N_CELLS_REF, N_CELLS_QUERY, OFFSET, DELTA, SEED, N_OTHERS
)

Q0 = 1e-1
make_dir_if_necessary(DIR_PATH)
print(os.listdir(DIR_PATH))

# # Generate Datase
cell_info = pd.read_csv("data/muscat_cell_info.csv", index_col=0)
gene_info = pd.read_csv("data/muscat_gene_info.csv", index_col="gene")
counts = pd.read_csv("data/muscat_counts.csv", index_col=0).T
batch_info = cell_info.sample_id.str.split(".", expand=True).loc[:, 0]
ct_info = cell_info.cluster_id + "_" + cell_info.group_id
lfc_info = gene_info[lambda x: x.cluster_id == "cluster2"]

where_de_ee = lfc_info.category.isin(["de", "ee"])
# Subsampling genes based on policy
gene_info = gene_info[where_de_ee]
counts = counts.loc[:, where_de_ee]
lfc_info = gene_info[lambda x: x.cluster_id == "cluster2"]

n_examples_ct = ct_info.groupby(ct_info).size()
n_examples_ct["cluster2_B"] = N_CELLS_QUERY
n_examples_ct["cluster2_A"] = N_CELLS_REF
if N_OTHERS is not None:
    n_examples_ct.loc[lambda x: ~x.index.isin(["cluster2_A", "cluster2_B"])] = N_OTHERS
print(n_examples_ct)
selected_indices = ct_info.groupby(ct_info).apply(
    lambda x: x.sample(n_examples_ct[x.name], random_state=42).index.values
)
indices = np.concatenate(selected_indices.values)

ct_info_ = ct_info.loc[indices]
batch_info_ = batch_info.loc[indices]
counts_ = counts.loc[indices]
labels_, labels_conv_ = pd.factorize(ct_info_)
batch_indices_ = pd.factorize(batch_info_)[0]

dataset = GeneExpressionDataset()
dataset.populate_from_data(
    X=counts_.values,
    batch_indices=batch_indices_,
    labels=labels_,
    gene_names=counts_.columns,
)
n_examples = len(dataset)
TEST_INDICES = np.arange(len(dataset))
n_genes = dataset.nb_genes

logmeans = np.log(dataset.X.mean(0) + 1e-3)
logmeans[logmeans <= 0.0] = 0.0
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
_label_a = "cluster2_A"
_label_b = "cluster2_B"
label_a = np.where(labels_conv_ == _label_a)[0][0]
label_b = np.where(labels_conv_ == _label_b)[0][0]
print("label_a: {} examples".format(n_examples_ct[label_a]))
print("label_b: {} examples".format(n_examples_ct[label_b]))
print(label_a, _label_a)
print(label_b, _label_b)

lfcs_gt = lfc_info.logFC.fillna(0)
print(OFFSET)

lfc_gt = -lfcs_gt
is_significant_de = lfc_info.category == "de"

where_de_ee = np.arange(n_genes)
where_ = np.arange(n_genes)
where_.sum()

idx_a = np.where(y_all == label_a)[0]
gene_means = np.log10(dataset.X[idx_a].mean(0))

dataset_properties = pd.DataFrame(
    dict(
        is_significant_de=is_significant_de,
        lfc_gt=lfc_gt,
        gene_means=gene_means,
    )
)
dataset_properties.to_pickle(
    os.path.join(DIR_PATH, filename_formatter("dataset_properties.pickle"))
)
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
    SIZES_A=SIZES_A,
    SIZE_B=SIZE_B,
    data_path=data_path,
    labels_path=labels_path,
    PATH_TO_SCRIPTS=PATH_TO_SCRIPTS,
    label_a=label_a,
    label_b=label_b,
    batches_path=batches_path,
    folder_path=DIR_PATH,
)
scvi_kwargs = dict(
    USE_BATCH_NORM=USE_BATCH_NORM,
    RECONSTRUCTION_LOSS=RECONSTRUCTION_LOSS,
    N_EPOCHS=N_EPOCHS,
    N_SAMPLES=N_SAMPLES,
    offset=OFFSET,
    do_sph_deep=True,
)

savepath = os.path.join(DIR_PATH, "all_res.pkl")
# idx_a = np.where(y_all == 1)[0]
# idx_b = np.where(y_all == 2)[0]
idx_a = np.where(ct_info_ == "cluster2_A")[0]
idx_b = np.where(ct_info_ == "cluster2_B")[0]

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