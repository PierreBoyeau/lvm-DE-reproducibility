#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import get_cmap
import scanpy as sc
import anndata

from scvi.dataset import GeneExpressionDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE, LDVAE
from scvi.utils import make_dir_if_necessary


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# parser = argparse.ArgumentParser(description="Process some integers.")
# # parser.add_argument("--dkey", type=str)
# # parser.add_argument("--n_epochs", type=str)
# parser.add_argument("--n_cells_rare", type=int)
# # parser.add_argument("--offset", type=float)
# parser.add_argument("--delta", type=float, default=1.1)
# parser.add_argument("--offset", type=float, default=1e-10)

# args = parser.parse_args()
# N_CELLS_QUERY = int(args.n_cells_rare)
# # d_key = args.dkey
# OFFSET = args.offset

N_CELLS_QUERY = 50
OFFSET = 2e-10
DELTA = 1.0
# N_EPOCHS = int(args.n_epochs)
# if OFFSET is not None:
#     OFFSET = float(OFFSET)
# DELTA = float(args.delta)
N_EPOCHS = 250
N_PICKS = 1
N_SAMPLES = 30000

print("USING OFFSET = {}".format(OFFSET))

RECONSTRUCTION_LOSS = "nb"
N_PICKS = 2
ADVANCED_PROPS = True
N_SAMPLES = 30000
# N_EPOCHS = 500
N_LATENT = 50
PIN_MEMORY = False
USE_BATCH_NORM = False
PATH_TO_SCRIPTS = "/home/pierre/lvm-DE-reproducibility/conquer_comparison/scripts"
DIR_PATH = "lfc_estimates_new/symsim_bimod_fairE_{}{}{}".format(
    N_CELLS_QUERY, OFFSET, DELTA
)

Q0 = 1e-1
np.random.seed(42)
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
symsim_data_path = "/data/yosef2/users/pierreboyeau/symsim_result_complete/DE"
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

select_gene = np.arange(x_obs_all.values.shape[1])
non_null_genes = x_obs_all.sum(0) > 0
_lfc_info = pd.read_csv(
    os.path.join(symsim_data_path, "med_theoreticalFC.csv"), index_col=0
)
bimod_select = (_lfc_info["23"].abs() <= 0.2) | (_lfc_info["23"].abs() >= 1)

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
n_examples_ct = np.array([1000, 2250, N_CELLS_QUERY, 2250, 2250])

selected_indices = (
    metadata["pop"]
    .groupby(metadata["pop"])
    .apply(lambda x: x.sample(n_examples_ct[x.name - 1], random_state=42).index.values)
)
indices = np.concatenate(selected_indices.values) - 1

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


def test_iw():
    # mdl_iw = VAE(
    #     n_input=dataset.nb_genes,
    #     n_batch=dataset.n_batches,
    #     use_batch_norm=USE_BATCH_NORM,
    #     reconstruction_loss=RECONSTRUCTION_LOSS,
    #     dispersion="gene",
    #     n_layers=1,
    #     use_weight_norm=True,
    # )
    # trainer_iw = UnsupervisedTrainer(
    #     model=mdl_iw,
    #     gene_dataset=dataset,
    #     train_library=True,
    #     k=15,
    #     loss_type="IWELBO",
    #     test_indices=[],
    # )
    # trainer_iw.train(n_epochs=5, lr=1e-3)

    mdl_ld = LDVAE(
        n_input=dataset.nb_genes,
        n_batch=dataset.n_batches,
        reconstruction_loss=RECONSTRUCTION_LOSS,
        dispersion="gene",
        n_layers=1,
    )
    trainer_ld = UnsupervisedTrainer(
        model=mdl_ld,
        gene_dataset=dataset,
        train_library=True,
        k=1,
        loss_type="ELBO",
        test_indices=[],
    )
    a = trainer_ld.train_set.get_key("log_ratio")
    test_reconstruction_loss = (
        trainer_ld.train_set.sequential().get_key("log_px_zl").mean(0)
    )
    print(test_reconstruction_loss.shape)
    print(a.shape)
    # trainer_ld.train(n_epochs=30, lr=1e-3)
