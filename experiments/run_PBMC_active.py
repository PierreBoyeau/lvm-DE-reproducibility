import os

DIR_PATH1 = "./lfc_estimates_final"
if not os.path.exists(DIR_PATH1):
    os.makedirs(DIR_PATH1)

from sklearn.covariance import EllipticEnvelope
import torch.nn as nn
import logging
import plotnine as p9
import umap
from sklearn.cluster import DBSCAN
import anndata as ad

from scvi.models import SCSphere, SCSphereFull
from experiments_utils import load_scvi_model_if_exists

import logging
import torch.nn as nn

from experiments_utils import all_predictionsB, get_eb_full, get_props
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata
from tqdm.auto import tqdm


from scvi.dataset import GeneExpressionDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE
from scvi.utils import make_dir_if_necessary

import time
from experiments_utils import get_eb_full, all_predictionsB

import anndata
import pandas as pd
from scvi.dataset import PbmcDataset


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


import torch.nn as nn
import logging

import scanpy as sc


N_CELLS_QUERY = None
FOLDER_NAME = "runs/pbmc_inter_cts"
LIBRARY_FACTOR = None
OFFSET = None
DELTA = float(0.8)
SEED = 1
N_EPOCHS = 300
N_PICKS = 1
N_SAMPLES = 10000

print("USING OFFSET = {}".format(OFFSET))

RECONSTRUCTION_LOSS = "nb"
N_PICKS = 2
ADVANCED_PROPS = True
N_SAMPLES = 10000
# N_EPOCHS = 500
N_LATENT = 50
PIN_MEMORY = False
USE_BATCH_NORM = False
PATH_TO_SCRIPTS = "/home/pierre/lvm-DE-reproducibility/conquer_comparison/scripts"
DIR_PATH = "{FOLDER_NAME}/pbmc_bdc__{N_CELLS_QUERY}_{OFFSET}_{DELTA}_{LIBRARY_FACTOR}_{SEED}".format(
    FOLDER_NAME=FOLDER_NAME,
    N_CELLS_QUERY=N_CELLS_QUERY,
    OFFSET=OFFSET,
    DELTA=DELTA,
    LIBRARY_FACTOR=LIBRARY_FACTOR,
    SEED=SEED,
)
print(DIR_PATH)
Q0 = 1e-1
np.random.seed(42)
make_dir_if_necessary(DIR_PATH)
print(os.listdir(DIR_PATH))

#####################################################
#####################################################
# # Generate Datase

# lfc_gt
# is_significant_de
adata = anndata.read_h5ad(
    "./data/PBMC_updated/pbmc_final.h5ad"
)
meta = pd.read_csv(
    "./data/PBMC_updated/atac_v1_pbmc_10k_singlecell.csv"
)
mat = np.array(adata.X.todense())
X = mat
X_cp = np.copy(X)
print(X.min(), X.max())

new_adata = anndata.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
new_adata = adata
n_cells_thres = int(0.01 * len(new_adata))
print(n_cells_thres)
sc.pp.filter_genes(new_adata, min_cells=n_cells_thres)
sc.pp.highly_variable_genes(
    adata=new_adata,
    flavor="seurat_v3",
    inplace=True,
    n_top_genes=3000,  # inplace=False looks buggy,
)
new_adata
mat = np.array(adata.X.todense())
X = mat
X_cp = np.copy(X)

labels, cell_types = pd.factorize(adata.obs.celltype)
best_ = new_adata.var["highly_variable"].values
best_genes = adata.var.index[best_]
genes_to_keep = adata.var.index.isin(best_genes)
gene_names = adata.var.index[genes_to_keep]

unique_elements, counts_elements = np.unique(labels, return_counts=True)
for (name, counts) in zip(unique_elements, counts_elements):
    print(name, cell_types, counts)

label_a = 12
label_b = 11

df = pd.DataFrame(dict(counts=counts_elements, cell_types=cell_types))
print("Cell types: ", cell_types)
for (idx, gname) in enumerate(cell_types):
    print(idx, gname)

_, original_counts = np.unique(labels, return_counts=True)
print(
    "Original counts distribution: ",
    np.unique(labels, return_counts=True),
)
n_genes = X.shape[-1]
pops = pd.Series(labels)
n_examples_ct = original_counts
if N_CELLS_QUERY is not None:
    n_examples_ct[label_b] = N_CELLS_QUERY
else:
    print("using all cells")


indices = np.arange(len(new_adata))
counts = X[:, genes_to_keep]

if LIBRARY_FACTOR is not None:
    counts = np.ceil(counts * LIBRARY_FACTOR)

dataset = GeneExpressionDataset()
dataset.populate_from_data(
    X=counts,
    batch_indices=np.zeros_like(labels[indices]),
    labels=labels[indices],
    cell_types=cell_types,
    gene_names=gene_names,
)
print(dataset.X.shape)
_, final_counts = np.unique(dataset.labels, return_counts=True)
print("Final counts distribution: ", np.unique(dataset.labels, return_counts=True))
print("Selected cell types: ", cell_types[[label_a, label_b]])
print("Selected cell types counts: ", final_counts[[label_a, label_b]])


n_examples = len(dataset)
TEST_INDICES = np.arange(len(dataset))
n_genes = dataset.nb_genes
x_test, y_test = dataset.X[TEST_INDICES, :], dataset.labels[TEST_INDICES, :].squeeze()
batch_test = dataset.batch_indices[TEST_INDICES, :].squeeze()
x_test[:, -1] = 1.0 + x_test[:, -1]
y_all = dataset.labels.squeeze()
data_path = os.path.join(DIR_PATH1, "data.npy")
labels_path = os.path.join(DIR_PATH1, "labels.npy")
batches_path = None
np.save(data_path, x_test.squeeze().astype(int))
np.savetxt(labels_path, y_test.squeeze())
# np.savetxt(batches_path, batch_test.squeeze())


idx_a = np.where(y_all == label_a)[0]
gene_means = np.log10(dataset.X[idx_a].mean(0))

where_de_ee = np.arange(n_genes)
where = where_de_ee

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
    DIR_PATH1, "other_predictions_{}_{}_delta{}.pickle".format(label_a, label_b, DELTA)
)

##################################
OFFSET = 0.1 / np.median(dataset.X.sum(-1))
##################################

other_kwargs = dict(
    filename=filename,
    PATH_TO_SCRIPTS=PATH_TO_SCRIPTS,
    folder_path=DIR_PATH1,
)
scvi_kwargs = dict(
    USE_BATCH_NORM=USE_BATCH_NORM,
    RECONSTRUCTION_LOSS=RECONSTRUCTION_LOSS,
    N_EPOCHS=N_EPOCHS,
    LR=1e-3,
    N_SAMPLES=N_SAMPLES,
    offset=OFFSET,
)

savepath = os.path.join(DIR_PATH1, "all_res.pkl")
idx_a = np.where(y_all == label_a)[0]
idx_b = np.where(y_all == label_b)[0]


# In[54]:


final_ann = dataset


# In[57]:


ser = pd.Series(np.asarray((final_ann.X > 0.5).mean(0)).squeeze())
ser.describe()


# In[58]:


plt.hist(np.asarray(final_ann.X.sum(1)).squeeze(), bins=100)
plt.xlabel("Library size")
print()


# In[60]:


libs = np.asarray(final_ann.X.sum(1)).squeeze()
expressed = np.asarray((final_ann.X > 0.5).sum(1)).squeeze()
# b = pd.factorize(final_ann.obs.Donor_full)[0]
plt.scatter(libs, expressed)


# In[61]:


ser = pd.Series(expressed)
ser.describe()


# In[62]:


plt.hist(expressed, bins=100)
print()


# In[378]:


libs = np.asarray(final_ann.X.sum(1)).squeeze()
expressed = np.asarray((final_ann.X > 0.5).sum(1)).squeeze()
plt.scatter(libs, expressed)

# # Preliminaries

# In[194]:


cell_types

fine_cts = pd.Series(cell_types[y_all].values).astype(str)
fine_cts[fine_cts.str.contains("Monocyte")] = "monocyte"
fine_cts[fine_cts == "B cell progenitor"] = "b"
fine_cts[fine_cts == "pDC"] = "pdc"
fine_cts[fine_cts == "Dendritic cell"] = "dc"
fine_cts

# fine_cts.unique()

coarse_cts = fine_cts

fine_cts.unique()

pairs = [
    ("b", "monocyte"),
    ("b", "dc"),
    ("b", "pdc"),
    ("monocyte", "pdc"),
    ("dc", "pdc"),
    ("monocyte", "dc"),
]


def get_dgm_preds(
    trainer,
    key,
    label_a,
    label_b,
    idx_a,
    idx_b,
    n_samples=30000,
    filter_cts=True,
    offset=None,
):
    start = time.time()
    iw_props_eb_opt = get_eb_full(
        trainer=trainer,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=offset,
        delta=None,
        n_samples=n_samples,
        do_batch_specific="separation",
        posterior_chunks=200,
        filter_cts=filter_cts,
        #         coef_truncate=0.5,
        #         keeping_k_best_subsamples=20000,
        #         do_normalize=True,
    )
    stop = time.time() - start

    iw_props = {
        #         **iw_props,
        #         **iw_props_eb,
        **iw_props_eb_opt,
        #         **f_iw_props_eb_opt,
    }
    iw_res = pd.DataFrame([iw_props])
    prefix_key = "filt_" if filter_cts else ""
    values = [
        dict(
            algorithm="{}-fullIS-OPT".format(key),
            lfc_estim=iw_res["{}opt_full_eb_lfc_estim".format(prefix_key)].iloc[0],
            lfc_std=iw_res["{}opt_full_eb_lfc_std".format(prefix_key)].iloc[0],
            de_score=iw_res["{}opt_full_eb_is_de2".format(prefix_key)].iloc[0],
            GS=gene_names,
            is_proba=True,
        ),
    ]
    props_df = pd.DataFrame(values).assign(
        is_proba=True,
        is_pval=False,
        label_a=label_a,
        label_b=label_b,
        time=stop,
    )
    return props_df


# N_EPOCHS = 10
N_EPOCHS = 250

# In[65]:
iw_filename = os.path.join(DIR_PATH1, "iw_model_NEWA.pt")
sph_filename = os.path.join(DIR_PATH1, "sph_model_NEWA.pt")


# In[67]:


# # Training models

# In[2]:


torch.manual_seed(42)


# In[3]:


mdl_sph_kwargs = dict(
    n_genes=dataset.nb_genes,
    n_batches=dataset.n_batches,
    n_latent=11,
    do_depth_reg=False,
    constant_pxr=True,
    cell_specific_px=False,
    scale_norm="softmax",
    library_nn=True,
    dropout_rate=0.0,
    deep_architecture=False,
)
trainer_sph_kwargs = dict(
    train_library=True,
    k=25,
    loss_type="IWELBO",
    batch_size=128,
    weight_decay=1e-4,
    optimizer_type="adam",
    test_indices=[],
)
mdl_sph = SCSphereFull(**mdl_sph_kwargs)
trainer_sph = UnsupervisedTrainer(
    model=mdl_sph, gene_dataset=dataset, **trainer_sph_kwargs
)
lr_sph = trainer_sph.find_lr()
logging.info("Using learning rate {}".format(lr_sph))
mdl_sph = SCSphereFull(**mdl_sph_kwargs)
# mdl_sph, train_sph = load_scvi_model_if_exists(mdl_sph, filename=sph_filename)
trainer_sph = UnsupervisedTrainer(
    model=mdl_sph, gene_dataset=dataset, **trainer_sph_kwargs
)
# if train_sph:
trainer_sph.train(n_epochs=500, lr=lr_sph)
mdl_sph.eval()

# print()
mdl_iw_kwargs = dict(
    n_latent=10,
    dropout_rate=0.0,
    decoder_dropout_rate=0.0,
    reconstruction_loss="nb",
    dispersion="gene",
    n_layers=1,
    use_batch_norm=False,
    use_weight_norm=False,
    use_layer_norm=True,
    with_activation=nn.ReLU(),
)
trainer_iw_kwargs = dict(
    train_library=True,
    k=25,
    loss_type="IWELBO",
    #         batch_size=1024,
    batch_size=128,
    weight_decay=1e-4,
    optimizer_type="adam",
    test_indices=[],
)
mdl_iw = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_iw_kwargs)
trainer_iw = UnsupervisedTrainer(
    model=mdl_iw, gene_dataset=dataset, **trainer_iw_kwargs
)


params = filter(lambda p: p.requires_grad, mdl_iw.parameters())
lr_iw = trainer_iw.find_lr()

mdl_iw = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_iw_kwargs)
# mdl_iw, train_iw = load_scvi_model_if_exists(mdl_iw, filename=iw_filename)
trainer_iw = UnsupervisedTrainer(
    model=mdl_iw, gene_dataset=dataset, **trainer_iw_kwargs
)
logging.info("Using learning rate {}".format(lr_iw))
# if train_iw:
trainer_iw.train(n_epochs=250, lr=lr_iw)
mdl_iw.eval()
torch.save(mdl_iw.state_dict(), iw_filename)

torch.save(mdl_sph.state_dict(), sph_filename)
torch.save(mdl_iw.state_dict(), iw_filename)


# # NCs + Inter

# In[5]:


plt.plot(trainer_iw.train_losses[5:])
plt.plot(trainer_sph.train_losses[5:])

plt.plot(trainer_iw.train_losses[-100:])
plt.plot(trainer_sph.train_losses[-100:])

trainer_vae = trainer_iw
delta = 0.8
n_samples = 20000
dataset.X.max()

n_cts_max = 500
res = pd.DataFrame()

savefile = os.path.join(DIR_PATH1, "pbmc_inter_cts_3k_OFF_D.pickle")

for label_a, label_b in tqdm(pairs):
    idx_a = np.where(
        (coarse_cts == label_a)
        #             & (is_stim == "Healthy")
    )[0]
    idx_b = np.where(
        (coarse_cts == label_b)
        #             & (is_stim == "Healthy")
    )[0]

    print(idx_a.shape, idx_b.shape)
    print(len(idx_a), len(idx_b))
    labels = np.array([0] * len(idx_a) + [1] * len(idx_b))
    all_inds = np.concatenate([idx_a, idx_b])
    xobs = dataset.X[all_inds]
    xobs[:, 0] += 1
    batches = dataset.batch_indices[all_inds].squeeze()
    # folder_path = "."
    folder_path = DIR_PATH1
    filename = os.path.join(folder_path, "other_saves_pbmc_active.pickle")
    data_path = os.path.join(folder_path, "data_path_pbmc_active.npy")
    labels_path = os.path.join(folder_path, "labels_path_pbmc_active.npy")
    batches_path = os.path.join(folder_path, "batches_path_pbmc_active.npy")
    np.save(data_path, xobs.astype(int))
    np.savetxt(labels_path, labels)
    np.savetxt(batches_path, batches)
    batches_path = None

    other_predictions = all_predictionsB(
        filename=filename,
        data_path=data_path,
        labels_path=labels_path,
        path_to_scripts=PATH_TO_SCRIPTS,
        lfc_threshold=0.5,
        batches=batches_path,
        n_cells_max=500,
    )
    df_other = other_predictions.assign(
        is_proba=False,
        is_pval=True,
        label_a=label_a,
        label_b=label_b,
    )
    res = res.append(df_other, ignore_index=True)
    res.to_pickle(savefile)

    idx_a = np.where(
        (coarse_cts == label_a)
        #             & (is_stim == "Healthy")
    )[0]
    idx_b = np.where(
        (coarse_cts == label_b)
        #             & (is_stim == "Healthy")
    )[0]
    batch_data = trainer_vae.gene_dataset.batch_indices.squeeze()
    batch_a = batch_data[idx_a]
    batch_b = batch_data[idx_b]
    batches_of_interest = np.intersect1d(batch_a, batch_b)
    print(batches_of_interest)

    props_df = get_dgm_preds(
        trainer_sph,
        key="ScPhere",
        label_a=label_a,
        label_b=label_b,
        idx_a=idx_a,
        idx_b=idx_b,
        n_samples=n_samples,
    )
    res = res.append(props_df, ignore_index=True)
    res.to_pickle(savefile)

    props_df = get_dgm_preds(
        trainer_vae,
        key="scVI-DE",
        label_a=label_a,
        label_b=label_b,
        idx_a=idx_a,
        idx_b=idx_b,
        n_samples=n_samples,
    )
    res = res.append(props_df, ignore_index=True)
    res.to_pickle(savefile)

res.to_pickle(savefile)

delta = 0.8

adata.obs.celltype.cat.categories

coarse_cts = adata.obs.celltype
# batches_ = final_ann.obs.Status
# batches

coarse_cts.cat.categories


# In[103]:


np.random.seed(0)
idx_a = np.where(
    #         (coarse_cts == "dc")
    #         lab == 2
    #         (coarse_cts == "monocyte")
    (coarse_cts == "B cell progenitor")
    #     & (batches_ == "Healthy")
    #     & (final_ann.obs.Donor_full == "H5")
)[0]
n_exs = idx_a.shape[0]
rdm = np.random.permutation(n_exs)  # [:200]
idx_a = idx_a[rdm]
print(idx_a.shape)
n_exs = idx_a.shape[0]
rdm = np.random.permutation(n_exs)
rdm_aa = rdm[: n_exs // 2]
rdm_ab = rdm[n_exs // 2 :]
idx_aa = idx_a[rdm_aa]
idx_ab = idx_a[rdm_ab]
print(idx_aa.shape, idx_ab.shape)

idx_a = idx_aa
idx_b = idx_ab
print(idx_a.shape, idx_b.shape)


# In[104]:


label_a = "DC1"
label_b = "DC2"


# In[105]:


res = pd.DataFrame()
labels = np.array([0] * len(idx_a) + [1] * len(idx_b))
all_inds = np.concatenate([idx_a, idx_b])
xobs = dataset.X[all_inds]
xobs[:, 0] += 1
batches = dataset.batch_indices[all_inds].squeeze()
folder_path = "."
filename = os.path.join(folder_path, "other_saves_pbmcs_ncsB.pickle")
data_path = os.path.join(folder_path, "data_path_pbmcs_ncs.npy")
labels_path = os.path.join(folder_path, "labels_path_pbmcs_ncs.npy")
batches_path = os.path.join(folder_path, "batches_path_pbmcs_ncs.npy")
np.save(data_path, xobs.astype(int))
np.savetxt(labels_path, labels)
np.savetxt(batches_path, batches)
batches_path = None

other_predictions = all_predictionsB(
    filename=filename,
    data_path=data_path,
    labels_path=labels_path,
    path_to_scripts=PATH_TO_SCRIPTS,
    lfc_threshold=0.5,
    batches=batches_path,
)
df_other = other_predictions.assign(
    is_proba=False,
    is_pval=True,
    label_a=label_a,
    label_b=label_b,
)
res = res.append(df_other, ignore_index=True)

props_df = get_dgm_preds(
    trainer_sph,
    key="ScPhere",
    label_a=label_a,
    label_b=label_b,
    idx_a=idx_a,
    idx_b=idx_b,
    filter_cts=False,
)
res = res.append(props_df, ignore_index=True)

props_df = get_dgm_preds(
    trainer_iw,
    key="scVI-DE",
    label_a=label_a,
    label_b=label_b,
    idx_a=idx_a,
    idx_b=idx_b,
    filter_cts=False,
)
res = res.append(props_df, ignore_index=True)
savefile = os.path.join(DIR_PATH1, "pbmcs_NCs_3kOFF.pickle")
res.to_pickle(savefile)

# # Post clustering bias

# In[7]:

def get_latents(trainer):
    all_qs = []
    for tensors in trainer.train_set.sequential():
        #     print(tensors)
        sample_batch, _, _, batch_index, label = tensors
        with torch.no_grad():
            outs = trainer.model.inference(sample_batch, batch_index=batch_index)
        qz = outs["qz_m"].squeeze().detach().cpu()
        all_qs.append(qz)
    all_qs = torch.cat(all_qs, 0)
    return all_qs


# In[140]:


savedir = os.path.join(DIR_PATH1, "psb")
if not os.path.exists(savedir):
    os.makedirs(savedir)


# In[ ]:


# ## Transfer labels

# In[142]:


# In[ ]:


# In[134]:
