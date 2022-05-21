#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[260]:


import os


DIR_PATH1 = "./runs/blish_active"
if not os.path.exists(DIR_PATH1):
    os.makedirs(DIR_PATH1)


# In[2]:


import scanpy as sc
import anndata as ad
import pandas as pd
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm.auto import tqdm
from tqdm.auto import tqdm
import time
from experiments_utils import get_eb_full

from scvi.dataset import GeneExpressionDataset
from scvi.models import VAE
from scvi.models import SCSphereFull

from scvi.inference import UnsupervisedTrainer
from experiments_utils import all_predictionsB, get_eb_full

import matplotlib.pyplot as plt


from experiments_utils import get_props, all_predictionsB


import os

PATH_TO_SCRIPTS = "/home/pierre/lvm-DE-reproducibility/conquer_comparison/scripts"


# # Blish data


annd = ad.read_h5ad(
    "./data/blish_sars/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection_sparse_clean.h5ad"
)


nonzero_data = annd.X[:5]
nonzero_data = nonzero_data[nonzero_data != 0]


# ### Subsetting number of genes
lit_keys_to_dataset = pd.DataFrame(
    [
        [["memory_cd4", "mem_cd4"], "cd4m t", "Memory CD4"],
        [["effector_cd8", "eff_cd8"], "cd8eff t", "Effector CD8"],
        [["memory_cd8", "mem_cd8"], "cd8m t", "Memory CD8"],
        [["cd16"], "cd16 monocyte", "CD16"],
        [["dendritic", "_dc_"], "dc", "Dendritic"],
        # Removed because too small
        #         [["_iga_"], "iga pb", "IGA"],
        [["igg"], "igg pb", "IGG"],
        [["cd8_neg", "neg_cd8"], "cd4n t", "CD8-"],
        [["platelet"], "platelet", "Platelets"],
        [["granulo"], "granulocyte", "granulocyte"],
        [["nk_cell"], "nk", "NK"],
        [["neutrophil"], "neutrophil", "Neutrophil"],
        #
        #         [# rbc],
        [["gammadelta"], "gd t", "Gammadeltas"],
        [["pdc"], "pdc", "pdc"],
        ### Coarse cell types
        [["_cd4_"], "cd4 t", "CD4", False],
        [["_cd8_"], "cd8 t", "CD8", False],
        [["bcell", "b_cell"], "b", "B cell", False],
        [["monocyte"], "monocyte", "monocyte", False],
    ],
    columns=["Literature key", "Dataset key", "Cell-Type", "is_fine"],
).fillna(True)

# #### Using SEURAT

annd_s = ad.AnnData(X=annd.X, obs=annd.obs, var=annd.var)
_annd_s = ad.AnnData(X=annd.X, obs=annd.obs, var=annd.var).copy()

# Further subsampling of those genes
# sc.pp.normalize_total(_annd_s, target_sum=1e4)
# sc.pp.log1p(_annd_s)
sc.pp.filter_genes(
    _annd_s,
    min_cells=int(0.01 * (annd.X).shape[0]),
)
sc.pp.highly_variable_genes(
    adata=_annd_s,
    inplace=True,
    n_top_genes=3000,  # inplace=False looks buggy,
)

best_ = _annd_s.var["highly_variable"].values
best_ = _annd_s.var.loc[best_].index.values


# In[106]:


check = np.expm1(annd_s[:, best_].X)
check[check >= 1e-4].min()


# In[107]:


final_ann = ad.AnnData(
    X=np.around(np.expm1(annd_s[:, best_].X)),
    #     X=np.around(np.expm1(annd_s.X[:, :3000])),
    obs=annd_s.obs,
    var=annd_s.var.loc[best_],
)


# In[108]:


plt.hist(np.asarray((final_ann.X > 0.5).mean(0)).squeeze(), bins=100)
plt.xlabel("% of cells expressing gene")
print()


# In[109]:


plt.hist(np.log10(np.asarray(final_ann.X.sum(0)).squeeze()), bins=100)
plt.xlabel("Counts per gene")
print()


# In[111]:


ser = pd.Series(np.asarray((final_ann.X > 0.5).mean(0)).squeeze())
ser.describe()


# In[113]:


final_ann.X.sum(1).min()


# In[114]:


plt.hist(np.asarray(final_ann.X.sum(1)).squeeze(), bins=100)
plt.xlabel("Library size")
print()


# In[115]:


libs = np.asarray(final_ann.X.sum(1)).squeeze()
expressed = np.asarray((final_ann.X > 0.5).sum(1)).squeeze()
b = pd.factorize(final_ann.obs.Donor_full)[0]
plt.scatter(libs, expressed, c=b)


# In[116]:


ser = pd.Series(expressed)
ser.describe()


# In[117]:


plt.hist(expressed, bins=100)
print()


# In[118]:


libs = np.asarray(final_ann.X.sum(1)).squeeze()
expressed = np.asarray((final_ann.X > 0.5).sum(1)).squeeze()
plt.scatter(libs, expressed)


# In[119]:


final_ann.X.sum(1).min()


# In[ ]:


# In[120]:


# plt.hist(final_ann.X.sum(0))


# In[121]:


# final_ann.var.to_pickle("blish_genes_no_specific_5perc.pickle")


# In[122]:


annd_s.X.max()


# In[123]:


_annd_s.X.max()


# In[124]:


final_ann.X.max()


# In[125]:


final_ann.obs.keys()


# In[126]:


batch_indices, batch_cats = pd.factorize(final_ann.obs.Donor_full)
lbl_indices, lbl_cats = pd.factorize(final_ann.obs.cell_type_fine)

is_covid, covid_cats = pd.factorize(final_ann.obs.Status)


# In[127]:


np.asarray(final_ann.X.todense())


# In[128]:


dataset = GeneExpressionDataset()
dataset.populate_from_data(
    X=np.asarray(final_ann.X.todense()),
    batch_indices=batch_indices.astype(float),
    labels=lbl_indices,
    cell_types=lbl_cats,
)


# In[129]:


(lbl_indices == dataset.labels.squeeze()).all()


# ### Training models

# In[267]:


N_EPOCHS = 500

import logging

print(dataset.X.shape)

from experiments_utils import load_scvi_model_if_exists

sph_filename = os.path.join(DIR_PATH1, "blish_sph.pt")
iw_filename = os.path.join(DIR_PATH1, "blish_vae.pt")
# mdl_vae

# torch.manual_seed(0)
N_EPOCHS = 250
import logging


# In[268]:


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
    # train_library=True,
    # k=50,
    # batch_size=1024,
    # weight_decay=1e-4,
    # loss_type="DREG",
    # optimizer_type="adamw",
    # test_indices=[],
    train_library=True,
    k=25,
    loss_type="IWELBO",
    batch_size=128,
    #         batch_size=1024,
    weight_decay=1e-4,
    optimizer_type="adam",
    test_indices=[],
)


# In[271]:


mdl_sph_kwargs = dict(
    n_genes=dataset.nb_genes,
    n_batches=dataset.n_batches,
    n_latent=11,
    do_depth_reg=False,
    # constant_pxr=False,
    # cell_specific_px=True,
    constant_pxr=True,
    cell_specific_px=False,
    scale_norm="softmax",
    library_nn=True,
    deep_architecture=True,
    dropout_rate=0.0,
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
mdl_sph, train_sph = load_scvi_model_if_exists(mdl_sph, filename=sph_filename)
trainer_sph = UnsupervisedTrainer(
    model=mdl_sph, gene_dataset=dataset, **trainer_sph_kwargs
)
if train_sph:
    lr_sph = trainer_sph.find_lr()
    logging.info("Using learning rate {}".format(lr_sph))
    mdl_sph = SCSphereFull(**mdl_sph_kwargs)
    trainer_sph = UnsupervisedTrainer(
        model=mdl_sph, gene_dataset=dataset, **trainer_sph_kwargs
    )
    trainer_sph.train(n_epochs=N_EPOCHS, lr=lr_sph)
mdl_sph.eval()


# In[272]:


mdl_iw = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_iw_kwargs)
mdl_iw, train_iw = load_scvi_model_if_exists(mdl_iw, filename=iw_filename)
trainer_iw = UnsupervisedTrainer(
    model=mdl_iw, gene_dataset=dataset, **trainer_iw_kwargs
)
if train_iw:
    lr_iw = trainer_iw.find_lr()
    mdl_iw = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_iw_kwargs)
    trainer_iw = UnsupervisedTrainer(
        model=mdl_iw, gene_dataset=dataset, **trainer_iw_kwargs
    )
    logging.info("Using learning rate {}".format(lr_iw))
    trainer_iw.train(n_epochs=250, lr=lr_iw)
mdl_iw.eval()
torch.save(mdl_iw.state_dict(), iw_filename)
# mdl_iw.eval()
print()


# In[189]:


torch.save(mdl_sph.state_dict(), sph_filename)


# In[276]:


plt.plot(trainer_sph.train_losses[5:])
plt.plot(trainer_iw.train_losses[5:])


# In[277]:


plt.plot(trainer_sph.train_losses[-100:])
plt.plot(trainer_iw.train_losses[-100:])

trainer_vae = trainer_iw

dataset.X.max()

mdl_vae = mdl_iw
trainer_vae = trainer_iw


# ## NCs

# In[284]:


delta = 0.8


# In[285]:


def get_dgm_preds(trainer, key, label_a, label_b, idx_a, idx_b):

    start = time.time()
    iw_props_eb = get_eb_full(
        trainer=trainer,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=delta,
        #         n_samples=100000,
        n_samples=50000,
        #         do_batch_specific="infer",
        do_batch_specific="separation",
        posterior_chunks=200,
        filter_cts=True,
        coef_truncate=0.5,
    )
    stop = time.time() - start
    iw_props_eb_opt = get_eb_full(
        trainer=trainer,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        #         n_samples=100000,
        n_samples=50000,
        #         do_batch_specific="infer",
        do_batch_specific="separation",
        posterior_chunks=200,
        filter_cts=True,
        coef_truncate=0.5,
    )

    iw_props = {
        #         **iw_props,
        **iw_props_eb,
        **iw_props_eb_opt,
    }
    iw_res = pd.DataFrame([iw_props])

    values = [
        dict(
            algorithm="{}-fullIS".format(key),
            #             lfc_estim=iw_res["full_eb_lfc_estim"].iloc[0],
            lfc_estim=iw_res["filt_full_eb_lfc_estim"].iloc[0],
            lfc_std=iw_res["filt_full_eb_lfc_std"].iloc[0],
            de_score=iw_res["filt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=sph_res["is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="{}-fullIS-OPT".format(key),
            #             lfc_estim=iw_res["opt_full_eb_lfc_estim"].iloc[0],
            lfc_estim=iw_res["filt_opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=iw_res["filt_opt_full_eb_lfc_std"].iloc[0],
            de_score=iw_res["filt_opt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=sph_res["opt_is_med6_cred_maps"].iloc[0],
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


# In[ ]:


# In[286]:


final_ann.obs.cell_type_coarse.cat.categories


# In[287]:


# final_ann.obs.Donor_full


# In[290]:


coarse_cts = final_ann.obs.cell_type_coarse
batches_ = final_ann.obs.Status
# batches

info_df = final_ann.obs

info_df.loc[lambda x: x.Status == "Healthy"].groupby("cell_type_coarse").size()


# In[291]:


idx_a = np.where(
    #     (coarse_cts == "DC")
    (coarse_cts == "gd T")
    & (batches_ == "Healthy")
    #     & (final_ann.obs.Donor_full == "H5")
)[0]
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


np.intersect1d(idx_a, idx_b)


# In[295]:


label_a = "DC1"
label_b = "DC2"


# In[296]:


ser = final_ann.obs.Donor_full[idx_ab]
ser.groupby(ser).size()


# In[297]:

savefile = os.path.join(DIR_PATH1, "tototo_nc.pickle")
res = pd.DataFrame()
labels = np.array([0] * len(idx_a) + [1] * len(idx_b))
all_inds = np.concatenate([idx_a, idx_b])
xobs = dataset.X[all_inds]
xobs[:, 0] += 1
batches = dataset.batch_indices[all_inds].squeeze()
folder_path = DIR_PATH1
filename = os.path.join(folder_path, "other_saves_blish_ncsB.pickle")
data_path = os.path.join(folder_path, "data_path_blish_ncs.npy")
labels_path = os.path.join(folder_path, "labels_path_blish_ncs.npy")
batches_path = os.path.join(folder_path, "batches_path_blish_ncs.npy")
np.save(data_path, xobs.astype(int))
np.savetxt(labels_path, labels)
np.savetxt(batches_path, batches)
#     batches_path = None

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
res.to_pickle(savefile)

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
)
res = res.append(props_df, ignore_index=True)
# res.to_pickle(savefile)

props_df = get_dgm_preds(
    trainer_vae,
    key="scVI-DE",
    label_a=label_a,
    label_b=label_b,
    idx_a=idx_a,
    idx_b=idx_b,
)
res = res.append(props_df, ignore_index=True)

savefile = os.path.join(DIR_PATH1, "blish_deEB_FULL_NCs_final3.pickle")

res.to_pickle(savefile)


# ### NC calibration

# In[309]:


final_ann.obs

lbl_cats.categories.values

is_stim = final_ann.obs.Status.values

offset = 1e-10
delta = 0.8


# In[310]:


def get_dgm_preds(trainer, key, cell_type, idx_a, idx_b):
    iw_props_eb = get_eb_full(
        trainer=trainer,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=1e-16,
        n_samples=20000,
        do_batch_specific="disjoint",
        posterior_chunks=200,
        min_thres=0.0,
        filter_cts=True,
    )
    iw_props0 = {
        **iw_props_eb,
    }
    iw_res0 = pd.DataFrame([iw_props0])

    iw_props_eb_opt = get_eb_full(
        trainer=trainer,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        n_samples=20000,
        do_batch_specific="disjoint",
        posterior_chunks=200,
        min_thres=0.0,
        filter_cts=True,
    )
    iw_props = {
        **iw_props_eb_opt,
    }
    iw_res = pd.DataFrame([iw_props])

    values = [
        dict(
            algorithm="{}-fullIS-OPT_NOOFF".format(key),
            lfc_estim=iw_res0["filt_opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=iw_res0["filt_opt_full_eb_lfc_std"].iloc[0],
            de_score=iw_res0["filt_opt_full_eb_is_de2"].iloc[0],
            a_mean=dataset.X[idx_a].mean(0),
            a_max=np.max(dataset.X[idx_a], 0),
            b_mean=dataset.X[idx_b].mean(0),
            b_max=np.max(dataset.X[idx_b], 0),
            GS=final_ann.var.index.values,
            is_proba=True,
            # credible_intervals=sph_res["opt_is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="{}-fullIS-OPT".format(key),
            lfc_estim=iw_res["filt_opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=iw_res["filt_opt_full_eb_lfc_std"].iloc[0],
            de_score=iw_res["filt_opt_full_eb_is_de2"].iloc[0],
            a_mean=dataset.X[idx_a].mean(0),
            a_max=np.max(dataset.X[idx_a], 0),
            b_mean=dataset.X[idx_b].mean(0),
            b_max=np.max(dataset.X[idx_b], 0),
            #             ha_max=
            #             hb_max=
            GS=final_ann.var.index.values,
            is_proba=True,
            # credible_intervals=sph_res["opt_is_med6_cred_maps"].iloc[0],
        ),
    ]
    props_df = pd.DataFrame(values).assign(
        is_proba=True,
        is_pval=False,
        dataset_key=cell_type.lower(),
    )
    return props_df


# In[313]:


coarse_cts = final_ann.obs.cell_type_coarse.str.lower().copy()
fine_cts = final_ann.obs.cell_type_fine.str.lower().copy()

coarse_cts = coarse_cts.replace(
    {"cd16 monocyte": "monocyte", "cd14 monocyte": "monocyte"}
)

torch.cuda.empty_cache()


# In[315]:


# Intra CTs
savefile = os.path.join(DIR_PATH1, "blish_deEB_intra_3k_FINAL128.pickle")

np.random.seed(0)
res = pd.DataFrame()
cats = lbl_cats.categories.values

# for cat_id in np.unique(lbl_indices):
for idx, row in tqdm(lit_keys_to_dataset.iterrows()):
    is_fine = row.is_fine
    cat_id = row["Dataset key"]
    if is_fine:
        idx_a = np.where((fine_cts == cat_id).values & (is_stim == "COVID"))[0]
        idx_b = np.where((fine_cts == cat_id).values & (is_stim == "Healthy"))[0]
    else:
        idx_a = np.where((coarse_cts == cat_id).values & (is_stim == "COVID"))[0]
        idx_b = np.where((coarse_cts == cat_id).values & (is_stim == "Healthy"))[0]

    idx_a = np.random.permutation(idx_a)
    idx_b = np.random.permutation(idx_b)

    print(len(idx_a), len(idx_b))
    print(fine_cts.iloc[idx_a].unique())
    print(fine_cts.iloc[idx_b].unique())

    if (len(idx_a) == 0) or (len(idx_b) == 0):
        continue

    idx_a_save = idx_a.copy()
    idx_b_save = idx_b.copy()

    labels = np.array([0] * len(idx_a) + [1] * len(idx_b))
    all_inds = np.concatenate([idx_a, idx_b])
    xobs = dataset.X[all_inds]
    xobs[:, 0] += 1
    batches = dataset.batch_indices[all_inds].squeeze()
    folder_path = "."
    filename = os.path.join(folder_path, "other_saves_blish.pickle")
    data_path = os.path.join(folder_path, "data_path_blish.npy")
    labels_path = os.path.join(folder_path, "labels_path_blish.npy")
    batches_path = os.path.join(folder_path, "batches_path_blish.npy")
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
        dataset_key=cat_id,
    )
    res = res.append(df_other, ignore_index=True)
    res.to_pickle(savefile)

    idx_a = idx_a_save.copy()
    idx_b = idx_b_save.copy()

    try:
        props_df = get_dgm_preds(
            trainer_sph,
            key="ScPhere",
            cell_type=cat_id,
            idx_a=idx_a,
            idx_b=idx_b,
        )
        res = res.append(props_df, ignore_index=True)
        res.to_pickle(savefile)

        props_df = get_dgm_preds(
            trainer_vae,
            key="scVI-DE",
            cell_type=cat_id,
            idx_a=idx_a,
            idx_b=idx_b,
        )
        res = res.append(props_df, ignore_index=True)
        res.to_pickle(savefile)
    except RuntimeError:
        pass


# In[ ]:


serr = pd.DataFrame()


# In[ ]:


batch_data[(fine_cts == "neutrophil")]


# In[ ]:


# In[ ]:


res


# In[ ]:


savefile


# In[ ]:


res.to_pickle(savefile)


# In[320]:


def get_dgm_preds(trainer, key, label_a, label_b, idx_a, idx_b):
    start = time.time()
    iw_props_eb_opt = get_eb_full(
        trainer=trainer,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        n_samples=20000,
        do_batch_specific="separation",
        posterior_chunks=200,
    )
    stop = time.time() - start
    iw_props = {
        **iw_props_eb_opt,
    }
    iw_res = pd.DataFrame([iw_props])

    values = [
        dict(
            algorithm="{}-fullIS-OPT".format(key),
            lfc_estim=iw_res["opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=iw_res["opt_full_eb_lfc_std"].iloc[0],
            de_score=iw_res["opt_full_eb_is_de2"].iloc[0],
            GS=final_ann.var.index.values,
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


# In[321]:


coarse_cts.unique()


# In[322]:


pairs = [
    ("b", "monocyte"),
    ("b", "dc"),
    ("b", "pdc"),
    ("monocyte", "pdc"),
    ("dc", "pdc"),
    ("monocyte", "dc"),
]


# In[323]:


coarse_cts = final_ann.obs.cell_type_coarse.str.lower().copy()
fine_cts = final_ann.obs.cell_type_fine.str.lower().copy()


# In[324]:


coarse_cts.unique()


# In[325]:


coarse_cts = coarse_cts.replace(
    {"cd16 monocyte": "monocyte", "cd14 monocyte": "monocyte"}
)


# In[327]:


res = pd.DataFrame()
savefile = os.path.join(DIR_PATH1, "blish_inter_cts_3k_offs_final128.pickle")

for label_a, label_b in tqdm(pairs):
    idx_a = np.where((coarse_cts == label_a) & (is_stim == "Healthy"))[0]
    idx_b = np.where((coarse_cts == label_b) & (is_stim == "Healthy"))[0]

    print(len(idx_a), len(idx_b))

    #     if len(idx_a) >= 500:
    #         idx_a = np.random.choice(idx_a, size=np.minimum(len(idx_a), 500), replace=False)
    #     if len(idx_b) >= 500:
    #         idx_b = np.random.choice(idx_b, size=np.minimum(len(idx_b), 500), replace=False)
    print(len(idx_a), len(idx_b))
    # if np.minimum(len(idx_a), len(idx_b)) <= 10:
    #     continue

    labels = np.array([0] * len(idx_a) + [1] * len(idx_b))
    all_inds = np.concatenate([idx_a, idx_b])
    xobs = dataset.X[all_inds]
    xobs[:, 0] += 1
    batches = dataset.batch_indices[all_inds].squeeze()
    folder_path = DIR_PATH1
    filename = os.path.join(folder_path, "other_saves_blish_inter_cts.pickle")
    data_path = os.path.join(folder_path, "data_path_blish_inter_cts.npy")
    labels_path = os.path.join(folder_path, "labels_path_blish_inter_cts.npy")
    batches_path = os.path.join(folder_path, "batches_path_blish_inter_cts.npy")
    np.save(data_path, xobs.astype(int))
    np.savetxt(labels_path, labels)
    np.savetxt(batches_path, batches)
    #     batches_path = None

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

    idx_a = np.where((coarse_cts == label_a) & (is_stim == "Healthy"))[0]
    idx_b = np.where((coarse_cts == label_b) & (is_stim == "Healthy"))[0]
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
    )
    res = res.append(props_df, ignore_index=True)
    res.to_pickle(savefile)


# In[ ]:


savefile


# In[ ]:


res.to_pickle(savefile)


# In[ ]:


res


# In[ ]:


idx_a = np.where(
    np.isin(dataset.labels.squeeze(), [8, 2])
    #         (dataset.labels.squeeze() == 5)
    & (is_stim == "Healthy")
)[0]
idx_b = np.where(
    (dataset.labels.squeeze() == 19)
    #         (dataset.labels.squeeze() == 6)
    & (is_stim == "Healthy")
)[0]

print(idx_a.shape, idx_b.shape)


# In[ ]:


lbl_cats[8]


# In[ ]:


lbl_cats[19]


# In[ ]:


dataset.cell_types


# In[ ]:


mdl_iw_kwargs = dict(
    n_latent=10,
    # n_latent=3,
    dropout_rate=0.1,
    decoder_dropout_rate=0.0,
    reconstruction_loss="nb",
    dispersion="gene",
    n_layers=1,
    use_batch_norm=False,
    #     n_hidden=256,
    #     n_layers=3,
    use_weight_norm=False,
    use_layer_norm=True,
    with_activation=nn.ReLU(),
)
trainer_iw_kwargs = dict(
    train_library=True,
    k=50,
    batch_size=128,
    # weight_decay=0,
    weight_decay=1e-4,
    # loss_type="IWELBO",
    loss_type="DREG",
    optimizer_type="adamw",
    # grad_clip_value=100.0,
    test_indices=[],
    # beta_policy="constant",
)
mdl_iw = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_iw_kwargs)
# mdl_iw, train_iw = load_scvi_model_if_exists(mdl_iw, filename=iw_filename)
trainer_iw = UnsupervisedTrainer(
    model=mdl_iw, gene_dataset=dataset, **trainer_iw_kwargs
)
# if train_iw:
lr_iw = trainer_iw.find_lr()
mdl_iw = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_iw_kwargs)
trainer_iw = UnsupervisedTrainer(
    model=mdl_iw, gene_dataset=dataset, **trainer_iw_kwargs
)
logging.info("Using learning rate {}".format(lr_iw))
trainer_iw.train(n_epochs=200, lr=lr_iw)
mdl_iw.eval()
# torch.save(mdl_iw.state_dict(), iw_filename)
# mdl_iw.eval()
print()


# In[ ]:


trainer_iw.train_losses[-50:]


# In[ ]:


plt.plot(trainer_iw.train_losses)


# In[ ]:


coarse_cts.unique()


# In[ ]:


# idx_a = np.where(
#     (coarse_cts == 'cd8 t')
# #         (dataset.labels.squeeze() == 5)
#         & (is_stim == "Healthy")
# )[0]
# idx_b = np.where(
#     (coarse_cts == 'cd4 t')
# #         (dataset.labels.squeeze() == 6)
#         & (is_stim == "Healthy")
# )[0]


# In[ ]:


idx_a = np.where(
    (dataset.labels.squeeze() == 8)
    #         (dataset.labels.squeeze() == 5)
    & (is_stim == "Healthy")
)[0]
idx_b = np.where(
    (dataset.labels.squeeze() == 19)
    #         (dataset.labels.squeeze() == 6)
    & (is_stim == "Healthy")
)[0]


print(len(idx_a), len(idx_b))

# nmax = 500
# if len(idx_a) >= nmax:
#     idx_a = np.random.choice(idx_a, size=np.minimum(len(idx_a), nmax), replace=False)
# if len(idx_b) >= nmax:
#     idx_b = np.random.choice(idx_b, size=np.minimum(len(idx_b), nmax), replace=False)


# In[ ]:


np.unique(dataset.batch_indices[idx_a].squeeze(), return_counts=True)


# In[ ]:


mdl_iw.eval()


# In[ ]:


iw_props_eb = get_eb_full(
    trainer=trainer_iw,
    idx_a=idx_a,
    idx_b=idx_b,
    offset=None,
    #     offset=1e-5,
    delta=delta,
    n_samples=60000,
    #     do_batch_specific="infer",
    do_batch_specific="separation",
    posterior_chunks=1000,
)
