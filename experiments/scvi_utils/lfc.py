import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as db
from sklearn.metrics import r2_score, precision_score

from scvi.inference import UnsupervisedTrainer
from scvi.utils import predict_de_genes


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res


def train_model(
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    filename: str = None,
):
    """

    :param mdl_class: Class of algorithm
    :param dataset: Dataset
    :param mdl_params:
    :param train_params:
    :param train_fn_params:
    :param filename
    :return:
    """
    # if os.path.exists(filename):
    #     res = load_pickle(filename)
    #     return res["vae"], res["trainer"]

    if "test_indices" not in train_params:
        warnings.warn("No `test_indices` attribute found.")
    my_vae = mdl_class(
        n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_params
    )
    my_trainer = UnsupervisedTrainer(my_vae, dataset, **train_params)
    my_trainer.train(**train_fn_params)
    print(my_trainer.train_losses)
    return my_vae, my_trainer


def train_or_load(
    filepath, dataset, my_mdl_class, my_mdl_params, my_train_params, my_train_fn_params
):
    if os.path.exists(filepath):
        tup = load_pickle(filepath)
    else:
        tup = train_model(
            mdl_class=my_mdl_class,
            dataset=dataset,
            mdl_params=my_mdl_params,
            train_params=my_train_params,
            train_fn_params=my_train_fn_params,
        )
        save_pickle(tup, filepath)
    return tup


def filename_formatter(filepath):
    filename, filetype = filepath.split(".")
    return "{}_2.{}".format(filename, filetype)


def get_r2(preds, gt):
    return r2_score(preds, gt)


def subsampled_posterior(post, indices):
    post.data_loader.sampler.indices = indices
    return post


def compute_lfc(
    my_trainer,
    my_idx_a,
    my_idx_b,
    n_samples=1000,
    importance_sampling=False,
    train_library=None,
    safe_post=True,
):
    softmax = nn.Softmax(dim=0)

    original_post = my_trainer.train_set
    if safe_post:
        original_post = my_trainer.create_posterior(
            model=my_trainer.model,
            gene_dataset=my_trainer.gene_dataset,
            indices=np.arange(len(my_trainer.gene_dataset)),
        )
    post_a = subsampled_posterior(original_post, my_idx_a)

    n_post_samples_a = n_samples // len(my_idx_a)
    print(
        "Using {} posterior samples for population a {}".format(
            n_post_samples_a, len(my_idx_a)
        )
    )
    outputs_a = post_a.get_latents(
        n_samples=n_post_samples_a,
        other=True,
        device="cpu",
        train_library=train_library,
    )
    scales_a, weights_a = outputs_a["scale"], outputs_a["log_probas"]

    original_post = my_trainer.train_set
    if safe_post:
        original_post = my_trainer.create_posterior(
            model=my_trainer.model,
            gene_dataset=my_trainer.gene_dataset,
            indices=np.arange(len(my_trainer.gene_dataset)),
        )
    post_b = subsampled_posterior(my_trainer.train_set, my_idx_b)
    n_post_samples_b = n_samples // len(my_idx_b)
    print(
        "Using {} posterior samples for population b {}".format(
            n_post_samples_b, len(my_idx_b)
        )
    )
    outputs_b = post_b.get_latents(
        n_samples=n_post_samples_b,
        other=True,
        device="cpu",
        train_library=train_library,
    )
    scales_b, weights_b = outputs_b["scale"], outputs_b["log_probas"]
    #     assert (outputs_b["label"].squeeze() == label_b).all()
    #     dropouts_b = torch.tensor(post_b.generate_parameters()[0])
    #     dropouts_p_b = softmax(dropouts_b)

    print("Plugin estimator ...")
    print(scales_a.shape)
    scales_a = scales_a
    scales_b = scales_b
    batch_a = outputs_a["batch_index"]
    batch_b = outputs_b["batch_index"]
    return (scales_a, batch_a, weights_a), (scales_b, batch_b, weights_b)


def get_fast_cred(arr, confidence=95.0):
    low_q = (100.0 - confidence) / 2
    high_q = 100.0 - low_q
    low_b = np.percentile(arr, low_q, axis=0)
    high_b = np.percentile(arr, high_q, axis=0)
    return np.stack([low_b, high_b])


def extract_lfc_properties(dataset, delta, prop_a, prop_b, offset=0.0, levels=None):
    scales_a_, batch_a_, _ = prop_a
    scales_b_, batch_b_, _ = prop_b

    if levels is None:
        _levels = np.array([50, 55, 60, 65, 70, 75, 80, 90, 95])
    else:
        _levels = levels
    assert scales_a_.ndim == 3 and scales_b_.ndim == 3, (
        scales_a_.shape,
        scales_b_.shape,
    )

    scales_a = scales_a_.clone().reshape((-1, dataset.nb_genes)).numpy()
    scales_b = scales_b_.clone().reshape((-1, dataset.nb_genes)).numpy()
    batch_a = batch_a_.clone().view(-1).numpy()
    batch_b = batch_b_.clone().view(-1).numpy()
    # if offset is None:
    #     threshold_a = np.percentile(np.median(scales_a, 0), 20.0)
    #     threshold_b = np.percentile(np.median(scales_b, 0), 20.0)
    #     offset_to_use = np.maximum(threshold_a, threshold_b)
    # else:
    offset_to_use = offset
    print("OFFSET TO USE : ", offset_to_use)

    batch_ids = np.unique(batch_a)
    assert np.array_equal(batch_ids, np.unique(batch_b))
    lfc = []
    print(batch_ids)
    for batch in batch_ids:
        where_sa = batch_a == batch
        where_sb = batch_b == batch
        samp_a = scales_a[where_sa]
        samp_b = scales_b[where_sb]
        n_samp_total = np.minimum(len(samp_a), len(samp_b))
        samp_a = samp_a[np.random.choice(len(samp_a), n_samp_total, replace=False)]
        samp_b = samp_b[np.random.choice(len(samp_b), n_samp_total, replace=False)]
        lfc.append(np.log2(offset_to_use + samp_a) - np.log2(offset_to_use + samp_b))
    lfc_mf = np.concatenate(lfc, 0)
    is_de = (np.abs(lfc_mf) >= delta).mean(0)
    is_de2 = np.maximum((lfc_mf >= delta).mean(0), (lfc_mf <= -delta).mean(0))
    adjusted_preds_005 = predict_de_genes(is_de, desired_fdr=0.05)
    adjusted_preds_01 = predict_de_genes(is_de, desired_fdr=0.1)
    adjusted_preds_02 = predict_de_genes(is_de, desired_fdr=0.2)

    props = dict(
        lfc_mean=lfc_mf.mean(0),
        lfc_median=np.median(lfc_mf, 0),
        lfc_std=np.std(lfc_mf, 0),
        lfc_other_def=np.log2(scales_a.mean(0) - scales_b.mean(0)),
        is_de=is_de,
        is_de2=is_de2,
        adjusted_preds_005=adjusted_preds_005,
        adjusted_preds_01=adjusted_preds_01,
        adjusted_preds_02=adjusted_preds_02,
    )
    all_creds = []
    for level in _levels:
        dic_key = "lfc_cred_{}".format(level)
        creds = get_fast_cred(lfc_mf, confidence=float(level))
        props[dic_key] = creds
        all_creds.append(creds)
    props["cred_maps"] = pd.Series(all_creds, index=_levels.astype(float) / 100.0)
    return props


def extract_lfc_properties_med(
    delta,
    prop_a,
    prop_b,
    offset=0.0,
    mode=0,
    importance_sampling=False,
    levels=None,
    **subcells_kwargs
):
    scales_a_, batch_a_, w_a_ = prop_a
    scales_b_, batch_b_, w_b_ = prop_b
    ncs = subcells_kwargs.get("ncs", None)
    n_samples_c = subcells_kwargs.get("n_samples_c", None)

    if levels is None:
        _levels = np.array([50, 55, 60, 65, 70, 75, 80, 90, 95])
    else:
        _levels = levels
    n_genes = scales_a_.shape[-1]
    scales_a = scales_a_.clone().numpy()
    scales_b = scales_b_.clone().numpy()

    batch_a = batch_a_.clone().numpy()
    batch_b = batch_b_.clone().numpy()
    w_a = w_a_.clone()
    w_b = w_b_.clone()
    # if offset is None:
    #     threshold_a = np.percentile(np.median(scales_a, 0), 20.0)
    #     threshold_b = np.percentile(np.median(scales_b, 0), 20.0)
    #     offset_to_use_a = np.maximum(threshold_a, threshold_b)
    #     offset_to_use_b = np.maximum(threshold_a, threshold_b)
    # else:
    offset_to_use_a = offset
    offset_to_use_b = offset

    print("OFFSET TO USE : ", offset_to_use_a, offset_to_use_b)
    assert scales_a.ndim == 3 and scales_b.ndim == 3
    batch_ids = np.unique(batch_a)
    assert np.array_equal(batch_ids, np.unique(batch_b))
    lfc = []
    print(batch_ids)
    for batch in batch_ids:
        where_sa = batch_a == batch
        where_sb = batch_b == batch
        assert (where_sa == where_sa[:, [0]]).all()
        assert (where_sb == where_sb[:, [0]]).all()

        samp_a = scales_a[where_sa[:, 0]]
        samp_b = scales_b[where_sb[:, 0]]
        w_ab = w_a[where_sa[:, 0]]
        w_bb = w_b[where_sb[:, 0]]
        if importance_sampling:
            print("Using importance sampling ...")
            # Shapes: n_post_i, n_cells_i
            npost_a, ncells_a = w_ab.shape
            npost_b, ncells_b = w_bb.shape

            p_a = nn.Softmax(0)(w_ab).T
            p_b = nn.Softmax(0)(w_bb).T

            # Shapes n_cells_i, n_post_i
            is_idx_a = db.Categorical(p_a).sample((200,))
            is_idx_b = db.Categorical(p_b).sample((200,))
            # Shapes n_post_new, n_cells_i
            is_idx_a = is_idx_a.unsqueeze(-1).expand([200, ncells_a, n_genes])
            is_idx_b = is_idx_b.unsqueeze(-1).expand([200, ncells_b, n_genes])

            print(torch.tensor(samp_a).shape, is_idx_a.shape)
            samp_a = torch.gather(torch.tensor(samp_a), dim=0, index=is_idx_a).numpy()
            samp_b = torch.gather(torch.tensor(samp_b), dim=0, index=is_idx_b).numpy()
            print(torch.tensor(samp_a).shape, is_idx_a.shape)

        h_a = np.log2(samp_a + offset_to_use_a)
        h_b = np.log2(samp_b + offset_to_use_b)
        print(h_a.shape, w_ab.shape)
        print(h_b.shape, w_bb.shape)

        if mode == 0:
            print("MODE == 0: using medians of logs")
            h_a = np.median(h_a, 1)
            h_b = np.median(h_b, 1)
            n_samp_total = np.maximum(len(h_a), len(h_b))
            h_a = h_a[np.random.choice(len(h_a), n_samp_total, replace=True)]
            h_b = h_b[np.random.choice(len(h_b), n_samp_total, replace=True)]
            lfc.append(h_a - h_b)
        elif mode == 1:
            print("MODE == 1: using means of logs")
            #             print(h_a)
            h_a = np.mean(h_a, 1)
            h_b = np.mean(h_b, 1)
            n_samp_total = np.maximum(len(h_a), len(h_b))
            h_a = h_a[np.random.choice(len(h_a), n_samp_total, replace=True)]
            h_b = h_b[np.random.choice(len(h_b), n_samp_total, replace=True)]
            lfc.append(h_a - h_b)
        elif mode == 2:
            print("MODE == 2: using means of hs")
            #             print(h_a)
            h_a = np.log2(np.mean(samp_a, 1) + offset_to_use_a)
            h_b = np.log2(np.mean(samp_b, 1) + offset_to_use_b)
            n_samp_total = np.maximum(len(h_a), len(h_b))
            h_a = h_a[np.random.choice(len(h_a), n_samp_total, replace=True)]
            h_b = h_b[np.random.choice(len(h_b), n_samp_total, replace=True)]
            lfc.append(h_a - h_b)
        elif mode == 3:
            print("MODE == 3: using mean lfc")
            #             print(h_a)
            h_a = np.mean(h_a, 1)
            h_b = np.mean(h_b, 1)
            n_samp_total = np.maximum(len(h_a), len(h_b))
            h_a = h_a[np.random.choice(len(h_a), n_samp_total, replace=True)]
            h_b = h_b[np.random.choice(len(h_b), n_samp_total, replace=True)]
            lfc.append(h_a - h_b)
        elif mode == 4:
            print("MODE == 4: mean lfc dist")
            # shapes (n_pa, A, n_genes)
            # shapes (n_pb, B, n_genes)
            n_cells = np.maximum(h_a.shape[1], h_b.shape[1])
            n_cells = 2 * n_cells
            n_cells = np.minimum(n_cells, 400)

            r_ca = np.random.choice(h_a.shape[1], n_cells)
            r_cb = np.random.choice(h_b.shape[1], n_cells)
            h_a_ = h_a[:, r_ca]
            h_b_ = h_b[:, r_cb]

            # shapes are now n_post_i, n_cells, n_genes
            n_post = np.maximum(h_a_.shape[0], h_b_.shape[0])
            r_post_a = np.random.choice(h_a_.shape[0], n_post)
            r_post_b = np.random.choice(h_b_.shape[0], n_post)
            h_a_ = h_a_[r_post_a]
            h_b_ = h_b_[r_post_b]

            lfc_ = h_a_ - h_b_
            lfc_ = np.mean(lfc_, 1)
            lfc.append(lfc_)
        elif mode == 5:
            print("MODE == 5: mean lfc dist")
            # shapes (n_pa, A, n_genes)
            # shapes (n_pb, B, n_genes)
            n_cells = np.maximum(h_a.shape[1], h_b.shape[1])
            n_cells = 2 * n_cells
            n_cells = np.minimum(n_cells, 400)

            print("h_a :", h_a.shape)
            print("h_b :", h_b.shape)

            r_ca = np.random.choice(h_a.shape[1], n_cells)
            r_cb = np.random.choice(h_b.shape[1], n_cells)
            h_a_ = h_a[:, r_ca]
            h_b_ = h_b[:, r_cb]

            # shapes are now n_post_i, n_cells, n_genes
            n_post = np.maximum(h_a_.shape[0], h_b_.shape[0])
            r_post_a = np.random.choice(h_a_.shape[0], n_post)
            r_post_b = np.random.choice(h_b_.shape[0], n_post)
            h_a_ = h_a_[r_post_a]
            h_b_ = h_b_[r_post_b]

            lfc_ = h_a_ - h_b_
            lfc.append(lfc_.reshape((-1, h_a.shape[-1])))

        elif mode == 6:
            idx_a_subs = np.random.choice(samp_a.shape[1], size=(ncs * n_samples_c))
            samp_a_subs = samp_a[:, idx_a_subs].reshape(
                (samp_a.shape[0], n_samples_c, ncs, n_genes)
            )
            idx_b_subs = np.random.choice(samp_b.shape[1], size=(ncs * n_samples_c))
            samp_b_subs = samp_b[:, idx_b_subs].reshape(
                (samp_b.shape[0], n_samples_c, ncs, n_genes)
            )
            n_post_samples = np.minimum(samp_a_subs.shape[0], samp_b_subs.shape[0])

            samp_a_subs = samp_a_subs[:n_post_samples]
            samp_b_subs = samp_b_subs[:n_post_samples]

            # n posterior samples, n group cells samples, n cells, n_genes
            _samp_a = samp_a_subs.mean(2)
            _samp_b = samp_b_subs.mean(2)
            _samp_a.shape, _samp_b.shape

            lfcs = (
                np.log2(_samp_a + offset_to_use_a) - np.log2(_samp_b + offset_to_use_b)
            ).reshape((-1, n_genes))
            lfc.append(lfcs)

    lfc_mf = np.concatenate(lfc, 0)
    is_de = (np.abs(lfc_mf) >= delta).mean(0)
    is_de2 = np.maximum((lfc_mf >= delta).mean(0), (lfc_mf <= -delta).mean(0))
    adjusted_preds_005 = predict_de_genes(is_de, desired_fdr=0.05)
    adjusted_preds_01 = predict_de_genes(is_de, desired_fdr=0.1)
    adjusted_preds_02 = predict_de_genes(is_de, desired_fdr=0.2)

    props = dict(
        lfc_mean=lfc_mf.mean(0),
        lfc_median=np.median(lfc_mf, 0),
        lfc_std=np.std(lfc_mf, 0),
        #         lfc_other_def=np.log2(scales_a.mean(0) - scales_b.mean(0)),
        is_de=is_de,
        is_de2=is_de2,
        adjusted_preds_005=adjusted_preds_005,
        adjusted_preds_01=adjusted_preds_01,
        adjusted_preds_02=adjusted_preds_02,
    )
    # levels = np.array([50, 55, 60, 65, 70, 75, 80, 90, 95])
    all_creds = []
    for level in _levels:
        dic_key = "lfc_cred_{}".format(level)
        creds = get_fast_cred(lfc_mf, confidence=float(level))
        props[dic_key] = creds
        all_creds.append(creds)
    props["cred_maps"] = pd.Series(all_creds, index=_levels.astype(float) / 100.0)

    iskey = "is_" if importance_sampling else ""
    props = {iskey + "med{}_".format(mode) + key: val for key, val in props.items()}
    return props


def fdr_score(y_true, y_pred):
    return 1.0 - precision_score(y_true, y_pred)


def true_fdr(y_true, y_pred):
    """
    Computes GT FDR
    """
    n_genes = len(y_true)
    probas_sorted = np.argsort(-y_pred)
    true_fdr_array = np.zeros(n_genes)
    for idx in range(1, len(probas_sorted) + 1):
        y_pred_tresh = np.zeros(n_genes, dtype=bool)
        where_pos = probas_sorted[:idx]
        y_pred_tresh[where_pos] = True
        # print(y_pred_tresh)
        true_fdr_array[idx - 1] = fdr_score(y_true, y_pred_tresh)
    return true_fdr_array


def posterior_expected_fdr(y_pred, fdr_target=0.05) -> tuple:
    """
    Computes posterior expected FDR
    """
    sorted_genes = np.argsort(-y_pred)
    sorted_pgs = y_pred[sorted_genes]
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))

    n_positive_genes = (cumulative_fdr <= fdr_target).sum()
    pred_de_genes = sorted_genes[:n_positive_genes]
    is_pred_de = np.zeros_like(cumulative_fdr).astype(bool)
    is_pred_de[pred_de_genes] = True
    return cumulative_fdr, is_pred_de


def plot_fdr(is_significant_de, preds, where_):
    true_fdr_arr = true_fdr(y_true=is_significant_de[where_], y_pred=preds[where_])
    pe_fdr_arr, y_decision_rule = posterior_expected_fdr(y_pred=preds[where_])
    plt.plot(true_fdr_arr, label="FDR (GT)")
    plt.plot(pe_fdr_arr, label="FDR (PE)")
