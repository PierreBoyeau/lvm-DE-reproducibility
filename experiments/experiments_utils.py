import logging
import sys
import os
from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.cm import get_cmap
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, auc
from sklearn.mixture import GaussianMixture
import plotnine as p9
from arviz import psislw
import pickle

import torch
import torch.nn as nn
from scvi.utils import predict_de_genes
from scvi_utils import plot_fdr, true_fdr, posterior_expected_fdr
from R_interop import all_predictionsB
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE, SCSphereFull

from scvi_utils import (
    compute_lfc,
    extract_lfc_properties,
    extract_lfc_properties_med,
)

from matplotlib_venn import venn3

GOOD_INDEX = [
    "DESeq2",
    "edgeR",
    "MAST",
    "scVI",
    "CellBender",
    "CellBenderS",
    "MF",
    "scVI-DE",
    "MF-AGG",
    "scVI-DE-AGG",
    "MF-OPT",
    "scVI-DE-OPT",
]
SUBSET_SCVI = [
    "MF",
    "scVI-DE",
    "MF-AGG",
    "scVI-DE-AGG",
    "MF-OPT",
    "scVI-DE-OPT",
]
CMAP_LI = get_cmap("tab20")(
    np.arange(
        20,
    )
)


def get_vals(**kwargs):
    savepath = kwargs["savepath"]

    if os.path.exists(savepath):
        return pickle.load(open(savepath, "rb"))

    other_predictions = kwargs["other_predictions"]
    scvi_res = kwargs["scvi_res"]
    mf_res = kwargs["mf_res"]
    iaf_res = kwargs["iaf_res"]
    mf2_res = kwargs["mf2_res"]
    iw_res = kwargs["iw_res"]
    zinb_res = kwargs["zinb_res"]
    sph_res = kwargs["sph_res"]
    # mf_obs_res = kwargs["mf_obs_res"]
    # cellbender_res = kwargs["cellbender_res"]
    # cellbender_simple_res = kwargs["cellbender_simple_res"]
    keyc = "lfc_median"

    dgm_res = [
        dict(
            algorithm="scVI",
            lfc_estim=scvi_res["lfc_median"].iloc[0],
            de_score=scvi_res["abayes_factor"].iloc[0],
        ),
        dict(
            algorithm="scVI-lvm(ELBO)",
            lfc_estim=mf_res["full_eb_lfc_estim"].iloc[0],
            lfc_std=mf_res["full_eb_lfc_std"].iloc[0],
            de_score=mf_res["full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=mf_res["is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="scVI-lvm(ELBO)-OPT",
            lfc_estim=mf_res["opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=mf_res["opt_full_eb_lfc_std"].iloc[0],
            de_score=mf_res["opt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=mf_res["opt_is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="scVI-lvm(ELBO)-OPTf",
            lfc_estim=mf_res["filt_opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=mf_res["filt_opt_full_eb_lfc_std"].iloc[0],
            de_score=mf_res["filt_opt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=mf_res["opt_is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="ScPhere-fullIS",
            lfc_estim=sph_res["full_eb_lfc_estim"].iloc[0],
            lfc_std=sph_res["full_eb_lfc_std"].iloc[0],
            de_score=sph_res["full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=sph_res["is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="ScPhere-fullIS-OPT",
            lfc_estim=sph_res["opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=sph_res["opt_full_eb_lfc_std"].iloc[0],
            de_score=sph_res["opt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=sph_res["opt_is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="ScPhere-fullIS-OPTf",
            lfc_estim=sph_res["filt_opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=sph_res["filt_opt_full_eb_lfc_std"].iloc[0],
            de_score=sph_res["filt_opt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=sph_res["opt_is_med6_cred_maps"].iloc[0],
        ),
        ######################################
        dict(
            algorithm="scVI-DE-fullIS",
            lfc_estim=iw_res["full_eb_lfc_estim"].iloc[0],
            lfc_std=iw_res["full_eb_lfc_std"].iloc[0],
            de_score=iw_res["full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=iw_res["is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="scVI-DE-fullIS-OPT",
            lfc_estim=iw_res["opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=iw_res["opt_full_eb_lfc_std"].iloc[0],
            de_score=iw_res["opt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=iw_res["opt_is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="scVI-DE-fullIS-OPTf",
            lfc_estim=iw_res["filt_opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=iw_res["filt_opt_full_eb_lfc_std"].iloc[0],
            de_score=iw_res["filt_opt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=iw_res["opt_is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="scVI-lvm(ZINB)",
            lfc_estim=zinb_res["full_eb_lfc_estim"].iloc[0],
            lfc_std=zinb_res["full_eb_lfc_std"].iloc[0],
            de_score=zinb_res["full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=zinb_res["is_med6_cred_maps"].iloc[0],
        ),
        dict(
            algorithm="scVI-lvm(ZINB)-OPT",
            lfc_estim=zinb_res["opt_full_eb_lfc_estim"].iloc[0],
            lfc_std=zinb_res["opt_full_eb_lfc_std"].iloc[0],
            de_score=zinb_res["opt_full_eb_is_de2"].iloc[0],
            is_proba=True,
            # credible_intervals=zinb_res["opt_is_med6_cred_maps"].iloc[0],
        ),
    ]
    dgm_res = pd.DataFrame(dgm_res)

    res = pd.concat([dgm_res, other_predictions], ignore_index=True)
    # vals = [
    # ]
    return res


def filter_algo(val, subset_names):
    if subset_names is not None:
        if val["name"] not in subset_names:
            return True
    return False


def get_lfc_mse(vals, lfc_gt, where, subset_names=None):
    res = []
    for it, val in enumerate(vals):
        algorithm_name = val["name"]
        preds = copy(val["lfc_estim"])
        # for pick, preds_pick in enumerate(preds):
        preds[np.isinf(preds)] = 0.0
        preds = np.nan_to_num(preds)

        preds_pick_ok = preds[where]
        lfc_gt_ok = lfc_gt[where]

        mse = (0.5 * (preds_pick_ok - lfc_gt_ok) ** 2).mean()
        mae = np.abs(preds_pick_ok - lfc_gt_ok).mean()
        res = res + [
            dict(mse=mse, mae=mae, name=algorithm_name),
        ]
    df_errors = pd.DataFrame(res)
    return df_errors


def plot_lfc_scatter(vals, lfc_gt, where, subset_names=None):
    res = []
    ncols = 3
    nrows = int(np.ceil(len(vals) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(9, 9),
    )

    ticks = [-2, 0, 2]
    for it, val in enumerate(vals):
        if filter_algo(val, subset_names):
            continue
        row = it // ncols
        col = it % ncols
        print(row, col)

        val["lfc_estim"][np.isinf(val["lfc_estim"])] = 0.0
        val["lfc_estim"] = np.nan_to_num(val["lfc_estim"])

        lfc_ok = lfc_gt[where]
        preds_ok = val["lfc_estim"][where]

        plt.sca(axes[row, col])
        plt.title(val["name"], fontdict=dict(fontsize=18))
        plt.scatter(
            lfc_ok,
            preds_ok,
            alpha=0.5,
        )
        plt.plot([-3.0, 3.0], [-3.0, 3.0], ls="--", c=".3")

        plt.xlabel("LFC GT")
        if col == 0:
            plt.ylabel("Estimated LFC")
            plt.xlim(-5.0, 5.0)
            plt.ylim(-5.0, 5.0)
        axes[row, col].set_yticks(ticks)
        axes[row, col].set_yticklabels(ticks)
        scores_p = pearsonr(preds_ok, lfc_ok)[0]
        scores_s = spearmanr(preds_ok, lfc_ok)[0]

        plt.text(
            0.5,
            -2.5,
            "$r$={0:.2f}".format(scores_p),
        )
        res.append(dict(name=val["name"], pearson=scores_p, spearmanr=scores_s))
    res = pd.DataFrame(res)
    return fig, axes, res


def ggplot_lfc_scatter(vals, lfc_gt, where, subset_names=None):
    res = []
    for it, val in enumerate(vals):
        if filter_algo(val, subset_names):
            continue

        val["lfc_estim"][np.isinf(val["lfc_estim"])] = 0.0
        val["lfc_estim"] = np.nan_to_num(val["lfc_estim"])

        lfc_ok = lfc_gt[where]
        preds_ok = val["lfc_estim"][where]
        scores_p = pearsonr(preds_ok, lfc_ok)[0]

        res.append(
            pd.DataFrame(
                {
                    "Algorithm": val["name"] + "@{0:.2f}".format(scores_p),
                    "LFC (Est.)": lfc_ok,
                    "LFC (Reference)": preds_ok,
                    "Pearson": scores_p,
                }
            )
        )
    res = pd.concat(res)

    ggp = (
        p9.ggplot(res, p9.aes(x="LFC (Reference)", y="LFC (Est.)"))
        + p9.facet_wrap("Algorithm")
        + p9.geom_point(color="#0099ff", alpha=0.5)
        + p9.theme_classic(base_size=15)
    )
    return ggp


def plot_heteroskedastic_props(vals, gene_means, subset_names=None):
    fig, ax = plt.subplots(figsize=(5, 3))

    for it, val in enumerate(vals):
        if filter_algo(val, subset_names):
            continue
        plt.scatter(
            gene_means[val["lfc_std"] != 0.0],
            val["lfc_std"][val["lfc_std"] != 0.0],
            alpha=0.5,
            label=val["name"],
        )
    plt.xscale("log")
    plt.xlabel("Gene mean")
    plt.ylabel("Posterior LFC std")
    plt.legend()
    return fig, ax


def plot_probas_hist(vals, where, subset_names=None):
    fig, ax = plt.subplots(figsize=(5, 3))
    for it, val in enumerate(vals):
        if filter_algo(val, subset_names):
            continue
        plt.hist(
            val["de_score"],
            alpha=0.5,
            label=val["name"],
        )
    plt.legend()
    return fig, ax


def plot_tprfdr_curves(vals, is_significant_de, where, **kwargs):
    """
    vals
    is_significant_de
    where
    fdr_levels
    plot_curve
    markers
    cmap_li
    """
    fig = kwargs.get("fig", None)
    ax = kwargs.get("ax", None)
    display_legend = kwargs.get("display_legend", True)
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    fdr_levels = kwargs.get("fdr_levels", [0.1])
    plot_curve = kwargs.get("plot_curve", True)
    markers = kwargs.get("markers", ["o"])
    cmap_li = kwargs.get(
        "cmap_li",
        get_cmap("tab10")(
            np.arange(
                10,
            )
        ),
    )

    algos_names = [val["name"] for val in vals]
    classif_res = {name: dict() for name in algos_names}

    for (lvl_idx, fdr_target), marker in zip(enumerate(fdr_levels), markers):
        plt.vlines(x=fdr_target, ymin=0.0, ymax=1.0, linestyle="--")
        plt.plot(
            fdr_target,
            0.2,
            c="white",
            marker=marker,
            markersize=15,
            markeredgewidth=4,
            markeredgecolor="black",
            markerfacecolor="white",
        )
        for i, algorithm_preds in enumerate(vals):
            algorithm_name = algorithm_preds["name"]
            is_pval = algorithm_preds.get("is_pval", False)
            is_proba = algorithm_preds.get("is_proba", False)
            scores_pick = np.array(copy(algorithm_preds["de_score"]))
            kwargs = dict()
            if np.isnan(is_pval):
                is_pval = False
            if lvl_idx == 0:
                kwargs = dict(label=algorithm_name)
            # scores_pick = scores_pick + 1e-8 * np.random.randn(len(scores_pick))
            scores_pick[np.isinf(scores_pick)] = 0.0

            y_target = is_significant_de[where]
            y_pred = scores_pick[where]
            if is_pval:
                y_pred = -y_pred

            pre_arr, rec_arr, _ = precision_recall_curve(y_target, y_pred)
            tpr_arr = rec_arr
            fdr_arr = 1.0 - pre_arr

            if is_pval:
                preds_pick = scores_pick <= fdr_target
            elif is_proba:
                preds_pick = predict_de_genes(scores_pick, desired_fdr=fdr_target)
            else:
                continue
            y_pred_bin = preds_pick[where]
            tpr = recall_score(y_true=y_target, y_pred=y_pred_bin)
            fdr = 1.0 - precision_score(y_true=y_target, y_pred=y_pred_bin)

            plt.plot(
                fdr,
                tpr,
                c=cmap_li[i],
                marker=marker,
                fillstyle="full",
                markersize=12,
                markeredgewidth=3,
                markeredgecolor="black",
                markerfacecolor=cmap_li[i],
                **kwargs,
            )
            if lvl_idx == 0:
                classif_res[algorithm_name]["AUC"] = auc(rec_arr, pre_arr)

            classif_res[algorithm_name]["FDR{}".format(fdr_target)] = fdr
            classif_res[algorithm_name]["TPR{}".format(fdr_target)] = tpr
            classif_res[algorithm_name]["FDR{}_ERR".format(fdr_target)] = (
                fdr - fdr_target
            )

            if plot_curve and lvl_idx == 0:
                plt.plot(
                    fdr_arr,
                    tpr_arr,
                    linewidth=2,
                    c=cmap_li[i],
                )

    classif_res = pd.DataFrame(classif_res)
    if display_legend:
        plt.legend()
    return fig, ax, classif_res


def plot_fdr_curves(vals, is_significant_de, where):
    # Compute number of required subplots
    n_plots = 0
    for val in vals:
        is_prob = int(val.get("is_proba", False))
        n_plots += is_prob

    fig, axes = plt.subplots(nrows=n_plots, figsize=(4, n_plots * 4))
    plot_i = 0
    for val in vals:
        is_prob = val.get("is_proba", False)
        alg_name = val.get("name")
        if is_prob:
            plt.sca(axes[plot_i])
            probs = copy(val["de_score"])

            plot_fdr(is_significant_de=is_significant_de, preds=probs, where_=where)
            axes[plot_i].set_ylabel(alg_name, rotation=0.0, labelpad=20)
            plot_i += 1
            plt.legend()
    return fig, axes


def ggplot_fdr_curves(vals, is_significant_de, where):
    # Compute number of required subplots
    dfs = []
    n_genes = is_significant_de[where].shape[-1]
    for val in vals:
        is_prob = val.get("is_proba", False)
        alg_name = val.get("name")
        if is_prob:
            probs = copy(val["de_score"])
            true_fdr_arr = true_fdr(
                y_true=is_significant_de[where], y_pred=probs[where]
            )
            pe_fdr_arr, y_decision_rule = posterior_expected_fdr(y_pred=probs[where])
            plot_df = pd.DataFrame({"FDR": true_fdr_arr}).assign(
                Algorithm=alg_name, Type="GT"
            )
            plot_df.loc[:, "# of genes"] = np.arange(n_genes)
            dfs.append(plot_df)

            plot_df = pd.DataFrame({"FDR": pe_fdr_arr}).assign(
                Algorithm=alg_name, Type="PE"
            )
            plot_df.loc[:, "# of genes"] = np.arange(n_genes)
            dfs.append(plot_df)
    dfs = pd.concat(dfs)

    ggp = (
        p9.ggplot(dfs, p9.aes(x="# of genes", y="FDR", color="Type"))
        + p9.facet_wrap("Algorithm")
        + p9.geom_line(size=3)
        + p9.theme_classic(base_size=16)
        # + p9.geom_line()
        # + p9.geom_point(x="# of genes", y="FDR (PE)")
    )
    # fig
    # p9.
    return ggp


def get_calibration(vals, lfc_gt, where, subset_names=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    cal_errors = []
    for val in vals:
        if filter_algo(val, subset_names):
            continue
        cred_inter = val["credible_intervals"]

        cred_levels = []
        empirical_freqs = []
        print(cred_inter)
        for cred_lvl, intervals in cred_inter.items():
            gt_is_in_lvl = (lfc_gt <= intervals[1]) * (intervals[0] <= lfc_gt)
            empirical_freq = gt_is_in_lvl.mean()
            cred_levels.append(cred_lvl)
            empirical_freqs.append(empirical_freq)
        ser_res = pd.Series(empirical_freqs, index=cred_levels)
        plt.plot(cred_levels, empirical_freqs, label=val["name"])

        cal_diff = ser_res.values - ser_res.index.values
        cal_errors.append(
            dict(
                name=val["name"],
                calibration_mse=(cal_diff ** 2).mean(),
                calibration_mae=np.abs(cal_diff).mean(),
            )
        )
    plt.legend()
    cal_errors = pd.DataFrame(cal_errors)
    return (fig, ax, cal_errors)


def load_scvi_model_if_exists(model, filename):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        train_model = False
    else:
        train_model = True
    return model, train_model


def gmm_fit(data: torch.Tensor, q=0.95, mode_coeff=0.6, min_thres=0.3):
    """Returns delta estimate using GMM technique"""
    # Custom definition
    gmm = GaussianMixture(n_components=3)
    gmm.fit(data.cpu().detach()[:, None])
    vals = np.sort(gmm.means_.squeeze())
    res = mode_coeff * np.abs(vals[[0, -1]]).mean()
    # if res <= 0.05:
    #     # TODO replace this condition with a consideration on the mixture properties, i.e, compare
    #     # TODO: means with stds or something
    #     # TODO: To ensure proper behavior
    #     res = np.quantile(data.abs().detach().cpu(), q=q)
    logging.info("DELTA VALUE: {}".format(res))
    logging.info("Using mode coefficient {}".format(mode_coeff))
    res = np.maximum(min_thres, res)
    return res


def get_indices(y_all, label_a, label_b, n_ex_a, n_ex_b):
    idx_a = np.where(y_all == label_a)[0]
    idx_a = np.random.choice(idx_a, n_ex_a, replace=False)

    idx_b = np.where(
        (y_all == label_b)
        * (
            ~np.isin(np.arange(len(y_all)), idx_a)
        )  # avoid using same samples for negative controls
    )[0]
    idx_b = np.random.choice(idx_b, n_ex_b, replace=False)
    return idx_a, idx_b


def compute_predictions(
    idx_a,
    idx_b,
    scvi_kwargs,
    others_kwargs,
    dataset,
    q0,
    delta,
    do_batch_specific="separation",
    q_gmm=0.95,
    **subcells_kwargs
):

    # scVI parameters
    N_EPOCHS = scvi_kwargs["N_EPOCHS"]
    N_SAMPLES = scvi_kwargs["N_SAMPLES"]
    do_sph_deep = scvi_kwargs.get("do_sph_deep", True)

    n_genes = dataset.nb_genes

    filename = others_kwargs["filename"]
    folder_path = others_kwargs["folder_path"]
    PATH_TO_SCRIPTS = others_kwargs["PATH_TO_SCRIPTS"]

    labels = np.array([0] * len(idx_a) + [1] * len(idx_b))
    all_inds = np.concatenate([idx_a, idx_b])
    xobs = dataset.X[all_inds]
    xobs[:, 0] += 1
    batches = dataset.batch_indices[all_inds].squeeze()
    data_path = os.path.join(folder_path, "data_path.npy")
    labels_path = os.path.join(folder_path, "labels_path.npy")
    batches_path = os.path.join(folder_path, "batches_path.npy")
    np.save(data_path, xobs.astype(int))
    np.savetxt(labels_path, labels)
    np.savetxt(batches_path, batches)
    if len(np.unique(batches)) == 1:
        batches_path = None

    other_predictions = all_predictionsB(
        filename=filename,
        data_path=data_path,
        labels_path=labels_path,
        path_to_scripts=PATH_TO_SCRIPTS,
        lfc_threshold=delta,
        batches=batches_path,
    )
    # other_predictions = all_de_predictions(
    #     other_predictions, significance_level=q0, delta=delta
    # )

    iw_filename = os.path.join(folder_path, "iw_model5.pt")
    zinb_filename = os.path.join(folder_path, "zinb_model5.pt")
    mf2_filename = os.path.join(folder_path, "mf2_model5.pt")
    iaf_filename = os.path.join(folder_path, "iaf_model5.pt")
    mf_filename = os.path.join(folder_path, "mf_model5.pt")
    sph_filename = os.path.join(folder_path, "sph_model.pt")
    # ## Baseline models
    mdl_sph_kwargs = dict(
        n_genes=dataset.nb_genes,
        n_batches=dataset.n_batches,
        n_latent=11,
        do_depth_reg=do_sph_deep,
        constant_pxr=True,
        cell_specific_px=False,
        scale_norm="softmax",
        library_nn=True,
        dropout_rate=0.0,
        deep_architecture=do_sph_deep,
    )
    trainer_sph_kwargs = dict(
        train_library=True,
        k=25,
        loss_type="IWELBO",
        batch_size=1024,
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
    trainer_sph = UnsupervisedTrainer(
        model=mdl_sph, gene_dataset=dataset, **trainer_sph_kwargs
    )
    trainer_sph.train(n_epochs=N_EPOCHS, lr=lr_sph)
    mdl_sph.eval()
    # Empirical experiments show that without dropout regularization, models will overfit in terms of reconstruction error after a while
    # eval mode does not work as well as train mode

    print()
    torch.save(mdl_sph.state_dict(), sph_filename)
    # mdl_sph.eval()

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
        batch_size=1024,
        weight_decay=1e-4,
        optimizer_type="adam",
        test_indices=[],
    )
    mdl_iw = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_iw_kwargs)
    # mdl_iw, train_iw = load_scvi_model_if_exists(mdl_iw, filename=iw_filename)
    trainer_iw = UnsupervisedTrainer(
        model=mdl_iw, gene_dataset=dataset, **trainer_iw_kwargs
    )
    lr_iw = trainer_iw.find_lr()
    # if train_iw:
    mdl_iw = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_iw_kwargs)
    # mdl_iw, train_iw = load_scvi_model_if_exists(mdl_iw, filename=iw_filename)
    trainer_iw = UnsupervisedTrainer(
        model=mdl_iw, gene_dataset=dataset, **trainer_iw_kwargs
    )
    logging.info("Using learning rate {}".format(lr_iw))
    trainer_iw.train(n_epochs=N_EPOCHS, lr=lr_iw)
    mdl_iw.eval()
    torch.save(mdl_iw.state_dict(), iw_filename)

    mdl_zinb_kwargs = dict(
        n_latent=10,
        # n_latent=3,
        dropout_rate=0.0,
        decoder_dropout_rate=0.0,
        reconstruction_loss="zinb",
        dispersion="gene",
        n_layers=1,
        use_batch_norm=False,
        use_weight_norm=False,
        use_layer_norm=True,
        with_activation=nn.ReLU(),
    )
    trainer_zinb_kwargs = dict(
        train_library=True,
        k=25,
        batch_size=1024,
        weight_decay=1e-4,
        loss_type="IWELBO",
        optimizer_type="adam",
        test_indices=[],
    )
    mdl_zinb = VAE(
        n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_zinb_kwargs
    )
    trainer_zinb = UnsupervisedTrainer(
        model=mdl_zinb, gene_dataset=dataset, **trainer_zinb_kwargs
    )
    lr_zinb = trainer_zinb.find_lr()
    mdl_zinb = VAE(
        n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_zinb_kwargs
    )
    trainer_zinb = UnsupervisedTrainer(
        model=mdl_zinb, gene_dataset=dataset, **trainer_zinb_kwargs
    )
    logging.info("Using learning rate {}".format(lr_zinb))
    trainer_zinb.train(n_epochs=N_EPOCHS, lr=lr_zinb)
    mdl_zinb.eval()
    torch.save(mdl_zinb.state_dict(), zinb_filename)
    # mdl_iw.eval()

    mdl_mf_kwargs = dict(
        n_latent=10,
        # n_latent=3,
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
    trainer_mf_kwargs = dict(
        train_library=True,
        k=1,
        batch_size=1024,
        weight_decay=1e-4,
        loss_type="ELBO",
        optimizer_type="adamw",
        test_indices=[],
    )
    mdl_mf = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_mf_kwargs)
    trainer_mf = UnsupervisedTrainer(
        model=mdl_mf, gene_dataset=dataset, **trainer_mf_kwargs
    )
    lr_mf = trainer_mf.find_lr()
    mdl_mf = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, **mdl_mf_kwargs)
    trainer_mf = UnsupervisedTrainer(
        model=mdl_mf, gene_dataset=dataset, **trainer_mf_kwargs
    )
    logging.info("Using learning rate {}".format(lr_mf))
    trainer_mf.train(n_epochs=N_EPOCHS, lr=lr_mf)
    mdl_mf.eval()
    torch.save(mdl_mf.state_dict(), mf_filename)
    print()
    # ## Perfs comparison

    print("SCALES SAMPLING ...")

    mf_res = []
    mf2_res = []
    iw_res = []
    iaf_res = []
    scvi_res = []
    mf_obs_res = []
    zinb_res = []
    sph_res = []
    # for _ in range(1):
    # idx_a = np.where(y_all == label_a)[0]
    # idx_a = np.random.choice(idx_a, n_ex_a, replace=False)

    # idx_b = np.where(
    #     (y_all == label_b)
    #     * (
    #         ~np.isin(np.arange(len(y_all)), idx_a)
    #     )  # avoid using same samples for negative controls
    # )[0]
    # idx_b = np.random.choice(idx_b, n_ex_b, replace=False)

    with torch.no_grad():
        scales_a, scales_b = compute_lfc(
            trainer_mf,
            idx_a,
            idx_b,
            n_samples=N_SAMPLES,
            importance_sampling=False,
            train_library=True,
        )

        # Computing reference scVI
        sc_a = scales_a[0].reshape((-1, n_genes))
        sc_b = scales_b[0].reshape((-1, n_genes))
        n_samp = np.minimum(len(sc_a), len(sc_b))
        lfc_dist = np.log2(sc_a[:n_samp]) - np.log2(sc_b[:n_samp])
        p_up = (sc_a[:n_samp] >= sc_b[:n_samp]).float().mean(0)
        bf = np.log10(1e-16 + p_up) - np.log10(1e-16 + 1.0 - p_up)
        bf[np.isinf(bf)] = 0.0
        bf[np.isnan(bf)] = 0.0
        scvi_props = dict(
            bayes_factor=bf.numpy(),
            abayes_factor=np.abs(bf).numpy(),
            lfc_mean=lfc_dist.mean(0).numpy(),
            lfc_median=np.median(lfc_dist, 0),
        )

    # mf_props = get_props(
    #     trainer_mf,
    #     idx_a=idx_a,
    #     idx_b=idx_b,
    #     offset=offset,
    #     delta=delta,
    #     **subcells_kwargs
    #     # n_samples=N_SAMPLES,
    #     # q_gmm=q_gmm,
    # )
    mf_props_eb = get_eb_full(
        trainer=trainer_mf,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=delta,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=False,
    )
    mf_props_eb_opt = get_eb_full(
        trainer=trainer_mf,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=False,
    )
    mf_props_eb_opt_f = get_eb_full(
        trainer=trainer_mf,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=False,
        filter_cts=True,
        # coef_truncate=0.5,
    )
    mf_props = {**mf_props_eb, **mf_props_eb_opt, **mf_props_eb_opt_f}

    # mdl_iw.train()
    # iw_props = get_props(
    #     trainer_iw,
    #     idx_a=idx_a,
    #     idx_b=idx_b,
    #     offset=offset,
    #     delta=delta,
    #     importance_sampling=True,
    #     **subcells_kwargs,
    #     # n_samples=N_SAMPLES,
    #     # q_gmm=q_gmm,
    # )

    logging.info("Using batch management procedure {}".format(do_batch_specific))

    iw_props_eb = get_eb_full(
        trainer=trainer_iw,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=delta,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=True,
    )
    iw_props_eb_opt = get_eb_full(
        trainer=trainer_iw,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=True,
    )
    iw_props_eb_opt_f = get_eb_full(
        trainer=trainer_iw,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=True,
        filter_cts=True,
        # coef_truncate=0.5,
    )
    # iw_props = {**iw_props, **iw_props_eb, **iw_props_eb_opt}
    iw_props = {**iw_props_eb, **iw_props_eb_opt, **iw_props_eb_opt_f}

    mdl_sph.train()
    # sph_props = get_props(
    #     trainer_sph,
    #     idx_a=idx_a,
    #     idx_b=idx_b,
    #     offset=offset,
    #     delta=delta,
    #     importance_sampling=True,
    #     **subcells_kwargs,
    #     # n_samples=N_SAMPLES,
    #     # q_gmm=q_gmm,
    # )
    sph_props_eb = get_eb_full(
        trainer=trainer_sph,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=delta,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=True,
    )
    sph_props_eb_opt = get_eb_full(
        trainer=trainer_sph,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=True,
    )
    sph_props_eb_opt_f = get_eb_full(
        trainer=trainer_sph,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=True,
        filter_cts=True,
        # coef_truncate=0.5,
    )
    # sph_props = {**sph_props, **sph_props_eb, **sph_props_eb_opt}
    sph_props = {**sph_props_eb, **sph_props_eb_opt, **sph_props_eb_opt_f}

    zinb_eb_opt = get_eb_full(
        trainer=trainer_zinb,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=None,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        do_normalize=False,
    )

    zinb_eb = get_eb_full(
        trainer=trainer_zinb,
        idx_a=idx_a,
        idx_b=idx_b,
        offset=None,
        delta=delta,
        # n_samples=30000,
        n_samples=30000,
        # keeping_k_best_subsamples=20000,
        do_batch_specific=do_batch_specific,
        posterior_chunks=200,
        # do_normalize=True,
        # coef_truncate=0.5,
    )
    zinb_props = {**zinb_eb, **zinb_eb_opt}

    sph_res.append(sph_props)
    # mf2_res.append(mf2_props)
    mf_res.append(mf_props)
    # mf_obs_res.append(mf_obs_props)
    # iaf_res.append(iaf_props)
    scvi_res.append(scvi_props)
    iw_res.append(iw_props)
    zinb_res.append(zinb_props)

    mf_res = pd.DataFrame(mf_res)
    mf2_res = pd.DataFrame(mf2_res)
    mf_obs_res = pd.DataFrame(mf_obs_res)
    scvi_res = pd.DataFrame(scvi_res)
    zinb_res = pd.DataFrame(zinb_res)
    iaf_res = pd.DataFrame(iaf_res)
    iw_res = pd.DataFrame(iw_res)
    sph_res = pd.DataFrame(sph_res)

    # cellbender_simple_res = get_cellbender_preds(
    #     dataset,
    #     idx_a=idx_a,
    #     idx_b=idx_b,
    #     delta=delta,
    #     offset=offset,
    #     prior_mode="simple",
    # )
    # cellbender_res = get_cellbender_preds(
    #     dataset, idx_a=idx_a, idx_b=idx_b, delta=delta, offset=offset, prior_mode="full"
    # )

    return dict(
        other_predictions=other_predictions,
        scvi_res=scvi_res,
        mf_res=mf_res,
        mf2_res=mf2_res,
        iw_res=iw_res,
        iaf_res=iaf_res,
        zinb_res=zinb_res,
        mf_obs_res=mf_obs_res,
        # cellbender_simple_res=cellbender_simple_res,
        # cellbender_res=cellbender_res,
        trainer_iw=trainer_iw,
        trainer_mf=trainer_mf,
        sph_res=sph_res,
    )


def evaluate_extended(vals, is_significant_de, lfc_info, fdr_target=0.1, kbest=None):
    #     def evaluate_multi(ser):
    #         is_de_pred = ser["is_de_pred"]
    #         is_significant_de = ser["is_significant_de"]
    #         return (

    #         )
    #         return dict(
    # #             dict(
    #                 fdr=1.0 - precision_score(is_significant_de, is_de_pred),
    #                 fdr_neg=is_de_pred.mean(),
    # #             )
    #         )

    if kbest is None:
        kbest = is_significant_de.sum()
    all_res = []
    for i, algorithm_preds in enumerate(vals):
        algorithm_name = algorithm_preds["name"]
        is_pval = algorithm_preds.get("is_pval", False)
        is_proba = algorithm_preds.get("is_proba", False)
        scores_pick = copy(algorithm_preds["de_score"])

        scores_pick = scores_pick + 1e-8 * np.random.randn(len(scores_pick))
        scores_pick[np.isinf(scores_pick)] = 0.0

        y_pred = scores_pick
        if is_pval:
            y_pred = -y_pred

        if is_pval:
            preds_pick = scores_pick <= fdr_target
        elif is_proba:
            preds_pick = predict_de_genes(scores_pick, desired_fdr=fdr_target)
        else:
            continue

        best_k = np.argsort(y_pred)[::-1][:kbest]
        best_k = np.isin(np.arange(len(is_significant_de)), best_k)
        #         print(best_k)
        dfm = pd.DataFrame(
            dict(
                is_de_pred=preds_pick[best_k],
                dcategory=lfc_info.category[best_k],
                is_significant_de=is_significant_de[best_k],
            )
        )
        #         ser_res = dfm.groupby("dcategory").apply(evaluate_multi)
        #         ser_res = dfm.groupby("dcategory").is_de_pred.sum() / dfm.groupby("dcategory").size()
        ser_res = dfm.groupby("dcategory").is_de_pred.sum() / kbest

        ser_res.name = algorithm_name
        all_res.append(ser_res)
    return pd.DataFrame(all_res).loc[:, ["db", "de", "dm", "dp"]]


def get_props(
    my_trainer,
    idx_a,
    idx_b,
    delta,
    offset=1e-10,
    offset_inplace=0.0,
    importance_sampling=False,
    n_samples=5000,
    q_gmm=0.95,
    levels=None,
    **subcells_kwargs
):
    with torch.no_grad():
        scales_a, scales_b = compute_lfc(
            my_trainer,
            idx_a,
            idx_b,
            n_samples=n_samples,
            importance_sampling=False,
            train_library=True,
        )
    _props_agg = extract_lfc_properties(
        dataset=my_trainer.gene_dataset,
        delta=delta,
        prop_a=scales_a,
        prop_b=scales_b,
        offset=offset,
        levels=levels,
    )
    _props = {**_props_agg}
    # lfc_point_estimates = torch.tensor(_props["lfc_median"])
    for mode in [5, 6]:
        _props_ = extract_lfc_properties_med(
            dataset=my_trainer.gene_dataset,
            delta=delta,
            prop_a=scales_a,
            prop_b=scales_b,
            offset=offset,
            offset_inplace=offset_inplace,
            mode=mode,
            importance_sampling=importance_sampling,
            levels=levels,
            **subcells_kwargs,
        )
        _props = {**_props, **_props_}

        prop_keys = _props_.keys()
        good_key = [key for key in prop_keys if key.endswith("lfc_median")][0]

        lfc_point_estimates = torch.tensor(_props_[good_key])
        delta_opt = gmm_fit(lfc_point_estimates, q=q_gmm)
        _props_opt = extract_lfc_properties_med(
            dataset=my_trainer.gene_dataset,
            delta=delta_opt,
            prop_a=scales_a,
            prop_b=scales_b,
            offset=offset,
            offset_inplace=offset_inplace,
            mode=mode,
            importance_sampling=importance_sampling,
            levels=levels,
            **subcells_kwargs,
        )
        _props_opt = {"opt_" + key: val for key, val in _props_opt.items()}
        _props = {**_props, **_props_opt}

    # qqprobs = get_qqprob(
    #     my_trainer,
    #     idx_a=idx_a,
    #     idx_b=idx_b,
    #     offset=offset,
    #     delta=1.0,
    # )
    # qqprobs_auto = get_qqprob(
    #     my_trainer,
    #     idx_a=idx_a,
    #     idx_b=idx_b,
    #     offset=offset,
    #     delta=None,
    # )
    # _props = {
    #     "qq_is_de": qqprobs,
    #     "qq_is_de_auto": qqprobs_auto,
    #     **_props,
    # }
    return _props


def get_eb_full(
    trainer,
    idx_a,
    idx_b,
    offset=0.0,
    delta=None,
    n_samples=10000,
    do_batch_specific=True,
    posterior_chunks=1000,
    do_normalize=False,
    do_clamp=False,
    subsample_size=None,
    subsample_replace=True,
    return_dists=False,
    mode_coeff=0.6,
    min_thres=0.3,
    include_lib=False,
    keeping_k_best_subsamples=None,
    filter_cts=False,
    v2_fix=True,
    coef_truncate=None,
    n_ll_samples=5000,
    plugin=False,
):
    if plugin:
        key = "px_scale"
    else:
        key = "px_scale_iw"
    logging.info("Pop A {} & Pop B {}".format(idx_a.shape[0], idx_b.shape[0]))
    psis_info = dict()
    where_zero_a = np.max(trainer.gene_dataset.X[idx_a], 0) == 0
    where_zero_b = np.max(trainer.gene_dataset.X[idx_b], 0) == 0

    def _get_offset(scales_a, scales_b, offset):
        if offset is not None:
            res = offset
        else:
            max_scales_a = scales_a.max(0).values
            max_scales_b = scales_b.max(0).values
            if where_zero_a.sum() >= 1:
                artefact_scales_a = max_scales_a[where_zero_a].numpy()
                eps_a = np.percentile(artefact_scales_a, q=90)
            else:
                eps_a = 1e-10

            if where_zero_b.sum() >= 1:
                artefact_scales_b = max_scales_b[where_zero_b].numpy()
                eps_b = np.percentile(artefact_scales_b, q=90)
            else:
                eps_b = 1e-10
            print("Data inferred offsets:", eps_a, eps_b)
            res = np.maximum(eps_a, eps_b)
        print("using offset:", res)
        return res

    def _get_scales(outs_a, outs_b, do_clamp, offset_of_use, n_samples_total):
        ha = outs_a[key]
        hb = outs_b[key]
        if do_clamp:
            log_ha = (ha).clamp(min=offset_of_use).log2()[:n_samples_total]
            log_hb = (hb).clamp(min=offset_of_use).log2()[:n_samples_total]
        else:
            log_ha = (ha + offset_of_use).log2()[:n_samples_total]
            log_hb = (hb + offset_of_use).log2()[:n_samples_total]
        return log_ha, log_hb

    def get_psis(outs):
        return psislw(outs["log_ws"].numpy())[1].squeeze()

    if do_batch_specific == "separation":
        logging.info("Using mode separation")
        batch_data = trainer.gene_dataset.batch_indices.squeeze()
        batch_a = batch_data[idx_a]
        batch_b = batch_data[idx_b]
        batches_of_interest = np.intersect1d(batch_a, batch_b)

        n_samples_per_batch = int(np.ceil(n_samples / len(batches_of_interest)))
        lfc = []
        for bat in batches_of_interest:
            idx_a_bat = np.where(batch_a == bat)[0]
            idx_b_bat = np.where(batch_b == bat)[0]
            idx_a_ = idx_a[idx_a_bat]
            idx_b_ = idx_b[idx_b_bat]

            if (len(idx_a_) <= 1) or (len(idx_b_) <= 1):
                continue
            if not include_lib:
                post_a = trainer.create_posterior(indices=idx_a_)
                outs_a = post_a.get_latents_full(
                    n_samples_overall=n_samples_per_batch,
                    posterior_chunks=posterior_chunks,
                    subsample_size=subsample_size,
                    subsample_replace=subsample_replace,
                    keeping_k_best_subsamples=keeping_k_best_subsamples,
                    filter_cts=filter_cts,
                    coef_truncate=coef_truncate,
                    v2_fix=v2_fix,
                    n_ll_samples=n_ll_samples,
                )

                post_b = trainer.create_posterior(indices=idx_b_)
                outs_b = post_b.get_latents_full(
                    n_samples_overall=n_samples_per_batch,
                    posterior_chunks=posterior_chunks,
                    subsample_size=subsample_size,
                    subsample_replace=subsample_replace,
                    keeping_k_best_subsamples=keeping_k_best_subsamples,
                    filter_cts=filter_cts,
                    coef_truncate=coef_truncate,
                    v2_fix=v2_fix,
                    n_ll_samples=n_ll_samples,
                )

                psis_info["khat_a{}".format(bat)] = get_psis(outs_a)
                psis_info["khat_b{}".format(bat)] = get_psis(outs_b)
            else:
                post_a = trainer.create_posterior(indices=idx_a_)
                outs_a = post_a.get_latents_full_lib(
                    n_samples_overall=n_samples_per_batch,
                    posterior_chunks=posterior_chunks,
                )

                post_b = trainer.create_posterior(indices=idx_b_)
                outs_b = post_b.get_latents_full_lib(
                    n_samples_overall=n_samples_per_batch,
                    posterior_chunks=posterior_chunks,
                )
            offset_of_use = _get_offset(
                scales_a=outs_a[key],
                scales_b=outs_b[key],
                offset=offset,
            )
            log_ha, log_hb = _get_scales(
                outs_a, outs_b, do_clamp, offset_of_use, n_samples_per_batch
            )
            _lfc = log_ha - log_hb
            logging.info("LFC MEAN in batch {}: {}".format(bat, _lfc.mean()))
            lfc.append(_lfc)
        lfc = torch.cat(lfc, 0)
    elif do_batch_specific == "infer":
        logging.info("Using mode infer")
        batch_data = trainer.gene_dataset.batch_indices.squeeze()
        batch_a = batch_data[idx_a]
        batch_b = batch_data[idx_b]
        batches_of_interest = np.union1d(batch_a, batch_b)

        n_samples_per_batch = int(np.ceil(n_samples / len(batches_of_interest)))
        lfc = []
        for bat in batches_of_interest:

            if not include_lib:
                post_a = trainer.create_posterior(indices=idx_a)
                outs_a = post_a.get_latents_full(
                    n_samples_overall=n_samples_per_batch,
                    posterior_chunks=posterior_chunks,
                    custom_batch_index=bat,
                    subsample_size=subsample_size,
                    subsample_replace=subsample_replace,
                    keeping_k_best_subsamples=keeping_k_best_subsamples,
                    filter_cts=filter_cts,
                    coef_truncate=coef_truncate,
                )

                post_b = trainer.create_posterior(indices=idx_b)
                outs_b = post_b.get_latents_full(
                    n_samples_overall=n_samples_per_batch,
                    posterior_chunks=posterior_chunks,
                    custom_batch_index=bat,
                    subsample_size=subsample_size,
                    subsample_replace=subsample_replace,
                    keeping_k_best_subsamples=keeping_k_best_subsamples,
                    filter_cts=filter_cts,
                    coef_truncate=coef_truncate,
                )
            else:
                post_a = trainer.create_posterior(indices=idx_a)
                outs_a = post_a.get_latents_full_lib(
                    n_samples_overall=n_samples_per_batch,
                    posterior_chunks=posterior_chunks,
                    custom_batch_index=bat,
                )

                post_b = trainer.create_posterior(indices=idx_b)
                outs_b = post_b.get_latents_full_lib(
                    n_samples_overall=n_samples_per_batch,
                    posterior_chunks=posterior_chunks,
                    custom_batch_index=bat,
                )
            offset_of_use = _get_offset(
                scales_a=outs_a[key],
                scales_b=outs_b[key],
                offset=offset,
            )
            log_ha, log_hb = _get_scales(
                outs_a, outs_b, do_clamp, offset_of_use, n_samples_per_batch
            )
            _lfc = log_ha - log_hb
            logging.info("LFC MEAN in batch {}: {}".format(bat, _lfc.mean()))
            lfc.append(_lfc)
        lfc = torch.cat(lfc, 0)
    elif do_batch_specific == "disjoint":
        logging.info("Using mode disjoint")
        logging.info("TEST")
        assert not include_lib
        batch_data = trainer.gene_dataset.batch_indices.squeeze()
        batch_a = batch_data[idx_a]
        batch_b = batch_data[idx_b]
        # unique_batch_a = np.unique(batch_a)
        # unique_batch_b = np.unique(batch_b)
        ser_a = pd.Series(batch_a)
        sz_a = ser_a.groupby(ser_a).size()
        unique_batch_a = sz_a[sz_a >= 2].index.values
        ser_b = pd.Series(batch_b)
        sz_b = ser_b.groupby(ser_b).size()
        unique_batch_b = sz_b[sz_b >= 2].index.values
        n_samples_per_batch_a = int(np.ceil(n_samples / len(unique_batch_a)))
        n_samples_per_batch_b = int(np.ceil(n_samples / len(unique_batch_b)))

        ha = []
        hb = []
        print("unique batches", unique_batch_a, unique_batch_b)
        for bat_a in unique_batch_a:
            print(bat_a)
            idx_a_bat = np.where(batch_a == bat_a)[0]
            idx_a_ = idx_a[idx_a_bat]
            print(idx_a_.shape)
            if len(idx_a_) <= 1:
                continue
            post_a = trainer.create_posterior(indices=idx_a_)
            outs_a = post_a.get_latents_full(
                n_samples_overall=n_samples_per_batch_a,
                posterior_chunks=posterior_chunks,
                subsample_size=subsample_size,
                subsample_replace=subsample_replace,
                keeping_k_best_subsamples=keeping_k_best_subsamples,
                filter_cts=filter_cts,
                coef_truncate=coef_truncate,
                v2_fix=v2_fix,
                n_ll_samples=n_ll_samples,
            )
            ha.append(outs_a[key])
        for bat_b in unique_batch_b:
            print(bat_b)
            idx_b_bat = np.where(batch_b == bat_b)[0]
            idx_b_ = idx_b[idx_b_bat]
            print(idx_b_.shape)
            if len(idx_b_) <= 1:
                continue
            post_b = trainer.create_posterior(indices=idx_b_)
            outs_b = post_b.get_latents_full(
                n_samples_overall=n_samples_per_batch_b,
                posterior_chunks=posterior_chunks,
                subsample_size=subsample_size,
                subsample_replace=subsample_replace,
                keeping_k_best_subsamples=keeping_k_best_subsamples,
                filter_cts=filter_cts,
                coef_truncate=coef_truncate,
                v2_fix=v2_fix,
                n_ll_samples=n_ll_samples,
            )
            hb.append(outs_b[key])
        ha = torch.cat(ha, 0)[:n_samples]
        hb = torch.cat(hb, 0)[:n_samples]
        # Randomize samples to ensure that all pairs of batches are considered
        ha = ha[np.random.permutation(len(ha))]
        hb = hb[np.random.permutation(len(hb))]
        offset_of_use = _get_offset(
            scales_a=ha,
            scales_b=hb,
            offset=offset,
        )
        lfc = (ha + offset_of_use).log2() - (hb + offset_of_use).log2()
    else:
        logging.info("Using mode None")
        if not include_lib:
            post_a = trainer.create_posterior(indices=idx_a)
            outs_a = post_a.get_latents_full(
                n_samples_overall=n_samples,
                posterior_chunks=posterior_chunks,
                subsample_size=subsample_size,
                subsample_replace=subsample_replace,
                filter_cts=filter_cts,
                coef_truncate=coef_truncate,
                v2_fix=v2_fix,
                n_ll_samples=n_ll_samples,
            )

            post_b = trainer.create_posterior(indices=idx_b)
            outs_b = post_b.get_latents_full(
                n_samples_overall=n_samples,
                posterior_chunks=posterior_chunks,
                subsample_size=subsample_size,
                subsample_replace=subsample_replace,
                filter_cts=filter_cts,
                coef_truncate=coef_truncate,
                v2_fix=v2_fix,
                n_ll_samples=n_ll_samples,
            )
        else:
            post_a = trainer.create_posterior(indices=idx_a)
            outs_a = post_a.get_latents_full_lib(
                n_samples_overall=n_samples,
                posterior_chunks=posterior_chunks,
            )

            post_b = trainer.create_posterior(indices=idx_b)
            outs_b = post_b.get_latents_full_lib(
                n_samples_overall=n_samples,
                posterior_chunks=posterior_chunks,
            )
        offset_of_use = _get_offset(
            scales_a=outs_a[key],
            scales_b=outs_b[key],
            offset=offset,
        )
        log_ha, log_hb = _get_scales(outs_a, outs_b, do_clamp, offset_of_use, n_samples)
        lfc = log_ha - log_hb
    if do_normalize:
        lfc = lfc - lfc.mean()
    prefix = "opt_" if delta is None else ""
    prefix = "filt_" + prefix if filter_cts else prefix
    if delta is None:
        delta = gmm_fit(lfc.mean(0), mode_coeff=mode_coeff, min_thres=min_thres)
    pde_sup = (lfc >= delta).float().mean(0).cpu().numpy()
    pde_inf = (lfc <= -delta).float().mean(0).cpu().numpy()
    pde2 = np.maximum(pde_sup, pde_inf)

    ppos = (lfc >= 0.0).float().mean(0).numpy()
    pneg = (lfc <= 0.0).float().mean(0).numpy()
    lfc_estim = np.median(lfc.numpy(), 0)
    lfc_estim2 = lfc_estim * (2 * (np.maximum(ppos, pneg) - 0.5))
    res = dict(
        full_eb_is_de2=pde2,
        full_eb_lfc_estim=lfc_estim,
        full_eb_lfc_std=np.std(lfc.numpy(), 0),
        full_eb_ppos=ppos,
        full_eb_pneg=pneg,
        full_eb_lfc_estim2=lfc_estim2,
    )
    if return_dists:
        res["ha"] = outs_a[key] + offset_of_use
        res["hb"] = outs_b[key] + offset_of_use
        res["lfc"] = lfc
    res = {prefix + key: val for key, val in res.items()}
    res = {**res, **psis_info}
    return res
