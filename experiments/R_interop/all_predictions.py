import time
import numpy as np
from numpy.core.fromnumeric import size
from tqdm import tqdm
import os
import pickle
import pandas as pd
from statsmodels.stats.multitest import multipletests
from . import NDESeq2, NEdgeRLTRT, MAST, NMASTcpm, MLimmaVoom
from .conquer import apply_deseq2, apply_edger, apply_mast, apply_voom


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res


def all_predictions(
    filename,
    n_genes,
    n_picks,
    sizes,
    data_path,
    labels_path,
    size_b=None,
    label_a=0,
    label_b=1,
    normalized_means=None,
    delta=None,
    path_to_scripts=None,
    lfc_threshold: float = 0.5,
    all_nature=True,
    mast_cmat_key="V31",
    batches: str = None,
):
    if os.path.exists(filename):
        return load_pickle(filename)
    n_sizes = len(sizes)

    results = dict()

    # Voom
    lfcs_voom = np.zeros((n_sizes, n_picks, n_genes))
    pvals_voom = np.zeros((n_sizes, n_picks, n_genes))
    pvals_true_voom = np.zeros((n_sizes, n_picks, n_genes))
    times_voom = np.zeros((n_sizes, n_picks))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        b_size = size_b if size_b is not None else size
        print("Size A {} and B {}".format(size, b_size))
        for exp in range(n_picks):
            timer = time.time()
            deseq_inference = MLimmaVoom(
                A=size,
                B=b_size,
                data=data_path,
                labels=labels_path,
                normalized_means=normalized_means,
                delta=delta,
                cluster=(label_a, label_b),
                path_to_scripts=path_to_scripts,
                batches=batches,
            )
            try:
                res_df = deseq_inference.fit()
                timer = time.time() - timer
                times_voom[size_ix, exp] = timer
                lfcs_voom[size_ix, exp, :] = res_df["lfc"].values
                pvals_voom[size_ix, exp, :] = res_df["padj"].values
                pvals_true_voom[size_ix, exp, :] = res_df["pval"].values
            except Exception as e:
                print(e)

    voom_res = dict(
        lfc=lfcs_voom.squeeze(),
        pval=pvals_voom.squeeze(),
        pval_true=pvals_true_voom.squeeze(),
        time=times_voom,
    )
    results["voom"] = voom_res
    save_pickle(data=results, filename=filename)

    # DESeq2
    lfcs_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    pvals_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    pvals_true_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    times_deseq2 = np.zeros((n_sizes, n_picks))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        b_size = size_b if size_b is not None else size
        print("Size A {} and B {}".format(size, b_size))
        for exp in range(n_picks):
            timer = time.time()
            deseq_inference = NDESeq2(
                A=size,
                B=b_size,
                data=data_path,
                labels=labels_path,
                cluster=(label_a, label_b),
                normalized_means=normalized_means,
                delta=delta,
                path_to_scripts=path_to_scripts,
                lfc_threshold=lfc_threshold,
                batches=batches,
            )
            try:
                res_df = deseq_inference.fit()
                timer = time.time() - timer
                times_deseq2[size_ix, exp] = timer
                lfcs_deseq2[size_ix, exp, :] = res_df["lfc"].values
                pvals_deseq2[size_ix, exp, :] = res_df["padj"].values
                pvals_true_deseq2[size_ix, exp, :] = res_df["pval"].values
            except Exception as e:
                print(e)
    deseq_res = dict(
        lfc=lfcs_deseq2.squeeze(),
        pval=pvals_deseq2.squeeze(),
        pval_true=pvals_true_deseq2.squeeze(),
        time=times_deseq2,
    )
    results["deseq2"] = deseq_res
    save_pickle(data=results, filename=filename)

    # EdgeR
    lfcs_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    pvals_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    pvals_true_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    times_edge_r = np.zeros((n_sizes, n_picks))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        b_size = size_b if size_b is not None else size
        print("Size A {} and B {}".format(size, b_size))
        for exp in range(n_picks):
            timer = time.time()
            deseq_inference = NEdgeRLTRT(
                A=size,
                B=b_size,
                data=data_path,
                labels=labels_path,
                normalized_means=normalized_means,
                delta=delta,
                cluster=(label_a, label_b),
                path_to_scripts=path_to_scripts,
                batches=batches,
            )
            try:
                res_df = deseq_inference.fit()
                timer = time.time() - timer
                times_edge_r[size_ix, exp] = timer
                lfcs_edge_r[size_ix, exp, :] = res_df["lfc"].values
                pvals_edge_r[size_ix, exp, :] = res_df["padj"].values
                pvals_true_edge_r[size_ix, exp, :] = res_df["pval"].values
            except Exception as e:
                print(e)

    edger_res = dict(
        lfc=lfcs_edge_r.squeeze(),
        pval=pvals_edge_r.squeeze(),
        pval_true=pvals_true_edge_r.squeeze(),
        time=times_edge_r,
    )
    results["edger"] = edger_res
    save_pickle(data=results, filename=filename)

    # MAST
    lfcs_mast = np.zeros((n_sizes, n_picks, n_genes))
    var_lfcs_mast = np.zeros((n_sizes, n_picks, n_genes))
    pvals_mast = np.zeros((n_sizes, n_picks, n_genes))
    times_mast = np.zeros((n_sizes, n_picks))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        b_size = size_b if size_b is not None else size
        print("Size A {} and B {}".format(size, b_size))
        for exp in range(n_picks):
            if all_nature:
                timer = time.time()
                mast_inference = NMASTcpm(
                    A=size,
                    B=b_size,
                    data=data_path,
                    labels=labels_path,
                    normalized_means=normalized_means,
                    delta=delta,
                    cluster=(label_a, label_b),
                    path_to_scripts=path_to_scripts,
                    batches=batches,
                )
                try:
                    res_df = mast_inference.fit()
                    timer = time.time() - timer
                    times_mast[size_ix, exp] = timer
                    print(res_df.info())
                    # var_lfcs_mast[size_ix, exp, :] = res_df["varLogFC"].values
                    lfcs_mast[size_ix, exp, :] = res_df["lfc"].values
                    pvals_mast[size_ix, exp, :] = res_df["pval"].values
                except Exception as e:
                    print(e)

            else:
                timer = time.time()
                mast_inference = MAST(
                    A=size,
                    B=b_size,
                    data=data_path,
                    labels=labels_path,
                    cluster=(label_a, label_b),
                    local_cmat_key=mast_cmat_key,
                )
                try:
                    res_df = mast_inference.fit(return_fc=True)
                    timer = time.time() - timer
                    times_mast[size_ix, exp] = timer
                    lfcs_mast[size_ix, exp, :] = res_df["lfc"].values
                    pvals_mast[size_ix, exp, :] = res_df["pval"].values
                except Exception as e:
                    print(e)
    mast_res = dict(
        lfc=lfcs_mast.squeeze(),
        pval=pvals_mast.squeeze(),
        var_lfc=var_lfcs_mast,
        time=times_mast,
    )
    results["mast"] = mast_res
    save_pickle(data=results, filename=filename)
    return results


def all_predictionsB(
    filename,
    data_path,
    labels_path,
    path_to_scripts=None,
    lfc_threshold: float = 0.5,
    batches: str = None,
    n_cells_max=None,
):
    results = dict()

    timer = time.time()
    df_ = apply_mast(
        data=data_path,
        labels=labels_path,
        batches=batches,
        path_to_scripts=path_to_scripts,
    )
    print(df_)
    results["MAST"] = dict(
        lfc=df_.lfc.values,
        pval=df_.pval.values,
        time=time.time() - timer,
    )
    save_pickle(data=results, filename=filename)

    timer = time.time()
    df_ = apply_edger(
        data=data_path,
        labels=labels_path,
        batches=batches,
        path_to_scripts=path_to_scripts,
        n_cells_max=n_cells_max,
    )
    results["edgeR"] = dict(
        lfc=df_.lfc.values,
        pval=df_.padj.values,
        time=time.time() - timer,
    )
    save_pickle(data=results, filename=filename)

    timer = time.time()
    df_ = apply_deseq2(
        data=data_path,
        labels=labels_path,
        batches=batches,
        path_to_scripts=path_to_scripts,
        lfc_threshold=0.0,
        # lfc_threshold=0,
    )
    results["DESeq20"] = dict(
        lfc=df_.lfc.values,
        pval=df_.padj.values,
        time=time.time() - timer,
    )
    save_pickle(data=results, filename=filename)

    timer = time.time()
    df_ = apply_deseq2(
        data=data_path,
        labels=labels_path,
        batches=batches,
        path_to_scripts=path_to_scripts,
        lfc_threshold=0.5,
        # lfc_threshold=0,
    )
    results["DESeq2"] = dict(
        lfc=df_.lfc.values,
        pval=df_.padj.values,
        time=time.time() - timer,
    )
    save_pickle(data=results, filename=filename)

    timer = time.time()
    df_ = apply_voom(
        data=data_path,
        labels=labels_path,
        batches=batches,
        path_to_scripts=path_to_scripts,
    )
    results["Voom"] = dict(
        lfc=df_.lfc.values,
        pval=df_.padj.values,
        time=time.time() - timer,
    )
    save_pickle(data=results, filename=filename)
    return (
        pd.DataFrame(results)
        .T.reset_index()
        .rename(columns={"index": "algorithm", "lfc": "lfc_estim", "pval": "de_score"})
        .assign(is_proba=False, is_pval=True)
    )


def all_de_predictions(dict_results, significance_level, delta):
    """

    :param dict_results: dictionnary of dictionnary with hierarchical keys:
        algorithm:
            lfc
            pval
    :param significance_level:
    :param delta:
    :return:
    """
    for algorithm_name in dict_results:
        my_pvals = dict_results[algorithm_name]["pval"]
        my_pvals[np.isnan(my_pvals)] = 1.0

        my_lfcs = dict_results[algorithm_name]["lfc"]
        my_lfcs[np.isnan(my_lfcs)] = 0.0

        if algorithm_name == "deseq2":
            is_de = my_pvals <= significance_level
        elif algorithm_name == "voom":
            is_de = my_pvals <= significance_level

        elif algorithm_name == "edger" or algorithm_name == "edger_robust":
            is_de = my_pvals <= significance_level

        elif algorithm_name == "mast":
            assert my_pvals.ndim == 2
            padjs = []
            for i in range(len(my_pvals)):
                padj = multipletests(my_pvals[i], method="fdr_bh")[1]
                padjs.append(padj[None, :])
            padjs = np.concatenate(padjs, 0)
            dict_results[algorithm_name]["pval"] = padjs
            is_de = (my_pvals <= significance_level) * (np.abs(my_lfcs) >= delta)
        else:
            raise KeyError("No DE policy for {}".format(algorithm_name))
        dict_results[algorithm_name]["is_de"] = is_de
    return dict_results
