import rpy2.robjects as ro
import pandas as pd
import numpy as np
import os
import logging
import rpy2.robjects.numpy2ri
import warnings
from rpy2.rinterface import RRuntimeWarning


def load_files(data, labels, batches, path_to_scripts, n_cells_max=None):
    # loading libraries
    if n_cells_max is not None:
        # Path manipulations
        folder_path = os.path.dirname(data)
        new_data_path = os.path.join(folder_path, "data_subs.npy")
        new_labels_path = os.path.join(folder_path, "labels_subs.npy")
        new_batches_path = os.path.join(folder_path, "batches_subs.npy")

        # Loading & modifying
        data_arr = np.load(data)
        labels_arr = np.loadtxt(labels).astype(np.int64)
        where_a = np.where(labels_arr == 0)[0][:n_cells_max]
        where_b = np.where(labels_arr == 1)[0][:n_cells_max]
        where_keep = np.concatenate([where_a, where_b])

        labels_new = labels_arr[where_keep]
        data_new = data_arr[where_keep]
        logging.info("Subsampling the cell populations ... ")
        ser = pd.Series(labels_new)
        print(ser.groupby(ser).size())
        np.save(new_data_path, data_new)
        np.savetxt(new_labels_path, labels_new)

        if batches is not None:
            batches_arr = np.loadtxt(batches).astype(np.int64)
            batches_new = batches_arr[where_keep]
            np.savetxt(new_batches_path, batches_new)

        data_ = new_data_path
        labels_ = new_labels_path
        batches_ = new_batches_path
    else:
        data_ = data
        labels_ = labels
        batches_ = batches

    warnings.filterwarnings("ignore", category=RRuntimeWarning)
    rpy2.robjects.numpy2ri.activate()
    ro.r["library"]("RcppCNPy")
    ro.r["library"]("BiocParallel")
    ro.r("BiocParallel::register(BiocParallel::MulticoreParam(workers=4))")

    ro.r.assign("path_to_scripts", path_to_scripts)
    ro.r(
        """
        count <- t(npyLoad("{data}", type="integer"))
        condt <-  factor(read.csv("{labels}", header = FALSE)[,1])
        """.format(
            data=data_,
            labels=labels_,
        )
    )
    if batches is not None:
        ro.r(
            """
            batch <- factor(read.csv("{batches}", header = FALSE)[,1])
            """.format(
                batches=batches_,
            )
        )
        ro.r("L <- list(count=count, condt=condt, batch=batch)")
    else:
        ro.r("L <- list(count=count, condt=condt)")


def apply_mast(
    data,
    labels,
    batches,
    path_to_scripts,
):
    load_files(data, labels, batches, path_to_scripts)
    ro.r("script_path <- paste(path_to_scripts, 'apply_MASTcpm.R', sep='/')")
    ro.r("source(script_path)")
    if batches is None:
        ro.r("res <- run_MASTcpm(L)")

    else:
        ro.r("res <- run_MASTcpm_multibatch(L)")
    return pd.DataFrame(ro.r("res$df"))


def apply_deseq2(
    data,
    labels,
    batches,
    path_to_scripts,
    lfc_threshold=0.5,
):
    load_files(data, labels, batches, path_to_scripts)
    ro.r.assign("lfc_threshold", lfc_threshold)
    ro.r("script_path <- paste(path_to_scripts, 'apply_DESeq2.R', sep='/')")
    ro.r("source(script_path)")
    if batches is None:
        ro.r("res <- run_DESeq2(L, lfcThreshold=lfc_threshold)")
    else:
        ro.r("res <- run_DESeq2_multibatch(L, lfcThreshold=lfc_threshold)")
    return pd.DataFrame(ro.r("res$df"))


def apply_edger(
    data,
    labels,
    batches,
    path_to_scripts,
    n_cells_max=None,
):
    load_files(data, labels, batches, path_to_scripts, n_cells_max=n_cells_max)
    ro.r("script_path <- paste(path_to_scripts, 'apply_edgeRLRT.R', sep='/')")
    ro.r("source(script_path)")
    if batches is None:
        ro.r("res <- run_edgeRLRT(L)")
    else:
        ro.r("res <- run_edgeRLRT_multibatch(L)")
    return pd.DataFrame(ro.r("res$df"))


def apply_voom(
    data,
    labels,
    batches,
    path_to_scripts,
):
    load_files(data, labels, batches, path_to_scripts)
    ro.r("script_path <- paste(path_to_scripts, 'apply_voomlimma.R', sep='/')")
    ro.r("source(script_path)")
    if batches is None:
        ro.r("res <- run_voomlimma(L)")
    else:
        ro.r("res <- run_voomlimma_multibatch(L)")
    return pd.DataFrame(ro.r("res$df"))
