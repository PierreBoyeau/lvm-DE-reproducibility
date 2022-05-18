import copy
import os
import logging
from tqdm.auto import tqdm

from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.distributions as db

from matplotlib import pyplot as plt
from scipy.stats import kde, entropy
from scvi.utils import softmax
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

# from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    SubsetRandomSampler,
    RandomSampler,
)
from scipy.special import betainc
from torch.distributions import Normal

from scvi.dataset import GeneExpressionDataset
from scvi.models.log_likelihood import (
    compute_elbo,
    compute_reconstruction_error,
    compute_marginal_log_likelihood,
)

logger = logging.getLogger(__name__)


class SequentialSubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)


class Posterior:
    r"""The functional data unit. A `Posterior` instance is instantiated with a model and a gene_dataset, and
    as well as additional arguments that for Pytorch's `DataLoader`. A subset of indices can be specified, for
    purposes such as splitting the data into train/test or labelled/unlabelled (for semi-supervised learning).
    Each trainer instance of the `Trainer` class can therefore have multiple `Posterior` instances to train a model.
    A `Posterior` instance also comes with many methods or utilities for its corresponding data.


    :param model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
    :param gene_dataset: A gene_dataset instance like ``CortexDataset()``
    :param shuffle: Specifies if a `RandomSampler` or a `SequentialSampler` should be used
    :param indices: Specifies how the data should be split with regards to train/test or labelled/unlabelled
    :param use_cuda: Default: ``True``
    :param data_loader_kwarg: Keyword arguments to passed into the `DataLoader`

    Examples:

    Let us instantiate a `trainer`, with a gene_dataset and a model

        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels, use_cuda=True)
        >>> trainer = UnsupervisedTrainer(vae, gene_dataset)
        >>> trainer.train(n_epochs=50)

    A `UnsupervisedTrainer` instance has two `Posterior` attributes: `train_set` and `test_set`
    For this subset of the original gene_dataset instance, we can examine the differential expression,
    log_likelihood, entropy batch mixing, ... or display the TSNE of the data in the latent space through the
    scVI model

        >>> trainer.train_set.differential_expression_stats()
        >>> trainer.train_set.reconstruction_error()
        >>> trainer.train_set.entropy_batch_mixing()
        >>> trainer.train_set.show_t_sne(n_samples=1000, color_by="labels")

    """

    def __init__(
        self,
        model,
        gene_dataset: GeneExpressionDataset,
        shuffle=False,
        indices=None,
        use_cuda=True,
        data_loader_kwargs=dict(),
    ):
        """

        When added to annotation, has a private name attribute
        """
        self.model = model
        self.gene_dataset = gene_dataset
        self.to_monitor = []
        self.use_cuda = use_cuda

        if indices is not None and shuffle:
            raise ValueError("indices is mutually exclusive with shuffle")
        if indices is None:
            if shuffle:
                sampler = RandomSampler(gene_dataset)
            else:
                sampler = SequentialSampler(gene_dataset)
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            sampler = SubsetRandomSampler(indices)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        self.data_loader_kwargs.update(
            {"collate_fn": gene_dataset.collate_fn_builder(), "sampler": sampler}
        )
        self.data_loader = DataLoader(gene_dataset, num_workers=8, **self.data_loader_kwargs)

    def accuracy(self):
        pass

    accuracy.mode = "max"

    @property
    def indices(self):
        if hasattr(self.data_loader.sampler, "indices"):
            return self.data_loader.sampler.indices
        else:
            return np.arange(len(self.gene_dataset))

    @property
    def nb_cells(self):
        if hasattr(self.data_loader.sampler, "indices"):
            return len(self.data_loader.sampler.indices)
        else:
            return self.gene_dataset.nb_cells

    def __iter__(self):
        return map(self.to_cuda, iter(self.data_loader))

    def to_cuda(self, tensors):
        return [t.cuda() if self.use_cuda else t for t in tensors]

    def update(self, data_loader_kwargs):
        posterior = copy.copy(self)
        posterior.data_loader_kwargs = copy.copy(self.data_loader_kwargs)
        posterior.data_loader_kwargs.update(data_loader_kwargs)
        posterior.data_loader = DataLoader(
            self.gene_dataset, **posterior.data_loader_kwargs
        )
        return posterior

    def sequential(self, batch_size=128):
        return self.update(
            {
                "batch_size": batch_size,
                "sampler": SequentialSubsetSampler(indices=self.indices),
            }
        )

    def corrupted(self):
        return self.update(
            {"collate_fn": self.gene_dataset.collate_fn_builder(corrupted=True)}
        )

    def uncorrupted(self):
        return self.update({"collate_fn": self.gene_dataset.collate_fn_builder()})

    @torch.no_grad()
    def get_key(self, my_key, n_samples=200, train_library=True):
        alls = []
        for i_batch, tensors in enumerate(self.sequential()):
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            outputs = self.model(
                sample_batch,
                batch_index=batch_index,
                local_l_mean=local_l_mean,
                local_l_var=local_l_var,
                loss=None,
                n_samples=n_samples,
                train_library=train_library,
            )
            keyv = outputs[my_key].detach().cpu()
            alls.append(keyv)
        concat_dim = int(
            np.where(
                [dim == self.data_loader_kwargs["batch_size"] for dim in alls[0].shape]
            )[0][0]
        )
        return torch.cat(alls, dim=concat_dim)

    @torch.no_grad()
    def get_qz_m(self):
        all_qs = []
        for tensors in self.sequential():
        #     print(tensors)
            sample_batch, _, _, batch_index, label = tensors
            with torch.no_grad():
                outs = self.model.inference(sample_batch, batch_index=batch_index)
            qz = outs["qz_m"].squeeze().detach().cpu()
            if qz.ndim == 1:
                # In this case this means that this is only one cell in one of the 
                # sub batches.
                qz = qz.unsqueeze(0)
            all_qs.append(qz)
        all_qs = torch.cat(all_qs, 0)
        return all_qs

    @torch.no_grad()
    def marginal_llb(self, custom_batch_index, n_samples=5000, n_samples_per_pass=200):
        logging.info("Using {} samples for log-evidence estimation ...".format(n_samples))
        log_px = []
        n_passes = int(n_samples / n_samples_per_pass)
        for tensors in self.sequential(batch_size=128):
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            if custom_batch_index is not None:
                my_batch_index = custom_batch_index * torch.ones_like(batch_index)
            else:
                my_batch_index = batch_index
            
            log_ratios = []
            for _ in range(n_passes):
                outputs = self.model(
                    sample_batch,
                    local_l_mean=local_l_mean,
                    local_l_var=local_l_var,
                    batch_index=my_batch_index,
                    loss=None,
                    n_samples=n_samples_per_pass,
                    train_library=True,
                    # multiplicative_std=multiplicative_std,
                )
                log_ratios.append(outputs["log_ratio"].detach().cpu())
            log_ratios = torch.cat(log_ratios, dim=0)
            log_px_est = torch.logsumexp(log_ratios, 0, keepdim=True) - np.log10(log_ratios.shape[0])
            # print(log_px_est.shape)
            log_px.append(log_px_est.detach().cpu())
        log_px = torch.cat(log_px, dim=-1)
        logging.info("... done!")
        return log_px

    @torch.no_grad()
    def get_latents_full(
        self,
        n_samples_overall,
        device="cpu",
        custom_batch_index=None,
        posterior_chunks=1000,
        subsample_size=None,
        subsample_replace=False,
        keeping_k_best_subsamples=None,
        filter_cts=False,
        return_full=False,
        coef_truncate=None,
        n_ll_samples=5000,
        v2_fix=True,
        # multiplicative_std=None,
    ):
        """Provides unified latent samples, scales, and log-ratios


        :param n_samples: [description]
        :type n_samples: [type]
        :param other: [description], defaults to None
        :type other: [type], optional
        :param device: [description], defaults to "cpu"
        :type device: str, optional
        :return: [description]
        :rtype: [type]
        """
        if filter_cts:
            qz_m = self.get_qz_m().numpy()
            try:
                idx_filt = EllipticEnvelope().fit_predict(qz_m)
            except ValueError:
                logging.warning("Could not properly estimate Cov!, using all samples")
                idx_filt = np.ones(qz_m.shape[0], dtype=bool)
            if (idx_filt == 1).sum() <= 1:
                # Avoid cases where only one cell is kept which is problematic for numerical reasons
                idx_filt = np.ones(qz_m.shape[0])
            post = self.update(
                {
                    "sampler": SequentialSubsetSampler(indices=self.indices[idx_filt == 1]),
                }
            )
            print(self.indices[idx_filt == 1])
            logging.info("Filtering observations: Keeping {} cells from original {} sample size".format(post.indices.shape, qz_m.shape[0]))
        else:
            post = self
        n_cells = len(post.indices)
        n_samples_per_cell = int(np.ceil(n_samples_overall / n_cells))
        logging.info("Using {} posterior samples per cell".format(n_samples_per_cell))
        logging.info("Step 1: Getting posterior samples")
        z_samples = []
        batch_samples = []
        print(custom_batch_index)
        with torch.no_grad():
            for tensors in post.sequential(batch_size=128):
                sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
                if custom_batch_index is not None:
                    my_batch_index = custom_batch_index * torch.ones_like(batch_index)
                else:
                    my_batch_index = batch_index
                outputs = post.model(
                    sample_batch,
                    local_l_mean=local_l_mean,
                    local_l_var=local_l_var,
                    batch_index=my_batch_index,
                    loss=None,
                    n_samples=n_samples_per_cell,
                    train_library=True,
                    # multiplicative_std=multiplicative_std,
                )
                zkey = "z"
                z = outputs[zkey]
                library = outputs["library"]

                n_cells_batch = sample_batch.shape[0]
                batch_samples.append(
                    my_batch_index.unsqueeze(0)
                    .expand(n_samples_per_cell, n_cells_batch, 1)
                    .cpu()
                )
                z_samples.append(z.cpu())
                # libraries.append(library.cpu())
        n_latent = z.shape[-1]
        z_samples = torch.cat(z_samples, dim=1).reshape(
            n_cells * n_samples_per_cell, 1, n_latent
        )
        batch_samples = torch.cat(batch_samples, dim=1).reshape(
            n_cells * n_samples_per_cell, 1, 1
        )
        # libraries = torch.cat(libraries, dim=1).view(n_cells * n_samples_per_cell, 1, n_latent)

        logging.info("Step 2: Compute overall importance weights")
        # log_ws = torch.logsumexp(log_pz + log_px_zl, -1) - torch.logsumexp(log_qz_x, -1)
        out_probs = post.get_probs(
            z_samples=z_samples,
            custom_batch_index=custom_batch_index,
            posterior_chunks=posterior_chunks,
            v2_fix=v2_fix,
            n_ll_samples=n_ll_samples,
            n_samples_per_pass=posterior_chunks,
            # multiplicative_std=multiplicative_std,
        )
        log_qz_x = out_probs["log_qz_x"]
        log_pz = out_probs["log_pz"]
        log_px_zl = out_probs["log_px_zl"]
        log_ws = out_probs["log_ws"]

        logging.info("Step 3: Compute scales from original batches")
        # The library is useless but we still need to feed some value to the decoder
        # Right now, the pooled z samples and batch indices have shape (N_total samples, 1, x)
        # Before feeding them to the decoder, we need to squeeze the second channel
        z_samples = z_samples.squeeze(1)
        batch_samples = batch_samples.squeeze(1)
        print(batch_samples.unique())
        lib_dummy = 5 * torch.ones_like(batch_samples).float()

        px_scale = []
        for z_samp, lib_samp, batch_samp in zip(
            z_samples.split(posterior_chunks),
            lib_dummy.split(posterior_chunks),
            batch_samples.split(posterior_chunks),
        ):
            z_samp = z_samp.cuda()
            lib_samp = lib_samp.cuda()
            batch_samp = batch_samp.cuda()
            # px_scale_, _, _, _ = post.model.decoder("gene", z_samp, lib_samp, batch_samp)
            px_scale_ = post.model.get_scale_(z_samp, lib_samp, batch_samp)
            px_scale.append(px_scale_.cpu())
        px_scale = torch.cat(px_scale, 0)

        if keeping_k_best_subsamples is not None:
            logging.info("Subsampling obtained samples to speed up IS procedure")
            assert (
                log_ws.shape[0]
                == px_scale.shape[0]
                == z_samples.shape[0]
                == batch_samples.shape[0]
            ), "Unexpected sample shapes {} {} {} {}".format(
                log_ws.shape[0], px_scale.shape[0], z.shape[0], batch_samples.shape[0]
            )
            best_k_idx = torch.argsort(-log_ws)[:keeping_k_best_subsamples]
            log_ws = log_ws[best_k_idx]
            px_scale = px_scale[best_k_idx]
            z = z_samples[best_k_idx]
            batch_samples = batch_samples[best_k_idx]

        if coef_truncate is not None:
            logging.info("Truncating log weights")
            n = log_ws.shape[0]
            log_chat = (torch.logsumexp(log_ws, 0) - np.log(n))
            thresh_n = log_chat + coef_truncate * np.log(n)
            log_ws = torch.min(log_ws, thresh_n)
        ws = nn.Softmax()(log_ws)
        n_samples = ws.shape[0]
        iw_idx = db.Categorical(probs=ws).sample((n_samples,))
        if subsample_size is not None:
            arr = np.arange(n_samples)
            iw_idx = np.random.choice(
                arr, size=subsample_size, replace=subsample_replace, p=ws.numpy()
            )
            logger.info("Using SIR with replacement {}".format(subsample_replace))
            iw_idx = torch.tensor(iw_idx)
        px_scale_iw = px_scale[iw_idx]

        logging.info("ESS: {}".format(1 / (ws **2).sum().item()))
        res = dict(
            px_scale=px_scale.cpu(),
            log_ws=log_ws.cpu(),
            z=z_samples.cpu(),
            batch_samples=batch_samples.cpu(),
            ws=ws.cpu(),
            iw_idx=iw_idx.cpu(),
            px_scale_iw=px_scale_iw.cpu(),
        )
        if return_full:
            res["log_qz_x"] = log_qz_x
            res["log_pz"] = log_pz
            res["log_px_zl"] = log_px_zl
        return res

    @torch.no_grad()
    def get_probs(
        self,
        z_samples,
        custom_batch_index=None,
        posterior_chunks=1000,
        n_ll_samples=5000,
        n_samples_per_pass=200,
        return_genes=False,
        v2_fix=True,
        # multiplicative_std=None,
    ):
        n_latent = z_samples.shape[-1]
        log_ratios = []
        log_qz_x = []
        log_pz = []
        log_px_zl = []
        # log_px = []
        log_pxg_zl = []
        with torch.no_grad():
            post = self.sequential(128)
            n_exs = len(post.data_loader)
            for tensors in tqdm(post, total=n_exs):
                sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
                n_cells_batch = sample_batch.shape[0]

                log_qz_x_ = []
                log_pz_ = []
                log_px_zl_ = []
                log_pxg_zl_ = []
                for z_samp in z_samples.split(posterior_chunks):
                    n_total_samples = z_samp.shape[0]
                    _z_samples = z_samp.expand(
                        n_total_samples, n_cells_batch, n_latent
                    ).cuda()

                    # print(_z_samples.shape, batch_index.shape)
                    if custom_batch_index is not None:
                        my_batch_index = custom_batch_index * torch.ones_like(
                            batch_index
                        )
                    else:
                        my_batch_index = batch_index
                    outputs = self.model._inference_is(
                        sample_batch,
                        _z_samples,
                        # local_l_mean=local_l_mean,
                        # local_l_var=local_l_var,
                        batch_index=my_batch_index,
                        n_samples=1,
                        # multiplicative_std=multiplicative_std,
                    )
                    # n_particles = outputs["log_qz_x"].shape[0]
                    # log_px_b = torch.logsumexp(outputs["log_pz"] + outputs["log_px_zl"] - outputs["log_qz_x"], 0) - np.log(n_particles)

                    log_qz_x_.append(outputs["log_qz_x"].cpu())
                    log_pz_.append(outputs["log_pz"].cpu())
                    # log_px_.append(log_px_b.cpu().unsqueeze(0))
                    log_px_zl_.append(outputs["log_px_zl"].cpu())
                    if return_genes:
                        log_pxg_zl_.append(outputs["log_pxg_zl"].cpu())
                    # Get log px_z
                    # In the case of scVI, we take the mean
                    # for the library size to avoid the need
                    # to marginalize l

                log_qz_x_batch = torch.cat(log_qz_x_, dim=0)
                log_pz_batch = torch.cat(log_pz_, dim=0)
                log_px_zl_batch = torch.cat(log_px_zl_, dim=0)

                # n_particles = log_px_zl_batch.shape[0]
                # log_px_batch = torch.logsumexp(log_px_zl_batch + log_pz_batch - log_qz_x_batch, dim=0, keepdim=True) - np.log(n_particles)
                # log_ratios_batch = log_px_zl_batch + log_pz_batch - log_qz_x_batch
                # log_ratios.append(log_ratios_batch.cpu())
                log_qz_x.append(log_qz_x_batch.cpu())
                log_pz.append(log_pz_batch.cpu())
                # log_px.append(log_px_batch.cpu())
                log_px_zl.append(log_px_zl_batch.cpu())

                if return_genes:
                    log_pxg_zl_batch = torch.cat(log_pxg_zl_, dim=0)
                    log_pxg_zl.append(log_pxg_zl_batch.cpu())


        # log_ratios = torch.cat(log_ratios, dim=1)
        log_qz_x = torch.cat(log_qz_x, dim=-1)
        log_pz = torch.cat(log_pz, dim=-1)
        log_px_zl = torch.cat(log_px_zl, dim=-1)
        # log_px = torch.cat(log_px, dim=-1)
        log_px = self.marginal_llb(custom_batch_index=custom_batch_index, n_samples=n_ll_samples, n_samples_per_pass=n_samples_per_pass)
        print(log_px.shape)
        print(log_px_zl.shape)
        if return_genes:
            log_pxg_zl = torch.cat(log_pxg_zl, dim=1)

        # importance weights
        if v2_fix:
            logging.info("px reweight")
            # log_ws = log_pz[..., 0] + torch.logsumexp(log_px_zl - log_px, -1) - torch.logsumexp(log_qz_x, -1)
            log_ws = log_pz[..., 0] + torch.logsumexp(
                log_px_zl - log_px - torch.logsumexp(log_qz_x, -1, keepdim=True), 
                -1
            )
        else:
            log_ws = log_pz[..., 0] + torch.logsumexp(
                log_px_zl - torch.logsumexp(log_qz_x, -1, keepdim=True), 
                -1
            )
#         log_pz
# log_px_zs
# log_px
# log_qz
        print("log_pz {} {}".format(log_pz.min().item(), log_pz.max().item()))
        print("log_px_zs {} {}".format(log_px_zl.min().item(), log_px_zl.max().item()))
        print("log_px {} {}".format(log_px.min().item(), log_px.max().item()))
        print("log_qz_x {} {}".format(log_qz_x.min().item(), log_qz_x.max().item()))
        print("log_probas {} {}".format(log_ws.min().item(), log_ws.max().item()))
        return dict(
            log_qz_x=log_qz_x,
            log_pz=log_pz,
            log_px_zl=log_px_zl,
            log_ws=log_ws,
            log_pxg_zl=log_pxg_zl,
        )

    @torch.no_grad()
    def get_latents(
        self,
        n_samples=1,
        other=None,
        device="cuda",
        train_library=None,
        sample_zmean=False,
    ):
        """
        Computes all quantities of interest for DE in a sequential order

        WARNING: BATCH EFFECTS NOT TAKEN INTO ACCOUNT AS FOR NOW
        # TODO: TAKE THEM INTO ACCOUNT (NOT THAT HARD)

        :param n_samples:
        :param other:
        :return:
        """
        zs = []
        labels = []
        scales = []
        libraries = []
        log_probas = []
        batch_indices = []
        n_bio_batches = self.gene_dataset.n_batches
        with torch.no_grad():
            for tensors in self.sequential():
                sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
                outputs = self.model(
                    sample_batch,
                    # batch_index,
                    local_l_mean=local_l_mean,
                    local_l_var=local_l_var,
                    batch_index=batch_index,
                    loss=None,
                    n_samples=n_samples,
                    train_library=train_library,
                )
                zkey = "z" if not sample_zmean else "qz_m"
                z = outputs[zkey]
                library = outputs["library"]
                log_probas_batch = outputs["log_ratio"]

                # log_probas_batch = self.model.ratio_loss(
                #     sample_batch,
                #     local_l_mean,
                #     local_l_var,
                #     return_mean=False,
                #     train_library=train_library,
                #     outputs=outputs,
                # )
                scale_batch = []
                new_log_probas = []
                batch_indices_batch = []
                for bio_batch in range(n_bio_batches):
                    new_log_probas.append(log_probas_batch)

                    batch_index = bio_batch * torch.ones_like(sample_batch[..., [0]])
                    try:
                        scale_batch.append(
                            self.model.decoder.forward("gene", z, library, batch_index)[
                                0
                            ]
                        )
                    except:
                        scale_batch.append(
                            self.model.get_scale(z, library, batch_index)
                        )
                    batch_index_save = bio_batch * torch.ones_like(
                        scale_batch[-1][..., 0]
                    )
                    batch_indices_batch.append(batch_index_save)
                # each elem of scale_batch has shape (n_samples, n_batch, n_genes)
                batch_indices_batch = torch.cat(batch_indices_batch, dim=0)
                scale_batch = torch.cat(scale_batch, dim=0)
                log_probas_batch = torch.cat(new_log_probas, dim=0)

                label = label.to(device=device)
                z = z.to(device=device)
                log_probas_batch = log_probas_batch.to(device=device)
                scale_batch = scale_batch.to(device=device)
                batch_indices_batch = batch_indices_batch.to(device=device)

                # print(label.device, z.device, scale_batch.device)
                labels.append(label)
                zs.append(z)
                libraries.append(library)
                scales.append(scale_batch)
                batch_indices.append(batch_indices_batch)
                log_probas.append(log_probas_batch)

        if n_samples > 1:
            # Then each z element has shape (n_samples, n_batch, n_latent)
            # Hence we concatenate on dimension 1
            zs = torch.cat(zs, dim=1)
            libraries = torch.cat(libraries, dim=1)
            scales = torch.cat(scales, dim=1)
            batch_indices = torch.cat(batch_indices, dim=1)

            # zs = zs.transpose(0, 1)
            # zs = zs.transpose(1, 2)
            # New shape (n_batch, b)
        else:
            zs = torch.cat(zs)
        log_probas = torch.cat(log_probas, dim=1)
        # Final log_probas shape (n_samples, n_cells)
        labels = torch.cat(labels)
        return dict(
            z=zs,
            library=libraries,
            label=labels,
            scale=scales,
            log_probas=log_probas,
            batch_index=batch_indices,
        )

    @torch.no_grad()
    def get_data(self):
        """

        :return:
        """
        xs, labels = [], []
        for tensors in self.sequential():
            sample_batch, _, _, batch_index, label = tensors
            xs.append(sample_batch.cpu())
            labels.append(label.cpu())
        xs = torch.cat(xs)
        labels = torch.cat(labels)
        return xs, labels

    @torch.no_grad()
    def elbo(self):
        elbo = compute_elbo(self.model, self)
        logger.debug("ELBO : %.4f" % elbo)
        return elbo

    elbo.mode = "min"

    @torch.no_grad()
    def elbo_ratio_loss(self, n_batches_used=10):
        elbo = 0.0
        n_samples = 0
        for i_batch, tensors in enumerate(self):
            n_samples += 1
            sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors[
                :5
            ]  # general fish case
            elbo_batch = self.model.ratio_loss(
                sample_batch,
                local_l_mean,
                local_l_var,
                batch_index=batch_index,
                y=labels,
                return_mean=True,
                # train_library=True
            )
            elbo += elbo_batch.sum().item()

            if n_samples >= n_batches_used:
                break
        n_samples = len(self.indices)
        return elbo / n_samples

    elbo_ratio_loss.mode = "min"

    @torch.no_grad()
    def reconstruction_error(self):
        reconstruction_error = compute_reconstruction_error(self.model, self)
        logger.debug("Reconstruction Error : %.4f" % reconstruction_error)
        return reconstruction_error

    reconstruction_error.mode = "min"

    @torch.no_grad()
    def marginal_ll(self, n_mc_samples=1000, ratio_loss=False):
        ll = compute_marginal_log_likelihood(
            self.model, self, n_mc_samples, ratio_loss=ratio_loss
        )
        logger.debug("True LL : %.4f" % ll)
        return ll

    @torch.no_grad()
    def get_latent(self, sample=False):
        """
        Output posterior z mean or sample, batch index, and label
        :param sample: z mean or z sample
        :return: three np.ndarrays, latent, batch_indices, labels
        """
        latent = []
        batch_indices = []
        labels = []
        for tensors in self:
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            give_mean = not sample
            latent += [
                self.model.sample_from_posterior_z(
                    sample_batch, give_mean=give_mean
                ).cpu()
            ]
            batch_indices += [batch_index.cpu()]
            labels += [label.cpu()]
        return (
            np.array(torch.cat(latent)),
            np.array(torch.cat(batch_indices)),
            np.array(torch.cat(labels)).ravel(),
        )

    @torch.no_grad()
    def entropy_batch_mixing(self, **kwargs):
        if self.gene_dataset.n_batches == 2:
            latent, batch_indices, labels = self.get_latent()
            be_score = entropy_batch_mixing(latent, batch_indices, **kwargs)
            logger.debug("Entropy batch mixing :", be_score)
            return be_score

    entropy_batch_mixing.mode = "max"

    @torch.no_grad()
    def differential_expression_stats(self, M_sampling=100):
        """
        Output average over statistics in a symmetric way (a against b)
        forget the sets if permutation is True
        :param M_sampling: number of samples
        :return: Tuple px_scales, all_labels where:
            - px_scales: scales of shape (M_sampling, n_genes)
            - all_labels: labels of shape (M_sampling, )
        """
        px_scales = []
        all_labels = []
        batch_size = max(
            self.data_loader_kwargs["batch_size"] // M_sampling, 2
        )  # Reduce batch_size on GPU
        if len(self.gene_dataset) % batch_size == 1:
            batch_size += 1
        for tensors in self.update({"batch_size": batch_size}):
            sample_batch, _, _, batch_index, labels = tensors
            px_scales += [
                np.array(
                    (
                        self.model.get_sample_scale(
                            sample_batch,
                            batch_index=batch_index,
                            y=labels,
                            n_samples=M_sampling,
                        )
                    ).cpu()
                )
            ]

            # Align the sampling
            if M_sampling > 1:
                px_scales[-1] = (px_scales[-1].transpose((1, 0, 2))).reshape(
                    -1, px_scales[-1].shape[-1]
                )
            all_labels += [np.array((labels.repeat(1, M_sampling).view(-1, 1)).cpu())]

        px_scales = np.concatenate(px_scales)
        all_labels = np.concatenate(all_labels).ravel()  # this will be used as boolean

        return px_scales, all_labels

    @torch.no_grad()
    def sample_scale_from_batch(self, n_samples, batchid=None, selection=None):
        # TODO: Implement log probas
        px_scales = []
        if selection is None:
            raise ValueError("selections should be a list of cell subsets indices")
        else:
            if selection.dtype is np.dtype("bool"):
                selection = np.asarray(np.where(selection)[0].ravel())
        old_loader = self.data_loader
        for i in batchid:
            idx = np.random.choice(
                np.arange(len(self.gene_dataset))[selection], n_samples
            )
            sampler = SubsetRandomSampler(idx)
            self.data_loader_kwargs.update({"sampler": sampler})
            self.data_loader = DataLoader(self.gene_dataset, **self.data_loader_kwargs)
            px_scales.append(self.get_harmonized_scale(i))
        self.data_loader = old_loader
        px_scales = np.concatenate(px_scales)
        return px_scales, None

    @torch.no_grad()
    def sample_poisson_from_batch(self, n_samples, batchid=None, selection=None):
        # TODO: Refactor?
        px_scales = []
        log_probas = []
        if selection is None:
            raise ValueError("selections should be a list of cell subsets indices")
        else:
            if selection.dtype is np.dtype("bool"):
                selection = np.asarray(np.where(selection)[0].ravel())
        old_loader = self.data_loader
        for i in batchid:
            idx = np.random.choice(
                np.arange(len(self.gene_dataset))[selection], n_samples
            )
            sampler = SubsetRandomSampler(idx)
            self.data_loader_kwargs.update({"sampler": sampler})
            self.data_loader = DataLoader(self.gene_dataset, **self.data_loader_kwargs)

            for tensors in self:
                sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
                (
                    px_scale,
                    px_dispersion,
                    px_rate,
                    px_dropout,
                    qz_m,
                    qz_v,
                    z,
                    ql_m,
                    ql_v,
                    library,
                ) = self.model.inference(sample_batch, batch_index, label)

                # px_rate = self.get_harmonized_scale(i)
                # p = px_rate / (px_rate + px_dispersion.cpu().numpy())
                # r = px_dispersion.cpu().numpy()

                # p = (px_scale / (px_scale + px_dispersion)).cpu().numpy()
                p = (px_rate / (px_rate + px_dispersion)).cpu().numpy()
                r = px_dispersion.cpu().numpy()

                l_train = np.random.gamma(r, p / (1 - p))
                px_scales.append(l_train)

                log_px_z = self.model._reconstruction_loss(
                    sample_batch, px_rate, px_dispersion, px_dropout
                )
                log_pz = (
                    Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
                    .log_prob(z)
                    .sum(dim=-1)
                )
                log_qz_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
                log_p = log_pz + log_px_z - log_qz_x
                log_probas.append(log_p.cpu().numpy())

        self.data_loader = old_loader
        px_scales = np.concatenate(px_scales)
        log_probas = np.concatenate(log_probas)
        return px_scales, log_probas

    @torch.no_grad()
    def sample_gamma_params_from_batch(self, n_samples, batchid=None, selection=None):
        shapes_res, scales_res = [], []
        if selection is None:
            raise ValueError("selections should be a list of cell subsets indices")
        else:
            if selection.dtype is np.dtype("bool"):
                selection = np.asarray(np.where(selection)[0].ravel())
        old_loader = self.data_loader
        for i in batchid:
            idx = np.random.choice(
                np.arange(len(self.gene_dataset))[selection], n_samples
            )
            sampler = SubsetRandomSampler(idx)
            self.data_loader_kwargs.update({"sampler": sampler})
            self.data_loader = DataLoader(self.gene_dataset, **self.data_loader_kwargs)
            #
            #     # fixed_batch = float(i)
            for tensors in self:
                sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
                px_scale, px_dispersion, px_rate = self.model.inference(
                    sample_batch, batch_index, label
                )[0:3]

                # px_rate = self.get_harmonized_scale(i)
                # p = px_rate / (px_rate + px_dispersion.cpu().numpy())
                # r = px_dispersion.cpu().numpy()

                # p = (px_scale / (px_scale + px_dispersion)).cpu().numpy()
                p = (px_rate / (px_rate + px_dispersion)).cpu().numpy()
                r = px_dispersion.cpu().numpy()

                shapes_batch = r
                scales_batch = p / (1.0 - p)

                if len(shapes_batch.shape) == 1:
                    shapes_batch = np.repeat(
                        shapes_batch.reshape((1, -1)),
                        repeats=scales_batch.shape[0],
                        axis=0,
                    )

                shapes_res.append(shapes_batch)
                scales_res.append(scales_batch)
        self.data_loader = old_loader
        shapes_res = np.concatenate(shapes_res)
        scales_res = np.concatenate(scales_res)

        assert shapes_res.shape == scales_res.shape, (
            shapes_res.shape,
            scales_res.shape,
        )
        return shapes_res, scales_res

    @torch.no_grad()
    def differential_expression_gamma(
        self,
        idx1,
        idx2,
        batchid1=None,
        batchid2=None,
        genes=None,
        n_samples=None,
        M_permutation=None,
        all_stats=True,
        sample_pairs=True,
    ):
        n_samples = 5000 if n_samples is None else n_samples
        M_permutation = 10000 if M_permutation is None else M_permutation
        if batchid1 is None:
            batchid1 = np.arange(self.gene_dataset.n_batches)
        if batchid2 is None:
            batchid2 = np.arange(self.gene_dataset.n_batches)

        shapes1, scales1 = self.sample_gamma_params_from_batch(
            selection=idx1, batchid=batchid1, n_samples=n_samples
        )
        shapes2, scales2 = self.sample_gamma_params_from_batch(
            selection=idx2, batchid=batchid2, n_samples=n_samples
        )
        print(shapes1.shape, scales1.shape, shapes2.shape, scales2.shape)
        all_labels = np.concatenate(
            (np.repeat(0, len(shapes1)), np.repeat(1, len(shapes2))), axis=0
        )

        if genes is not None:
            shapes1 = shapes1[:, self.gene_dataset._gene_idx(genes)]
            scales1 = scales1[:, self.gene_dataset._gene_idx(genes)]
            shapes2 = shapes2[:, self.gene_dataset._gene_idx(genes)]
            scales2 = scales2[:, self.gene_dataset._gene_idx(genes)]

        shapes = np.concatenate((shapes1, shapes2), axis=0)
        scales = np.concatenate((scales1, scales2), axis=0)

        assert shapes.shape == scales.shape, (shapes.shape, scales.shape)
        data = (shapes, scales)

        bayes1 = get_bayes_gamma(
            data,
            all_labels,
            cell_idx=0,
            M_permutation=M_permutation,
            permutation=False,
            sample_pairs=sample_pairs,
        )
        bayes1 = pd.Series(data=bayes1, index=self.gene_dataset.gene_names)
        return bayes1

    @torch.no_grad()
    def differential_expression_score(
        self,
        idx1: Union[List[bool], np.ndarray],
        idx2: Union[List[bool], np.ndarray],
        batchid1: Optional[Union[List[int], np.ndarray]] = None,
        batchid2: Optional[Union[List[int], np.ndarray]] = None,
        genes: Optional[Union[List[str], np.ndarray]] = None,
        n_samples: int = None,
        sample_pairs: bool = True,
        M_permutation: int = None,
        all_stats: bool = True,
        sample_gamma: bool = False,
        importance_sampling: bool = False,
    ):
        """Computes gene specific Bayes factors using masks idx1 and idx2

        To that purpose we sample the Posterior in the following way:
            1. The posterior is sampled n_samples times for each subpopulation
            2. For computation efficiency (posterior sampling is quite expensive), instead of
            comparing element-wise the obtained samples, we can permute posterior samples.
            Remember that computing the Bayes Factor requires sampling
            q(z_A | x_A) and q(z_B | x_B)

        :param idx1: bool array masking subpopulation cells 1. Should be True where cell is
        from associated population
        :param idx2: bool array masking subpopulation cells 2. Should be True where cell is
        from associated population
        :param batchid1: List of batch ids for which you want to perform DE Analysis for
        subpopulation 1. By default, all ids are taken into account
        :param batchid2: List of batch ids for which you want to perform DE Analysis for
        subpopulation 2. By default, all ids are taken into account
        :param genes: list Names of genes for which Bayes factors will be computed
        :param n_samples: Number of times the posterior will be sampled for each pop
        :param sample_pairs: Activates step 2 described above.
        Simply formulated, pairs obtained from posterior sampling (when calling
        `sample_scale_from_batch`) will be randomly permuted so that the number of
        pairs used to compute Bayes Factors becomes M_permutation.
            :param M_permutation: Number of times we will "mix" posterior samples in step 2.
            Only makes sense when sample_pairs=True
        :param all_stats: If False returns Bayes factors alone
        else, returns not only Bayes Factor of population 1 vs population 2 but other metrics as
        well, mostly used for sanity checks, such as
            - Bayes Factors of 2 vs 1
            - Bayes factors obtained when indices used to computed bayes are chosen randomly
            (ie we compute Bayes factors of Completely Random vs Completely Random).
            These can be seen as control tests.
            - Gene expression statistics (mean, scale ...)
        :return:
        """

        n_samples = 5000 if n_samples is None else n_samples
        M_permutation = 10000 if M_permutation is None else M_permutation
        if batchid1 is None:
            batchid1 = np.arange(self.gene_dataset.n_batches)
        if batchid2 is None:
            batchid2 = np.arange(self.gene_dataset.n_batches)

        if sample_gamma:
            px_scale1, log_probas1 = self.sample_poisson_from_batch(
                selection=idx1, batchid=batchid1, n_samples=n_samples
            )
            px_scale2, log_probas2 = self.sample_poisson_from_batch(
                selection=idx2, batchid=batchid2, n_samples=n_samples
            )
        else:
            px_scale1, log_probas1 = self.sample_scale_from_batch(
                selection=idx1, batchid=batchid1, n_samples=n_samples
            )
            px_scale2, log_probas2 = self.sample_scale_from_batch(
                selection=idx2, batchid=batchid2, n_samples=n_samples
            )
        px_scale_mean1 = px_scale1.mean(axis=0)
        px_scale_mean2 = px_scale2.mean(axis=0)
        px_scale = np.concatenate((px_scale1, px_scale2), axis=0)
        if log_probas1 is not None:
            log_probas = np.concatenate((log_probas1, log_probas2), axis=0)
        else:
            log_probas = None
        # print('px_scales1 shapes', px_scale1.shape)
        # print('px_scales2 shapes', px_scale2.shape)
        all_labels = np.concatenate(
            (np.repeat(0, len(px_scale1)), np.repeat(1, len(px_scale2))), axis=0
        )
        if genes is not None:
            px_scale = px_scale[:, self.gene_dataset.genes_to_index(genes)]
        bayes1 = get_bayes_factors(
            px_scale,
            all_labels,
            cell_idx=0,
            M_permutation=M_permutation,
            permutation=False,
            sample_pairs=sample_pairs,
            importance_sampling=importance_sampling,
            log_ratios=log_probas,
        )
        if all_stats is True:
            bayes1_permuted = get_bayes_factors(
                px_scale,
                all_labels,
                cell_idx=0,
                M_permutation=M_permutation,
                permutation=True,
                sample_pairs=sample_pairs,
                importance_sampling=importance_sampling,
                log_ratios=log_probas,
            )
            bayes2 = get_bayes_factors(
                px_scale,
                all_labels,
                cell_idx=1,
                M_permutation=M_permutation,
                permutation=False,
                sample_pairs=sample_pairs,
                importance_sampling=importance_sampling,
                log_ratios=log_probas,
            )
            bayes2_permuted = get_bayes_factors(
                px_scale,
                all_labels,
                cell_idx=1,
                M_permutation=M_permutation,
                permutation=True,
                sample_pairs=sample_pairs,
                importance_sampling=importance_sampling,
                log_ratios=log_probas,
            )
            (
                mean1,
                mean2,
                nonz1,
                nonz2,
                norm_mean1,
                norm_mean2,
            ) = self.gene_dataset.raw_counts_properties(idx1, idx2)
            res = pd.DataFrame(
                [
                    bayes1,
                    bayes1_permuted,
                    bayes2,
                    bayes2_permuted,
                    mean1,
                    mean2,
                    nonz1,
                    nonz2,
                    norm_mean1,
                    norm_mean2,
                    px_scale_mean1,
                    px_scale_mean2,
                ],
                index=[
                    "bayes1",
                    "bayes1_permuted",
                    "bayes2",
                    "bayes2_permuted",
                    "mean1",
                    "mean2",
                    "nonz1",
                    "nonz2",
                    "norm_mean1",
                    "norm_mean2",
                    "scale1",
                    "scale2",
                ],
                columns=self.gene_dataset.gene_names,
            ).T
            res = res.sort_values(by=["bayes1"], ascending=False)
            return res
        else:
            return pd.Series(data=bayes1, index=self.gene_dataset.gene_names)

    @torch.no_grad()
    def one_vs_all_degenes(
        self,
        subset: Optional[Union[List[bool], np.ndarray]] = None,
        cell_labels: Optional[Union[List, np.ndarray]] = None,
        min_cells: int = 10,
        n_samples: int = None,
        sample_pairs: bool = False,
        M_permutation: int = None,
        output_file: bool = False,
        save_dir: str = "./",
        filename="one2all",
    ):
        """
        Performs one population vs all others Differential Expression Analysis
        given labels or using cell types, for each type of population



        :param subset: None Or
        bool array masking subset of cells you are interested in (True when you want to select cell).
        In that case, it should have same length than `gene_dataset`
        :param cell_labels: optional: Labels of cells
        :param min_cells: Ceil number of cells used to compute Bayes Factors
        :param n_samples: Number of times the posterior will be sampled for each pop
        :param sample_pairs: Activates pair random permutations.
        Simply formulated, pairs obtained from posterior sampling (when calling
        `sample_scale_from_batch`) will be randomly permuted so that the number of
        pairs used to compute Bayes Factors becomes M_permutation.
            :param M_permutation: Number of times we will "mix" posterior samples in step 2.
                Only makes sense when sample_pairs=True
        :param output_file: Bool: save file?
            :param save_dir:
            :param filename:
        :return: Tuple (de_res, de_cluster)
            - de_res is a list of length nb_clusters (based on provided labels or on hardcoded cell
        types). de_res[i] contains Bayes Factors for population number i vs all the rest
            - de_cluster returns the associated names of clusters

            Are contains in this results only clusters for which we have at least `min_cells`
            elements to compute predicted Bayes Factors
        """
        if cell_labels is not None:
            if len(cell_labels) != len(self.gene_dataset):
                raise ValueError(
                    " the length of cell_labels have to be "
                    "the same as the number of cells"
                )

        if (cell_labels is None) and not hasattr(self.gene_dataset, "cell_types"):
            raise ValueError(
                "If gene_dataset is not annotated with labels and cell types,"
                " then must provide cell_labels"
            )
        # Input cell_labels take precedence over cell type label annotation in dataset
        elif cell_labels is not None:
            cluster_id = np.unique(cell_labels[cell_labels >= 0])
            # Can make cell_labels < 0 to filter out cells when computing DE
        else:
            cluster_id = self.gene_dataset.cell_types
            cell_labels = self.gene_dataset.labels.ravel()

        de_res = []
        de_cluster = []
        for i, x in enumerate(cluster_id):
            if subset is None:
                idx1 = cell_labels == i
                idx2 = cell_labels != i
            else:
                idx1 = (cell_labels == i) * subset
                idx2 = (cell_labels != i) * subset
            if np.sum(idx1) > min_cells and np.sum(idx2) > min_cells:
                de_cluster.append(x)
                # TODO: Understand issue when Sample_pairs=True
                res = self.differential_expression_score(
                    idx1=idx1,
                    idx2=idx2,
                    M_permutation=M_permutation,
                    n_samples=n_samples,
                    sample_pairs=sample_pairs,
                )
                res["clusters"] = np.repeat(x, len(res.index))
                de_res.append(res)
        if output_file:  # store as an excel spreadsheet
            writer = pd.ExcelWriter(
                save_dir + "differential_expression.%s.xlsx" % filename,
                engine="xlsxwriter",
            )
            for i, x in enumerate(de_cluster):
                de_res[i].to_excel(writer, sheet_name=str(x))
            writer.close()
        return de_res, de_cluster

    def within_cluster_degenes(
        self,
        cell_labels: Optional[Union[List, np.ndarray]] = None,
        min_cells: int = 10,
        states: Union[List[bool], np.ndarray] = [],
        batch1: Optional[Union[List[int], np.ndarray]] = None,
        batch2: Optional[Union[List[int], np.ndarray]] = None,
        subset: Optional[Union[List[bool], np.ndarray]] = None,
        n_samples: int = None,
        sample_pairs: bool = False,
        M_permutation: int = None,
        output_file: bool = False,
        save_dir: str = "./",
        filename: str = "within_cluster",
    ):
        """
        Performs Differential Expression within clusters for different cell states

        :param cell_labels: optional: Labels of cells
        :param min_cells: Ceil number of cells used to compute Bayes Factors
        :param states: States of the cells.
        :param batch1: List of batch ids for which you want to perform DE Analysis for
        subpopulation 1. By default, all ids are taken into account
        :param batch2: List of batch ids for which you want to perform DE Analysis for
        subpopulation 2. By default, all ids are taken into account
        :param subset: MASK: Subset of cells you are insterested in.
        :param n_samples: Number of times the posterior will be sampled for each pop
        :param sample_pairs: Activates pair random permutations.
        Simply formulated, pairs obtained from posterior sampling (when calling
        `sample_scale_from_batch`) will be randomly permuted so that the number of
        pairs used to compute Bayes Factors becomes M_permutation.
            :param M_permutation: Number of times we will "mix" posterior samples in step 2.
                Only makes sense when sample_pairs=True
        :param output_file: Bool: save file?
            :param save_dir:
            :param filename:
        :return: Tuple (de_res, de_cluster)
            - de_res is a list of length nb_clusters (based on provided labels or on hardcoded cell
        types). de_res[i] contains Bayes Factors for population number i vs all the rest
            - de_cluster returns the associated names of clusters

            Are contains in this results only clusters for which we have at least `min_cells`
            elements to compute predicted Bayes Factors
        """
        if len(self.gene_dataset) != len(states):
            raise ValueError(
                " the length of states have to be the same as the number of cells"
            )

        if (cell_labels is None) and not hasattr(self.gene_dataset, "cell_types"):
            raise ValueError(
                "If gene_dataset is not annotated with labels and cell types,"
                " then must provide cell_labels"
            )
        # Input cell_labels take precedence over cell type label annotation in dataset
        elif cell_labels is not None:
            cluster_id = np.unique(cell_labels[cell_labels >= 0])
            # Can make cell_labels < 0 to filter out cells when computing DE
        else:
            cluster_id = self.gene_dataset.cell_types
            cell_labels = self.gene_dataset.labels.ravel()
        de_res = []
        de_cluster = []
        states = np.asarray([1 if x else 0 for x in states])
        nstates = np.asarray([0 if x else 1 for x in states])

        for i, x in enumerate(cluster_id):
            if subset is None:
                idx1 = (cell_labels == i) * states
                idx2 = (cell_labels == i) * nstates
            else:
                idx1 = (cell_labels == i) * subset * states
                idx2 = (cell_labels == i) * subset * nstates
            if np.sum(idx1) > min_cells and np.sum(idx2) > min_cells:
                de_cluster.append(x)
                res = self.differential_expression_score(
                    idx1=idx1,
                    idx2=idx2,
                    batchid1=batch1,
                    batchid2=batch2,
                    M_permutation=M_permutation,
                    n_samples=n_samples,
                    sample_pairs=sample_pairs,
                )
                res["clusters"] = np.repeat(x, len(res.index))
                de_res.append(res)
        if output_file:  # store as an excel spreadsheet
            writer = pd.ExcelWriter(
                save_dir + "differential_expression.%s.xlsx" % filename,
                engine="xlsxwriter",
            )
            for i, x in enumerate(de_cluster):
                de_res[i].to_excel(writer, sheet_name=str(x))
            writer.close()
        return de_res, de_cluster

    @torch.no_grad()
    def imputation(self, n_samples=1):
        imputed_list = []
        for tensors in self:
            sample_batch, _, _, batch_index, labels = tensors
            px_rate = self.model.get_sample_rate(
                sample_batch, batch_index=batch_index, y=labels, n_samples=n_samples
            )
            imputed_list += [np.array(px_rate.cpu())]
        imputed_list = np.concatenate(imputed_list)
        return imputed_list.squeeze()

    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 100,
        genes: Union[list, np.ndarray] = None,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create observation samples from the Posterior Predictive distribution

        :param n_samples: Number of required samples for each cell
        :param genes: Indices of genes of interest
        :param batch_size: Desired Batch size to generate data

        :return: Tuple (x_new, x_old)
            Where x_old has shape (n_cells, n_genes)
            Where x_new has shape (n_cells, n_genes, n_samples)
        """
        assert self.model.reconstruction_loss in ["zinb", "nb"]
        zero_inflated = self.model.reconstruction_loss == "zinb"
        x_old = []
        x_new = []
        for tensors in self.update({"batch_size": batch_size}):
            sample_batch, _, _, batch_index, labels = tensors
            outputs = self.model.inference(
                sample_batch, batch_index=batch_index, y=labels, n_samples=n_samples
            )
            px_dispersion = outputs["px_r"]
            px_rate = outputs["px_rate"]
            px_dropout = outputs["px_dropout"]

            p = px_rate / (px_rate + px_dispersion)
            r = px_dispersion
            # Important remark: Gamma is parametrized by the rate = 1/scale!
            l_train = distributions.Gamma(concentration=r, rate=(1 - p) / p).sample()
            # Clamping as distributions objects can have buggy behaviors when
            # their parameters are too high
            l_train = torch.clamp(l_train, max=1e8)
            gene_expressions = distributions.Poisson(
                l_train
            ).sample()  # Shape : (n_samples, n_cells_batch, n_genes)
            if zero_inflated:
                p_zero = (1.0 + torch.exp(-px_dropout)).pow(-1)
                random_prob = torch.rand_like(p_zero)
                gene_expressions[random_prob <= p_zero] = 0

            gene_expressions = gene_expressions.permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)

            x_old.append(sample_batch)
            x_new.append(gene_expressions)

        x_old = torch.cat(x_old)  # Shape (n_cells, n_genes)
        x_new = torch.cat(x_new)  # Shape (n_cells, n_genes, n_samples)
        if genes is not None:
            gene_ids = self.gene_dataset.genes_to_index(genes)
            x_new = x_new[:, gene_ids, :]
            x_old = x_old[:, gene_ids]
        return x_new.cpu().numpy(), x_old.cpu().numpy()

    @torch.no_grad()
    def generate_parameters(self):
        dropout_list = []
        mean_list = []
        dispersion_list = []
        for tensors in self.sequential(1000):
            sample_batch, _, _, batch_index, labels = tensors

            outputs = self.model.inference(
                sample_batch, batch_index=batch_index, y=labels, n_samples=1
            )
            px_dispersion = outputs["px_r"]
            px_rate = outputs["px_rate"]
            px_dropout = outputs["px_dropout"]

            dispersion_list += [
                np.repeat(
                    np.array(px_dispersion.cpu())[np.newaxis, :],
                    px_rate.size(0),
                    axis=0,
                )
            ]
            mean_list += [np.array(px_rate.cpu())]
            dropout_list += [np.array(px_dropout.cpu())]

        return (
            np.concatenate(dropout_list),
            np.concatenate(mean_list),
            np.concatenate(dispersion_list),
        )

    @torch.no_grad()
    def get_stats(self):
        libraries = []
        for tensors in self.sequential(batch_size=128):
            x, local_l_mean, local_l_var, batch_index, y = tensors
            library = self.model.inference(x, batch_index, y)["library"]
            libraries += [np.array(library.cpu())]
        libraries = np.concatenate(libraries)
        return libraries.ravel()

    @torch.no_grad()
    def get_harmonized_scale(self, fixed_batch):
        px_scales = []
        fixed_batch = float(fixed_batch)
        for tensors in self:
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            px_scales += [self.model.scale_from_z(sample_batch, fixed_batch).cpu()]
        return np.concatenate(px_scales)

    @torch.no_grad()
    def get_sample_scale(self):
        px_scales = []
        for tensors in self:
            sample_batch, _, _, batch_index, labels = tensors
            px_scales += [
                np.array(
                    (
                        self.model.get_sample_scale(
                            sample_batch, batch_index=batch_index, y=labels, n_samples=1
                        )
                    ).cpu()
                )
            ]
        return np.concatenate(px_scales)

    @torch.no_grad()
    def imputation_list(self, n_samples=1):
        original_list = []
        imputed_list = []
        batch_size = 10000  # self.data_loader_kwargs["batch_size"] // n_samples
        for tensors, corrupted_tensors in zip(
            self.uncorrupted().sequential(batch_size=batch_size),
            self.corrupted().sequential(batch_size=batch_size),
        ):
            batch = tensors[0]
            actual_batch_size = batch.size(0)
            dropout_batch, _, _, batch_index, labels = corrupted_tensors
            px_rate = self.model.get_sample_rate(
                dropout_batch, batch_index=batch_index, y=labels, n_samples=n_samples
            )

            indices_dropout = torch.nonzero(batch - dropout_batch)
            if indices_dropout.size() != torch.Size([0]):
                i = indices_dropout[:, 0]
                j = indices_dropout[:, 1]

                batch = batch.unsqueeze(0).expand(
                    (n_samples, batch.size(0), batch.size(1))
                )
                original = np.array(batch[:, i, j].view(-1).cpu())
                imputed = np.array(px_rate[..., i, j].view(-1).cpu())

                cells_index = np.tile(np.array(i.cpu()), n_samples)

                original_list += [
                    original[cells_index == i] for i in range(actual_batch_size)
                ]
                imputed_list += [
                    imputed[cells_index == i] for i in range(actual_batch_size)
                ]
            else:
                original_list = np.array([])
                imputed_list = np.array([])
        return original_list, imputed_list

    @torch.no_grad()
    def imputation_score(self, original_list=None, imputed_list=None, n_samples=1):
        if original_list is None or imputed_list is None:
            original_list, imputed_list = self.imputation_list(n_samples=n_samples)

        original_list_concat = np.concatenate(original_list)
        imputed_list_concat = np.concatenate(imputed_list)
        are_lists_empty = (len(original_list_concat) == 0) and (
            len(imputed_list_concat) == 0
        )
        if are_lists_empty:
            logger.info(
                "No difference between corrupted dataset and uncorrupted dataset"
            )
            return 0.0
        else:
            return np.median(np.abs(original_list_concat - imputed_list_concat))

    @torch.no_grad()
    def imputation_benchmark(
        self, n_samples=8, show_plot=True, title_plot="imputation", save_path=""
    ):
        original_list, imputed_list = self.imputation_list(n_samples=n_samples)
        # Median of medians for all distances
        median_score = self.imputation_score(
            original_list=original_list, imputed_list=imputed_list
        )

        # Mean of medians for each cell
        imputation_cells = []
        for original, imputed in zip(original_list, imputed_list):
            has_imputation = len(original) and len(imputed)
            imputation_cells += [
                np.median(np.abs(original - imputed)) if has_imputation else 0
            ]
        mean_score = np.mean(imputation_cells)

        logger.debug(
            "\nMedian of Median: %.4f\nMean of Median for each cell: %.4f"
            % (median_score, mean_score)
        )

        plot_imputation(
            np.concatenate(original_list),
            np.concatenate(imputed_list),
            show_plot=show_plot,
            title=os.path.join(save_path, title_plot),
        )
        return original_list, imputed_list

    @torch.no_grad()
    def knn_purity(self):
        latent, _, labels = self.get_latent()
        score = knn_purity(latent, labels)
        logger.debug("KNN purity score :", score)
        return score

    knn_purity.mode = "max"

    @torch.no_grad()
    def clustering_scores(self, prediction_algorithm="knn"):
        if self.gene_dataset.n_labels > 1:
            latent, _, labels = self.get_latent()
            if prediction_algorithm == "knn":
                labels_pred = KMeans(
                    self.gene_dataset.n_labels, n_init=200
                ).fit_predict(
                    latent
                )  # n_jobs>1 ?
            elif prediction_algorithm == "gmm":
                gmm = GMM(self.gene_dataset.n_labels)
                gmm.fit(latent)
                labels_pred = gmm.predict(latent)

            asw_score = silhouette_score(latent, labels)
            nmi_score = NMI(labels, labels_pred)
            ari_score = ARI(labels, labels_pred)
            uca_score = unsupervised_clustering_accuracy(labels, labels_pred)[0]
            logger.debug(
                "Clustering Scores:\nSilhouette: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f"
                % (asw_score, nmi_score, ari_score, uca_score)
            )
            return asw_score, nmi_score, ari_score, uca_score

    @torch.no_grad()
    def nn_overlap_score(self, **kwargs):
        """
        Quantify how much the similarity between cells in the mRNA latent space resembles their similarity at the
        protein level. Compute the overlap fold enrichment between the protein and mRNA-based cell 100-nearest neighbor
        graph and the Spearman correlation of the adjacency matrices.
        """
        if hasattr(self.gene_dataset, "protein_expression_clr"):
            latent, _, _ = self.sequential().get_latent()
            protein_data = self.gene_dataset.protein_expression_clr[self.indices]
            spearman_correlation, fold_enrichment = nn_overlap(
                latent, protein_data, **kwargs
            )
            logger.debug(
                "Overlap Scores:\nSpearman Correlation: %.4f\nFold Enrichment: %.4f"
                % (spearman_correlation, fold_enrichment)
            )
            return spearman_correlation, fold_enrichment

    @torch.no_grad()
    def show_t_sne(
        self,
        n_samples=1000,
        color_by="",
        save_name="",
        latent=None,
        batch_indices=None,
        labels=None,
        n_batch=None,
    ):
        # If no latent representation is given
        if latent is None:
            latent, batch_indices, labels = self.get_latent(sample=True)
            latent, idx_t_sne = self.apply_t_sne(latent, n_samples)
            batch_indices = batch_indices[idx_t_sne].ravel()
            labels = labels[idx_t_sne].ravel()
        if not color_by:
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1])
        if color_by == "scalar":
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1], c=labels.ravel())
        else:
            if n_batch is None:
                n_batch = self.gene_dataset.n_batches
            if color_by == "batches" or color_by == "labels":
                indices = (
                    batch_indices.ravel() if color_by == "batches" else labels.ravel()
                )
                n = n_batch if color_by == "batches" else self.gene_dataset.n_labels
                if self.gene_dataset.cell_types is not None and color_by == "labels":
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(len(np.unique(indices)))]
                plt.figure(figsize=(10, 10))
                for i, label in zip(range(n), plt_labels):
                    plt.scatter(
                        latent[indices == i, 0], latent[indices == i, 1], label=label
                    )
                plt.legend()
            elif color_by == "batches and labels":
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                batch_indices = batch_indices.ravel()
                for i in range(n_batch):
                    axes[0].scatter(
                        latent[batch_indices == i, 0],
                        latent[batch_indices == i, 1],
                        label=str(i),
                    )
                axes[0].set_title("batch coloring")
                axes[0].axis("off")
                axes[0].legend()

                indices = labels.ravel()
                if hasattr(self.gene_dataset, "cell_types"):
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(len(np.unique(indices)))]
                for i, cell_type in zip(range(self.gene_dataset.n_labels), plt_labels):
                    axes[1].scatter(
                        latent[indices == i, 0],
                        latent[indices == i, 1],
                        label=cell_type,
                    )
                axes[1].set_title("label coloring")
                axes[1].axis("off")
                axes[1].legend()
        plt.axis("off")
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)

    @staticmethod
    def apply_t_sne(latent, n_samples=1000):
        idx_t_sne = (
            np.random.permutation(len(latent))[:n_samples]
            if n_samples
            else np.arange(len(latent))
        )
        if latent.shape[1] != 2:
            latent = TSNE().fit_transform(latent[idx_t_sne])
        return latent, idx_t_sne

    def raw_data(self):
        """
        Returns raw data for classification
        """
        return (
            self.gene_dataset.X[self.indices],
            self.gene_dataset.labels[self.indices].ravel(),
        )


def entropy_from_indices(indices):
    return entropy(np.array(np.unique(indices, return_counts=True)[1].astype(np.int32)))


def entropy_batch_mixing(
    latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100
):
    def entropy(hist_data):
        n_batches = len(np.unique(hist_data))
        if n_batches > 2:
            raise ValueError("Should be only two clusters for this metric")
        frequency = np.mean(hist_data == 1)
        if frequency == 0 or frequency == 1:
            return 0
        return -frequency * np.log(frequency) - (1 - frequency) * np.log(1 - frequency)

    n_neighbors = min(n_neighbors, len(latent_space) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(latent_space) - scipy.sparse.identity(
        latent_space.shape[0]
    )

    score = 0
    for t in range(n_pools):
        indices = np.random.choice(
            np.arange(latent_space.shape[0]), size=n_samples_per_pool
        )
        score += np.mean(
            [
                entropy(
                    batches[
                        kmatrix[indices].nonzero()[1][
                            kmatrix[indices].nonzero()[0] == i
                        ]
                    ]
                )
                for i in range(n_samples_per_pool)
            ]
        )
    return score / float(n_pools)


def get_sampling_pair_idx(
    list_1,
    list_2,
    do_sample=True,
    permutation=False,
    M_permutation=10000,
    probas_a=None,
    probas_b=None,
):
    """
    Returns the indices of the sampled quantities of populations 1 and 2
    that will be compared.

    This function has several modes based on the values of do_sample and permutation
    as described below


    :param list_1: Indices corresponding to population 1
    :param list_2: Indices corresponding to population 2
    :param do_sample: Are pairs sampled? If not, we compare the posterior quantities
    TERMISE
    :param permutation: has only effect when do_sample is True.
        - if permutation=False, NORMAL BEHAVIOR : elements used for pop 2 come from list_1
        and vice-versa for 2
        - if permutation=True, SPECIFIC BEHAVIOR: All elements are sampled.
        Should only be used as a sanity check.

    :param M_permutation:
    :param probas_a: used for Importance Sampling, set to None by default
    :param probas_b: used for Importance Sampling, set to None by default
    :return:
    """
    if do_sample:
        if not permutation:
            # case1: no permutation, sample from A and then from B
            u, v = (
                np.random.choice(list_1, size=M_permutation, p=probas_a),
                np.random.choice(list_2, size=M_permutation, p=probas_b),
            )
        else:
            # case2: permutation, sample from A+B twice
            u, v = (
                np.random.choice(list_1 + list_2, size=M_permutation),
                np.random.choice(list_1 + list_2, size=M_permutation),
            )
    else:
        # TODO: Assert good behavior
        u, v = list_1, list_2
    assert len(u) == len(v), "Inconsistent number of indices used for pairs"
    return u, v


def get_bayes_factors(
    px_scale: Union[List[float], np.ndarray],
    all_labels: Union[List, np.ndarray],
    cell_idx: Union[int, str],
    other_cell_idx: Optional[Union[int, str]] = None,
    genes_idx: Union[List[int], np.ndarray] = None,
    log_ratios: Union[List[int], np.ndarray] = None,
    importance_sampling: bool = False,
    M_permutation: int = 10000,
    permutation: bool = False,
    sample_pairs: bool = True,
):
    """
    Returns an array of bayes factor for all genes
    :param px_scale: The gene frequency array for all cells (might contain multiple samples per cells)
    :param all_labels: The labels array for the corresponding cell types
    :param cell_idx: The first cell type population to consider. Either a string or an idx
    :param other_cell_idx: (optional) The second cell type population to consider. Either a string or an idx
    :param genes_idx: Indices of genes for which DE Analysis applies
    :param sample_pairs: Activates subsampling.
        Simply formulated, pairs obtained from posterior sampling (when calling
        `sample_scale_from_batch`) will be randomly permuted so that the number of
        pairs used to compute Bayes Factors becomes M_permutation.
    :param log_ratios: un-normalized weights for importance sampling
    :param importance_sampling: whether to use IS
    :param M_permutation: Number of times we will "mix" posterior samples in step 2.
        Only makes sense when sample_pairs=True
    :param permutation: Whether or not to permute. Normal behavior is False.
        Setting permutation=True basically shuffles cell_idx and other_cell_idx so that we
        estimate Bayes Factors of random populations of the union of cell_idx and other_cell_idx.
    :return:
    """

    idx = all_labels == cell_idx
    idx_other = (
        (all_labels == other_cell_idx)
        if other_cell_idx is not None
        else (all_labels != cell_idx)
    )
    if genes_idx is not None:
        px_scale = px_scale[:, genes_idx]

    # first extract the data
    # Assert that at this point we no longer have batch dimensions
    assert len(px_scale.shape) == 2
    sample_rate_a = px_scale[idx, :]
    sample_rate_b = px_scale[idx_other, :]
    # sample_rate_a = px_scale[:, idx, :]
    # sample_rate_b = px_scale[:, idx_other, :]
    sample_rate_a = sample_rate_a.reshape((-1, px_scale.shape[-1]))
    sample_rate_b = sample_rate_b.reshape((-1, px_scale.shape[-1]))
    samples = np.concatenate((sample_rate_a, sample_rate_b), axis=0)
    # prepare the pairs for sampling
    list_1 = list(np.arange(sample_rate_a.shape[0]))
    list_2 = list(sample_rate_a.shape[0] + np.arange(sample_rate_b.shape[0]))

    if importance_sampling:
        print("Importance Sampling")
        # weight_a = log_ratios[:, idx]
        # weight_b = log_ratios[:, idx_other]
        print(log_ratios.shape)
        weight_a = log_ratios[idx]
        weight_b = log_ratios[idx_other]

        # second let's normalize the weights
        weight_a = softmax(weight_a)
        weight_b = softmax(weight_b)
        # reshape and aggregate dataset
        weight_a = weight_a.flatten()
        weight_b = weight_b.flatten()
        weights = np.concatenate((weight_a, weight_b))

        # probas_a = weight_a / np.sum(idx)
        # probas_b = weight_b / np.sum(idx_other)
        probas_a = weight_a
        probas_b = weight_b

        print("IS A MAX", probas_a.max(), "IS B MAX", probas_b.max())
        u, v = get_sampling_pair_idx(
            list_1,
            list_2,
            do_sample=sample_pairs,
            permutation=permutation,
            M_permutation=M_permutation,
            probas_a=probas_a,
            probas_b=probas_b,
        )

        # then constitutes the pairs
        first_samples = samples[u]
        second_samples = samples[v]
        first_weights = weights[u]
        second_weights = weights[v]

        # print('u v shapes', u.shape, v.shape)

        to_sum = (
            first_weights[:, np.newaxis]
            * second_weights[:, np.newaxis]
            * (first_samples >= second_samples)
        )
        incomplete_weights = first_weights * second_weights
        res = np.sum(to_sum, axis=0) / np.sum(incomplete_weights, axis=0)
    else:
        probas_a = None
        probas_b = None

        u, v = get_sampling_pair_idx(
            list_1,
            list_2,
            do_sample=sample_pairs,
            permutation=permutation,
            M_permutation=M_permutation,
            probas_a=probas_a,
            probas_b=probas_b,
        )

        # then constitutes the pairs
        first_samples = samples[u]
        second_samples = samples[v]
        res = np.mean(first_samples >= second_samples, axis=0)
    res = np.log(res + 1e-8) - np.log(1 - res + 1e-8)
    return res


def _p_wa_higher_wb(k1, k2, theta1, theta2):
    """

    :param k1: Shape of wa
    :param k2: Shape of wb
    :param theta1: Scale of wa
    :param theta2: Scale of wb
    :return:
    """
    a = k2
    b = k1
    x = theta1 / (theta1 + theta2)
    return betainc(a, b, x)


def get_bayes_gamma(
    data,
    all_labels,
    cell_idx,
    other_cell_idx=None,
    genes_idx=None,
    M_permutation=10000,
    permutation=False,
    sample_pairs=True,
):
    """
    Returns a list of bayes factor for all genes
    :param px_scale: The gene frequency array for all cells (might contain multiple samples per cells)
    :param all_labels: The labels array for the corresponding cell types
    :param cell_idx: The first cell type population to consider. Either a string or an idx
    :param other_cell_idx: (optional) The second cell type population to consider. Either a string or an idx
    :param M_permutation: The number of permuted samples.
    :param permutation: Whether or not to permute.
    :return:
    """
    res = []
    idx = all_labels == cell_idx
    idx_other = (
        (all_labels == other_cell_idx)
        if other_cell_idx is not None
        else (all_labels != cell_idx)
    )

    shapes, scales = data

    if genes_idx is not None:
        shapes = shapes[:, genes_idx]
        scales = scales[:, genes_idx]

    sample_shape_a = shapes[idx].squeeze()
    sample_scales_a = scales[idx_other].squeeze()
    sample_shape_b = shapes[idx].squeeze()
    sample_scales_b = scales[idx_other].squeeze()

    assert sample_shape_a.shape == sample_scales_a.shape
    assert sample_shape_b.shape == sample_scales_b.shape

    # agregate dataset
    samples_shape = np.vstack((sample_shape_a, sample_shape_b))
    samples_scales = np.vstack((sample_scales_a, sample_scales_b))

    # prepare the pairs for sampling
    list_1 = list(np.arange(sample_shape_a.shape[0]))
    list_2 = list(sample_shape_a.shape[0] + np.arange(sample_shape_b.shape[0]))

    u, v = get_sampling_pair_idx(
        list_1,
        list_2,
        permutation=permutation,
        M_permutation=M_permutation,
        probas_a=None,
        probas_b=None,
        do_sample=sample_pairs,
    )

    # then constitutes the pairs
    first_set = (samples_shape[u], samples_scales[u])
    second_set = (samples_shape[v], samples_scales[v])

    shapes_a, scales_a = first_set
    shapes_b, scales_b = second_set
    for shape_a, scale_a, shape_b, scale_b in zip(
        shapes_a, scales_a, shapes_b, scales_b
    ):
        res.append(_p_wa_higher_wb(shape_a, shape_b, scale_a, scale_b))
    res = np.array(res)
    res = np.mean(res, axis=0)
    assert len(res) == shapes_a.shape[1]
    res = np.log(res + 1e-8) - np.log(1 - res + 1e-8)
    return res


def plot_imputation(original, imputed, show_plot=True, title="Imputation"):
    y = imputed
    x = original

    ymax = 10
    mask = x < ymax
    x = x[mask]
    y = y[mask]

    mask = y < ymax
    x = x[mask]
    y = y[mask]

    l_minimum = np.minimum(x.shape[0], y.shape[0])

    x = x[:l_minimum]
    y = y[:l_minimum]

    data = np.vstack([x, y])

    plt.figure(figsize=(5, 5))

    axes = plt.gca()
    axes.set_xlim([0, ymax])
    axes.set_ylim([0, ymax])

    nbins = 50

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    xi, yi = np.mgrid[0 : ymax : nbins * 1j, 0 : ymax : nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.title(title, fontsize=12)
    plt.ylabel("Imputed counts")
    plt.xlabel("Original counts")

    plt.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds")

    a, _, _, _ = np.linalg.lstsq(y[:, np.newaxis], x, rcond=-1)
    linspace = np.linspace(0, ymax)
    plt.plot(linspace, a * linspace, color="black")

    plt.plot(linspace, linspace, color="black", linestyle=":")
    if show_plot:
        plt.show()
    plt.savefig(title + ".png")


def nn_overlap(X1, X2, k=100):
    """
    Compute the overlap between the k-nearest neighbor graph of X1 and X2 using Spearman correlation of the
    adjacency matrices.
    """
    assert len(X1) == len(X2)
    n_samples = len(X1)
    k = min(k, n_samples - 1)
    nne = NearestNeighbors(n_neighbors=k + 1)  # "n_jobs=8
    nne.fit(X1)
    kmatrix_1 = nne.kneighbors_graph(X1) - scipy.sparse.identity(n_samples)
    nne.fit(X2)
    kmatrix_2 = nne.kneighbors_graph(X2) - scipy.sparse.identity(n_samples)

    # 1 - spearman correlation from knn graphs
    spearman_correlation = scipy.stats.spearmanr(
        kmatrix_1.A.flatten(), kmatrix_2.A.flatten()
    )[0]
    # 2 - fold enrichment
    set_1 = set(np.where(kmatrix_1.A.flatten() == 1)[0])
    set_2 = set(np.where(kmatrix_2.A.flatten() == 1)[0])
    fold_enrichment = (
        len(set_1.intersection(set_2))
        * n_samples ** 2
        / (float(len(set_1)) * len(set_2))
    )
    return spearman_correlation, fold_enrichment


def unsupervised_clustering_accuracy(y, y_pred):
    """
    Unsupervised Clustering Accuracy
    """
    assert len(y_pred) == len(y)
    # u = np.unique(np.concatenate((y, y_pred)))
    # n_clusters = len(u)
    # mapping = dict(zip(u, range(n_clusters)))
    # reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    # for y_pred_, y_ in zip(y_pred, y):
    #     if y_ in mapping:
    #         reward_matrix[mapping[y_pred_], mapping[y_]] += 1
    # cost_matrix = reward_matrix.max() - reward_matrix
    # ind = linear_assignment(cost_matrix)
    # return sum([reward_matrix[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind


def knn_purity(latent, label, n_neighbors=30):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = nbrs.kneighbors(latent, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: label[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - label.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[label == i]) for i in np.unique(label)
    ]  # per cell-type purity

    return np.mean(res)


def proximity_imputation(real_latent1, normed_gene_exp_1, real_latent2, k=4):
    knn = KNeighborsRegressor(k, weights="distance")
    y = knn.fit(real_latent1, normed_gene_exp_1).predict(real_latent2)
    return y
