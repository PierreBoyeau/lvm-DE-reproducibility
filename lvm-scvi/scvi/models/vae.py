# -*- coding: utf-8 -*-
"""Main module."""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Poisson, Gamma, Normal, LogNormal, kl_divergence as kl

from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive, _log_zigamma
from scvi.models.modules import (
    Encoder,
    DecoderSCVI,
    LinearDecoderSCVI,
)
from scvi.models.utils import one_hot

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True


# VAE model
class VAE(nn.Module):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        decoder_dropout_rate: float = 0.0,
        dispersion: str = "gene",
        use_batch_norm: bool = False,
        use_weight_norm: bool = False,
        use_layer_norm: bool = False,
        use_layer_normb: bool = False,
        iaf_t: int = 0,
        scale_normalize: str = "softmax",
        log_variational: bool = True,
        simplex_input: bool = False,
        reconstruction_loss: str = "zinb",
        decoder_skip=False,
        gene_factors=False,
        disp_shrink=False,
        autoregresssive: bool = False,
        log_p_z=None,
        n_blocks=0,
        decoder_do_last_skip=False,
        prevent_library_saturation: bool = False,
        with_activation=nn.ReLU(),
        px_r_init: torch.Tensor = None,
        constant_lstd: bool = False,
        sum_feature: bool = False,
        local_lmean_overwrite=None,
        lognormal_library: bool = False,
        n_total=None,
        phi_conc_prior=1.0,
        phi_rate_prior=5.0,
        loss_eps=1e-8,
        l1_reg: float = 0.0,
        base_temp=False,
        res_lib=False,
    ):
        super().__init__()
        self.gene_factors = gene_factors
        self.disp_skrink = disp_shrink
        self.eps = loss_eps
        self.local_lmean_overwrite = local_lmean_overwrite
        using_selu = isinstance(with_activation, nn.SELU)
        do_alpha_dropout = using_selu

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            with_activation=with_activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_layer_normb=use_layer_normb,
            use_weight_norm=use_weight_norm,
            # do_alpha_dropout=do_alpha_dropout,
        )

        self.n_total = n_total
        self.n_input = n_input
        self.lognormal_library = lognormal_library
        logging.info("Normal parameterization of the library")
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_layer_normb=use_layer_normb,
            use_weight_norm=use_weight_norm,
            dropout_rate=dropout_rate,
            with_activation=with_activation,
            sum_feature=sum_feature,
        )
        self.decoder = None
        self.log_variational = log_variational
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            scale_normalize=scale_normalize,
            n_layers=1,
            n_hidden=128,
            use_layer_norm=use_layer_norm,
            use_layer_normb=use_layer_normb,
            use_batch_norm=use_batch_norm,
            use_weight_norm=use_weight_norm,
            with_activation=with_activation,
            decoder_skip=decoder_skip,
            n_blocks=n_blocks,
            dropout_rate=decoder_dropout_rate,
            do_last_skip=decoder_do_last_skip,
            do_alpha_dropout=do_alpha_dropout,
        )
        self.dispersion = dispersion
        self.n_latent = n_latent

        self.simplex = simplex_input
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels

        self.px_r = torch.nn.Parameter(
            torch.randn(
                n_input,
            )
        )

        self.softplus = nn.Softplus()

        self.res_lib = res_lib
        if self.res_lib:
            logging.info("Using observed library")
            # self.local_lmean_overwrite = 0.0


    @property
    def phi_conc(self):
        return self.softplus(self._phi_conc)

    @property
    def phi_rate(self):
        return self.softplus(self._phi_rate)

    def input_transformation(self, x):
        if self.log_variational:
            return torch.log(1.0 + x)
        elif self.simplex:
            return x / (1e-16 + x.sum(-1))
        return x

    def log_p_z(self, z: torch.Tensor):
        if self.log_p_z_fixed is not None:
            return self.log_p_z_fixed(z)
        else:
            z_prior_m, z_prior_v = self.get_prior_params(device=z.device)
            return Normal(z_prior_m, z_prior_v.sqrt()).log_prob(z).sum(-1)
            # return self.z_encoder.distrib(z_prior_m, z_prior_v).log_prob(z)

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[
            "px_scale"
        ]

    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[
            "px_rate"
        ]

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(
                x, px_rate, px_r, px_dropout, eps=self.eps
            )
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r, eps=self.eps)
        elif self.reconstruction_loss == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, batch_index=None, y=None, reparam=True, n_samples=1, train_library=True, multiplicative_std=None,
    ):
        x_ = x.clone()
        x_ = self.input_transformation(x_)
        # Sampling
        library_post = self.l_encoder(x_, n_samples=n_samples, reparam=reparam)
        log_ql_x_detach = Normal(library_post["q_m"].detach(), library_post["q_v"].sqrt().detach()).log_prob(library_post["latent"]).sum(-1)
        is_lognormal_library = library_post.get("lognormal", False)

        library_variables = dict(
            ql_m=library_post["q_m"],
            ql_v=library_post["q_v"],
            library=library_post["latent"],
            log_ql_x=library_post["posterior_density"],
            log_ql_x_detach=log_ql_x_detach,
            is_lognormal_library=is_lognormal_library,
        )

        z_post = self.z_encoder(x_, y, n_samples=n_samples, reparam=reparam, multiplicative_std=multiplicative_std)
        log_qz_x_detach = Normal(z_post["q_m"].detach(), z_post["q_v"].sqrt().detach()).log_prob(z_post["latent"]).sum(-1)
        z_variables = dict(
            qz_m=z_post["q_m"],
            qz_v=z_post["q_v"],
            z=z_post["latent"],
            log_qz_x=z_post["posterior_density"],
            log_qz_x_detach=log_qz_x_detach,
        )

        z = z_variables["z"]
        # if not train_library:
        #     library = x.sum(1, keepdim=True).log()
        # else:
        library = library_variables["library"]

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            z,
            library,
            batch_index,
            y,
            is_lognormal_library=is_lognormal_library,
        )

        if self.gene_factors:
            lib = library.exp()
            px_scale_ = self.gene_biases.exp() * px_scale
            px_rate = lib * px_scale_
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
            px_r = torch.exp(px_r)
            log_qphi = None

        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
            px_r = torch.exp(px_r)
            log_qphi = None

        elif self.dispersion == "gene":
            px_r = self.px_r
            px_r = torch.exp(px_r)
            log_qphi = None

        else:
            raise ValueError(self.dispersion)

        if self.res_lib:
            observed_lib = x.sum(-1, keepdims=True)
            px_rate = (observed_lib.log() + px_scale.log()).exp()

        decoder_variables = dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

        return {
            "log_qphi": log_qphi,
            **library_variables,
            **z_variables,
            **decoder_variables,
        }

    def get_scale_(self, z, library, batch_index):
        px_scale_, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale_

    def _inference_is(
        self,
        x,
        z_candidate,
        batch_index=None,
        y=None,
        reparam=True,
        l_candidate=None,
        local_l_mean=None,
        local_l_var=None,
        n_samples=1,
        multiplicative_std=None,
    ):
        x_ = x.clone()
        x_ = self.input_transformation(x_)
        # Sampling
        library_post = self.l_encoder(x_, n_samples=n_samples, reparam=reparam)

        ql_m = library_post["q_m"]
        ql_v = library_post["q_v"]
        # Using the mean library size
        if l_candidate is None:
            l_map = library_post["q_m"]  # log scaled
        else:
            l_map = l_candidate
        z_post = self.z_encoder(x_, y, n_samples=n_samples, reparam=reparam, multiplicative_std=multiplicative_std)
        qz_m = z_post["q_m"]
        qz_v = z_post["q_v"]

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            z_candidate,
            l_map,
            batch_index,
            y,
            is_lognormal_library=False,
        )
        if self.gene_factors:
            lib = l_map.exp()
            px_scale_ = self.gene_biases.exp() * px_scale
            px_rate = lib * px_scale_

        px_r = self.px_r
        px_r = torch.exp(px_r)

        log_px_zl = -self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
        log_pxg_zl = log_nb_positive(
            x, mu=px_rate, theta=px_r, eps=self.eps, return_sum=False
        )

        z_prior_m, z_prior_v = self.get_prior_params(device=x.device)
        log_pz = Normal(z_prior_m, z_prior_v.sqrt()).log_prob(z_candidate).sum(dim=-1)
        log_qz_x = Normal(qz_m, qz_v.sqrt()).log_prob(z_candidate).sum(-1)
        log_qz_x_detach = Normal(qz_m.detach(), qz_v.sqrt().detach()).log_prob(z_candidate).sum(-1)
        if l_candidate is None:
            log_ql_x = None
            log_pl = None
        else:
            log_pl = (
                Normal(local_l_mean, torch.sqrt(local_l_var))
                .log_prob(l_map)
                .sum(dim=-1)
            )
            log_ql_x = Normal(ql_m, ql_v.sqrt()).log_prob(l_map).sum(-1)
        return dict(
            ql_m=ql_m,
            ql_v=ql_v,
            qz_m=qz_m,
            qz_v=qz_v,
            log_qz_x=log_qz_x,
            log_qz_x_detach=log_qz_x_detach,
            log_ql_x=log_ql_x,
            log_pl=log_pl,
            log_pz=log_pz,
            log_px_zl=log_px_zl,
            px_scale=px_scale,
            log_pxg_zl=log_pxg_zl,
        )

    def forward(
        self,
        x,
        local_l_mean,
        local_l_var,
        batch_index=None,
        y=None,
        loss=None,
        n_samples=1,
        train_library=True,
        beta=None,
        multiplicative_std=None,
    ):
        r"""Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution
        outputs = self.inference(
            x,
            batch_index,
            y,
            n_samples=n_samples,
            train_library=train_library,
            reparam=True,
            multiplicative_std=multiplicative_std,
        )
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]
        library = outputs["library"]
        z = outputs["z"]
        px_scale = outputs["px_scale"]

        z_prior_m, z_prior_v = self.get_prior_params(device=x.device)
        log_px_zl = -self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        if self.local_lmean_overwrite is not None:
            local_l_mean = self.local_lmean_overwrite
        log_pl = (
            Normal(local_l_mean, torch.sqrt(local_l_var))
            .log_prob(library)
            .sum(dim=-1)
        )

        log_pz = Normal(z_prior_m, z_prior_v.sqrt()).log_prob(z).sum(dim=-1)
        log_qz_x = outputs["log_qz_x"]

        # log_ql_x = Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)
        log_ql_x = outputs["log_ql_x"]
        # if train_library:
        assert (
            log_px_zl.shape
            == log_pl.shape
            == log_pz.shape
            == log_qz_x.shape
            == log_ql_x.shape
        )
        log_ratio = (log_px_zl + log_pz + log_pl) - (log_qz_x + log_ql_x)
        if loss is None:
            return {
                "log_ratio": log_ratio,
                "log_px_zl": log_px_zl,
                **outputs,
            }
        elif loss == "ELBO":
            if beta is None:
                obj = -log_ratio.mean(0)
            else:
                log_px_z = log_px_zl
                loss_reconstruction = -log_px_z.mean()
                loss_kl = -(log_ratio - log_px_z).mean()
                obj = loss_reconstruction + (beta * loss_kl)
        elif loss == "IWELBO":
            assert beta is None
            obj = -(torch.softmax(log_ratio, dim=0).detach() * log_ratio).sum(dim=0)

            if self.disp_skrink:
                obj += torch.var(self.px_r.exp())
        elif loss == "DREG":
            log_ql_x_detach = outputs["log_ql_x_detach"]
            log_qz_x_detach = outputs["log_qz_x_detach"]
            lw = (log_px_zl + log_pz + log_pl) - (log_ql_x_detach + log_qz_x_detach)
            with torch.no_grad():
                reweight = torch.exp(lw - torch.logsumexp(lw, 0))
                z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)
            return -(reweight * lw).sum(0).mean(0)
        elif loss == "logratios":
            obj = log_ratio
        elif loss == "probs":
            return {
                "log_px_zl": log_px_zl,
                "log_pl": log_pl,
                "log_pz": log_pz,
                "log_qz_x": log_qz_x,
                "log_ql_x": log_ql_x,
                "log_ratio": log_ratio,
            }

        return obj

    @property
    def encoder_params(self):
        """
        :return: List of learnable encoder parameters (to feed to torch.optim object
        for instance
        """
        return self.get_list_params(
            self.z_encoder.parameters(), self.l_encoder.parameters()
        )

    @property
    def decoder_params(self):
        """
        :return: List of learnable decoder parameters (to feed to torch.optim object
        for instance
        """
        return self.get_list_params(self.decoder.parameters()) + [self.px_r]

    def get_latents(self, x, y=None):
        r"""returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        r"""samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        x = self.input_transformation(x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def sample_from_posterior_l(self, x):
        r"""samples the tensor of library sizes from the posterior
        #doesn't really sample, returns the tensor of the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: tensor of shape ``(batch_size, 1)``
        :rtype: :py:class:`torch.Tensor`
        """
        x = self.input_transformation(x)
        l_dist = self.l_encoder(x)
        library = l_dist["latent"]
        return library

    def get_prior_params(self, device):
        mean = torch.zeros((self.n_latent,), device=device)
        # if self.z_full_cov or self.z_autoregressive:
        #     scale = self.prior_scale * torch.eye(self.n_latent, device=device)
        # else:
        scale = torch.ones((self.n_latent,), device=device)
        return mean, scale

    @staticmethod
    def get_list_params(*params):
        res = []
        for param_li in params:
            res += list(filter(lambda p: p.requires_grad, param_li))
        return res


class LDVAE(VAE):
    r"""Linear-decoded Variational auto-encoder model.

    This model uses a linear decoder, directly mapping the latent representation
    to gene expression levels. It still uses a deep neural network to encode
    the latent representation.

    Compared to standard VAE, this model is less powerful, but can be used to
    inspect which genes contribute to variation in the dataset.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer (for encoder)
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            reconstruction_loss=reconstruction_loss,
        )

        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def get_loadings(self):
        """Extract per-gene weights (for each Z) in the linear decoder."""
        return self.decoder.factor_regressor.parameters()
