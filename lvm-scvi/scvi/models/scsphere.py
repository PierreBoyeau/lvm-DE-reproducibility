# -*- coding: utf-8 -*-
"""Main module."""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as db
from scipy.special import gamma
import math
import torch

from power_spherical import PowerSpherical

from scvi.models.log_likelihood import log_nb_positive
from scvi.models.modules import Encoder

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True


class SphereEncoder(nn.Module):
    def __init__(self, n_input, z_dim, dropout_rate, use_layer_norm=True, use_batch_norm=False, deep_architecture=True) -> None:
        super().__init__()
        norm_type = None
        if use_layer_norm:
            norm_type = nn.LayerNorm
        elif use_batch_norm:
            norm_type = nn.BatchNorm1d
        else:
            raise ValueError

        if deep_architecture:
            logging.info("Using Deep architecture ...")
            self.encoder_n = nn.Sequential(
                nn.Linear(n_input, 128),
                norm_type(128),
                nn.ELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                norm_type(64),
                nn.ELU(),
                nn.Linear(64, 32),
                norm_type(32),
                nn.ELU(),
            )
            self.mean_encoder = nn.Linear(32, z_dim)
            self.var_encoder = nn.Linear(32, 1)
        else:
            logging.info("Using regular architecture ...")
            self.encoder_n = nn.Sequential(
                nn.Linear(n_input, 128),
                norm_type(128),
                nn.ELU(),
                nn.Dropout(dropout_rate),
            )
            self.mean_encoder = nn.Linear(128, z_dim)
            self.var_encoder = nn.Linear(128, 1)

    def forward(self, input, n_samples=1, reparam=True):
        x_ = input
        h = self.encoder_n(x_)
        z_mu = self.mean_encoder(h)
        z_mu = z_mu / torch.norm(z_mu, p=2, dim=-1, keepdim=True)

        z_std = self.var_encoder(h)
        #         z_std = nn.Softplus()(z_std) + 1e-2
        z_std = nn.Softplus()(z_std)
        # z_std = torch.clamp(z_std, max=1e8)
        #         z_dist = VonMisesFisher(z_mu, z_std)
        #         if reparam:
        #             z = z_dist.rsample(n_samples)
        #         else:
        #             z = z_dist.sample(n_samples)
        z_dist = PowerSpherical(z_mu, z_std.squeeze(-1))
        if reparam:
            z = z_dist.rsample((n_samples,))
        else:
            z = z_dist.sample((n_samples,))
        return dict(qz_m=z_mu, qz_v=z_std, z=z)


class SphereDecoder(nn.Module):
    def __init__(self, n_input, n_output, use_layer_norm=True, use_batch_norm=False, scale_norm="softmax", deep_architecture=True):
        super().__init__()
        norm_type = None
        if use_layer_norm:
            norm_type = nn.LayerNorm
        elif use_batch_norm:
            norm_type = nn.BatchNorm1d
        else:
            raise ValueError

        if scale_norm == "softmax":
            scale_norm_u = nn.Softmax(-1)
        elif scale_norm == "relu":
            scale_norm_u = nn.ReLU()
        elif scale_norm == "softplus":
            scale_norm_u = nn.Softplus()
        else:
            raise ValueError
        
        if deep_architecture:
            logging.info("Using deep architecture")
            self.decoder_n = nn.Sequential(
                nn.Linear(n_input, 32),
                norm_type(32),
                nn.ELU(),
                nn.Linear(32, 128),
                norm_type(128),
                nn.ELU(),
                nn.Linear(128, 128),
                norm_type(128),
                nn.ELU(),
            )
        else:
            logging.info("Using regular architecture")
            self.decoder_n = nn.Sequential(
                nn.Linear(n_input, 128),
                norm_type(128),
                nn.ELU(),
            )
        self.decoder_scale = nn.Sequential(nn.Linear(128, n_output), scale_norm_u)
        self.decoder_std = nn.Linear(128, n_output)

    def forward(self, input):
        # This step requires a specific handling because the batchnorm needs to be applied elementwise
        # for layers in self.decoder_n:
        # for layer in self.decoder_n:
        #     if layer is not None:
        #         # print(input.shape)
        #         # print(layer)
        #         if isinstance(layer, nn.BatchNorm1d) and (input.ndim == 3):
        #             # xres = []
        #             # for xi in input:
        #             #     xii = layer(xi)
        #             #     # print("xii", xii.shape)
        #             #     xres.append(xii.unsqueeze(0))
        #             # input = torch.cat(xres, 0)
        #             n_features = input.shape[-1]
        #             original_shape = input.shape
        #             x2d = input.view(-1, n_features)
        #             x2d = layer(x2d)
        #             input = x2d.view(*original_shape)

        #         else:
        #             input = layer(input)
        if input.ndim == 3:
            n_features = input.shape[-1]
            original_shape = input.shape
            x2d = input.view(-1, n_features)
            x2d = self.decoder_n(x2d)
            input = x2d.view(original_shape[0], original_shape[1], -1)
        else:
            input = self.decoder_n(input)

        # z_ = input
        # h = self.decoder_n(z_)

        h = input

        x_scale = self.decoder_scale(h)
        x_var = self.decoder_std(h)
        return dict(
            x_scale=x_scale,
            x_var=x_var,
        )


class SCSphere(nn.Module):
    def __init__(
        self, 
        n_genes, 
        n_batches, 
        n_latent, 
        dropout_rate, do_depth_reg=True, use_layer_norm=True, use_batch_norm=False, library_nn=True, constant_pxr=False, scale_norm="softmax", 
        cell_specific_px=False,
        deep_architecture=True
    ) -> None:
        super().__init__()
        norm_type = None
        if use_layer_norm:
            norm_type = nn.LayerNorm
        elif use_batch_norm:
            norm_type = nn.BatchNorm1d
        else:
            raise ValueError
        self.z_encoder = SphereEncoder(
            n_input=n_genes,
            z_dim=n_latent,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            deep_architecture=deep_architecture,
        )

        self.decoder = SphereDecoder(
            n_input=n_latent + n_batches, n_output=n_genes, use_layer_norm=use_layer_norm, use_batch_norm=use_batch_norm, scale_norm=scale_norm,
            deep_architecture=deep_architecture,
        )
        self.n_genes = n_genes
        self.n_batches = n_batches
        self.n_latent = n_latent
        self.eps = 1e-8
        self.do_depth_reg = do_depth_reg
        self.cell_specific_px = cell_specific_px

        mhalf = self.n_latent / 2
        log_pz = -np.log((2.0 * np.pi ** (mhalf)) / gamma(mhalf))
        self.pz_cst = log_pz
        self.px_r_cst = torch.nn.Parameter(
            torch.randn(
                n_genes,
            ), requires_grad=True
        )
        self.constant_pxr = constant_pxr

        self.library_nn = library_nn
        if self.library_nn:
            self.library_size_nn = nn.Sequential(
                nn.Linear(n_genes + 1, 128),
                norm_type(128),
                nn.ELU(),
                nn.Linear(128, 1),
            )
        else:
            self.library_size_nn = None

    #         self.px_r_cst = torch.nn.Parameter(mdl_vae.px_r.data, requires_grad=False)

    def inference(
        self, x, batch_index=None, y=None, reparam=True, n_samples=1, train_library=True
    ):
        if self.n_batches >= 1:
            batch = F.one_hot(batch_index.squeeze().long(), num_classes=self.n_batches)
        else:
            batch = None
        n_examples = batch_index.shape[0]
        library_size = x.sum(-1, keepdims=True)

        x_ = torch.log1p(x).float()
        z_post = self.z_encoder(x_, n_samples=n_samples, reparam=reparam)
        qz_m = z_post["qz_m"]

        z_ = z_post["z"]
        if self.n_batches >= 1:
            batch_3d = (
                batch.unsqueeze(0)
                .expand((n_samples, n_examples, self.n_batches))
                .float()
            )
            z_ = torch.cat([z_, batch_3d.float()], -1).float()
        decoder_ = self.decoder(z_)
        x_scale = decoder_["x_scale"]

        x_b = torch.cat([x, library_size], -1)
        if self.library_size_nn is not None:
            library_size = self.library_size_nn(x_b).exp()
            # library_size = nn.Softplus()(self.library_size_nn(x_b))

        x_m = library_size * x_scale

        x_v = decoder_["x_var"]
        px_r = torch.clamp(x_v, min=-15, max=15).exp()
        # print(px_r.shape, px_r.max())
        # Computing regularization part

        if self.do_depth_reg:
            x_subsampled = db.Poisson(0.2 * x).sample()
            x_ = torch.log1p(x_subsampled).float()
            z_post_subs = self.z_encoder(x_)
            qz_m_subs = z_post_subs ["qz_m"]
            depth_reg = (((qz_m - qz_m_subs) ** 2).sum(-1)).mean()
        else:
            depth_reg = 0.0

        return dict(
            ql_m=None,
            ql_v=None,
            library=library_size.unsqueeze(0).expand((n_samples, n_examples, 1)),
            log_ql_x=None,
            is_lognormal_library=None,
            qz_m=qz_m.unsqueeze(0).expand((n_samples, n_examples, self.n_latent)),
            qz_v=z_post["qz_v"].unsqueeze(0).expand((n_samples, n_examples, 1)),
            z=z_post["z"],
            log_qz_x=None,
            px_scale=x_scale,
            px_r=px_r,
            px_rate=x_m,
            px_dropout=None,
            depth_reg=depth_reg,
        )

    def _inference_is(
        self,
        x,
        z_candidate,
        batch_index=None,
        y=None,
        reparam=True,
        n_samples=1,
    ):
        if self.n_batches >= 1:
            batch = F.one_hot(batch_index.squeeze().long(), num_classes=self.n_batches)
        else:
            batch = None
        n_examples = batch_index.shape[0]
        library_size = x.sum(-1, keepdims=True)

        x_ = torch.log1p(x).float()
        z_post = self.z_encoder(x_, n_samples=n_samples, reparam=reparam)

        z_ = z_candidate
        n_samples_z = z_candidate.shape[0]
        if self.n_batches >= 1:
            batch_3d = (
                batch.unsqueeze(0)
                .expand((n_samples_z, n_examples, self.n_batches))
                .float()
            )
            z_ = torch.cat([z_, batch_3d.float()], -1).float()
        decoder_ = self.decoder(z_)
        x_scale = decoder_["x_scale"]

        x_b = torch.cat([x, library_size], -1)
        if self.library_size_nn is not None:
            library_size = self.library_size_nn(x_b).exp()
            # library_size = nn.Softplus()(self.library_size_nn(x_b))
        x_m = library_size * x_scale
        px_rate = x_m

        x_v = decoder_["x_var"]
        px_r = torch.clamp(x_v, min=-15, max=15).exp()
        if not self.cell_specific_px:
            px_r = px_r.mean(0)

        if self.constant_pxr:
            # px_r = self.px_r_cst.exp()
            px_r = nn.Softplus()(self.px_r_cst)


        log_pz = HypersphericalUniform(dim=self.n_latent - 1, device="cuda").log_prob(
            z_candidate
        )
        #         log_qz_x = VonMisesFisher(z_post["qz_m"], z_post["qz_v"]).log_prob(z_candidate)
        log_qz_x = PowerSpherical(z_post["qz_m"], z_post["qz_v"].squeeze(-1)).log_prob(
            z_candidate
        )
        log_px_zl = log_nb_positive(x, mu=px_rate, theta=px_r, eps=self.eps)
        log_pxg_zl = log_nb_positive(
            x, mu=px_rate, theta=px_r, eps=self.eps, return_sum=False
        )

        return dict(
            ql_m=None,
            ql_v=None,
            qz_m=None,
            qz_v=None,
            log_qz_x=log_qz_x,
            log_pz=log_pz,
            log_px_zl=log_px_zl,
            px_scale=x_scale,
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
    ):

        outputs = self.inference(
            x,
            batch_index,
            y,
            n_samples=n_samples,
            train_library=train_library,
            reparam=True,
        )
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        z = outputs["z"]
        px_scale = outputs["px_scale"]
        depth_reg = outputs["depth_reg"]
        if not self.cell_specific_px:
            px_r = px_r.mean(0)
        
        if self.constant_pxr:
            # px_r = self.px_r_cst.exp()
            px_r = nn.Softplus()(self.px_r_cst)
        # print(px_r.shape, px_r.max())
        log_px_zl = log_nb_positive(x, mu=px_rate, theta=px_r, eps=self.eps)

        log_pz = self.pz_cst
        log_pz = HypersphericalUniform(dim=self.n_latent - 1, device="cuda").log_prob(z)
        #         log_qz_x = VonMisesFisher(outputs["qz_m"], outputs["qz_v"]).log_prob(z)
        log_qz_x = PowerSpherical(
            outputs["qz_m"], outputs["qz_v"].squeeze(-1)
        ).log_prob(z)
        log_qz_x_detach = PowerSpherical(
            outputs["qz_m"].detach(), outputs["qz_v"].detach().squeeze(-1)
        ).log_prob(z)
        log_ratio = log_px_zl + log_pz - log_qz_x
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
        elif loss == "DREG":
            # log_qz_x_detach = outputs["log_qz_x_detach"]
            lw = (log_px_zl + log_pz) - (log_qz_x_detach)
            with torch.no_grad():
                reweight = torch.exp(lw - torch.logsumexp(lw, 0))
                z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)

            obj = -(reweight * lw).sum(0).mean(0)
        elif loss == "logratios":
            obj = log_ratio
        elif loss == "probs":
            return {
                "log_px_zl": log_px_zl,
                "log_pl": None,
                "log_pz": log_pz,
                "log_qz_x": log_qz_x,
                "log_ql_x": None,
                "log_ratio": log_ratio,
            }

        return obj + depth_reg

    def get_scale(self, z, library, batch_index):
        n_samples = z.shape[0]
        n_examples = batch_index.shape[0]
        if self.n_batches >= 1:
            batch = F.one_hot(batch_index.squeeze().long(), num_classes=self.n_batches)
        else:
            batch = None
        z_ = z
        # if self.n_batches >= 1:
        #     batch_3d = batch.unsqueeze(0).expand(
        #         (n_samples, n_examples, self.n_batches)
        #     )
        #     z_ = torch.cat([z_, batch_3d], -1).float()
        if self.n_batches >= 1:
            batch_3d = (
                batch.unsqueeze(0)
                .expand((n_samples, n_examples, self.n_batches))
                .float()
            )
            z_ = torch.cat([z_, batch_3d.float()], -1).float()
        decoder_ = self.decoder(z_)
        x_scale = decoder_["x_scale"]
        return x_scale

    def get_scale_(self, z, library, batch_index):
        """Here only working with 2d inputs, and suppose that z and library are aligned

        :param z: [description]
        :type z: [type]
        :param library: [description]
        :type library: [type]
        :param batch_index: [description]
        :type batch_index: [type]
        :return: [description]
        :rtype: [type]
        """
        if self.n_batches >= 1:
            batch = F.one_hot(batch_index.squeeze().long(), num_classes=self.n_batches)
        else:
            batch = None
        z_ = z
        if self.n_batches >= 1:
            z_ = torch.cat([z_, batch.float()], -1).float()
        decoder_ = self.decoder(z_)
        x_scale = decoder_["x_scale"]
        return x_scale


class SCSphereFull(nn.Module):
    def __init__(
        self, 
        n_genes, 
        n_batches, 
        n_latent, 
        dropout_rate, do_depth_reg=True, use_layer_norm=True, use_batch_norm=False, library_nn=True, constant_pxr=False, scale_norm="softmax", 
        cell_specific_px=False,
        deep_architecture=True,
        do_px_reg=False,
    ) -> None:
        super().__init__()
        norm_type = None
        if use_layer_norm:
            norm_type = nn.LayerNorm
        elif use_batch_norm:
            norm_type = nn.BatchNorm1d
        else:
            raise ValueError
        self.z_encoder = SphereEncoder(
            n_input=n_genes,
            z_dim=n_latent,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            deep_architecture=deep_architecture,
        )
        self.l_encoder = Encoder(
            n_genes,
            1,
            n_layers=1,
            n_hidden=128,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            with_activation=nn.ELU(),
            # sum_feature=sum_feature,
        )

        self.decoder = SphereDecoder(
            n_input=n_latent + n_batches, n_output=n_genes, use_layer_norm=use_layer_norm, use_batch_norm=use_batch_norm, scale_norm=scale_norm,
            deep_architecture=deep_architecture,
        )
        self.n_genes = n_genes
        self.n_batches = n_batches
        self.n_latent = n_latent
        self.eps = 1e-8
        self.do_depth_reg = do_depth_reg
        self.cell_specific_px = cell_specific_px

        mhalf = self.n_latent / 2
        log_pz = -np.log((2.0 * np.pi ** (mhalf)) / gamma(mhalf))
        self.pz_cst = log_pz
        self.px_r_cst = torch.nn.Parameter(
            torch.randn(
                n_genes,
            ), requires_grad=True
        )
        self.constant_pxr = constant_pxr
        self.do_px_reg = do_px_reg


    #         self.px_r_cst = torch.nn.Parameter(mdl_vae.px_r.data, requires_grad=False)

    def inference(
        self, x, batch_index=None, y=None, reparam=True, n_samples=1, train_library=True
    ):
        if self.n_batches >= 1:
            batch = F.one_hot(batch_index.squeeze().long(), num_classes=self.n_batches)
        else:
            batch = None
        n_examples = batch_index.shape[0]
        library_size = x.sum(-1, keepdims=True)

        x_ = torch.log1p(x).float()
        z_post = self.z_encoder(x_, n_samples=n_samples, reparam=reparam)
        l_post = self.l_encoder(x_, n_samples=n_samples, reparam=reparam)
        qz_m = z_post["qz_m"]
        
        library = l_post["latent"]
        # print(library)
        z_ = z_post["z"]
        if self.n_batches >= 1:
            batch_3d = (
                batch.unsqueeze(0)
                .expand((n_samples, n_examples, self.n_batches))
                .float()
            )
            z_ = torch.cat([z_, batch_3d.float()], -1).float()
        decoder_ = self.decoder(z_)
        x_scale = decoder_["x_scale"]

        x_m = library.exp() * x_scale
        x_v = decoder_["x_var"]
        px_r = torch.clamp(x_v, min=-15, max=15).exp()
        # print(px_r.shape, px_r.max())
        # Computing regularization part

        if self.do_depth_reg:
            x_subsampled = db.Poisson(0.2 * x).sample()
            x_ = torch.log1p(x_subsampled).float()
            z_post_subs = self.z_encoder(x_)
            qz_m_subs = z_post_subs ["qz_m"]
            depth_reg = (((qz_m - qz_m_subs) ** 2).sum(-1)).mean()
        else:
            depth_reg = 0.0

        if self.do_px_reg:
            px_reg = px_r.var(-1).mean()
        else:
            px_reg = 0.

        return dict(
            library=library,
            ql_m=l_post["q_m"],
            ql_v=l_post["q_v"],
            log_ql_x=None,
            is_lognormal_library=None,
            qz_m=qz_m.unsqueeze(0).expand((n_samples, n_examples, self.n_latent)),
            qz_v=z_post["qz_v"].unsqueeze(0).expand((n_samples, n_examples, 1)),
            z=z_post["z"],
            log_qz_x=None,
            px_scale=x_scale,
            px_r=px_r,
            px_rate=x_m,
            px_dropout=None,
            depth_reg=depth_reg,
            px_reg=px_reg,
        )

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
    ):
        if self.n_batches >= 1:
            batch = F.one_hot(batch_index.squeeze().long(), num_classes=self.n_batches)
        else:
            batch = None
        n_examples = batch_index.shape[0]
        library_size = x.sum(-1, keepdims=True)

        x_ = torch.log1p(x).float()
        z_post = self.z_encoder(x_, n_samples=n_samples, reparam=reparam)

        library_post = self.l_encoder(x_, n_samples=n_samples, reparam=reparam)
        ql_m = library_post["q_m"]
        ql_v = library_post["q_v"]
        # Using the mean library size
        if l_candidate is None:
            l_map = library_post["q_m"]  # log scaled
        else:
            l_map = l_candidate

        z_ = z_candidate
        n_samples_z = z_candidate.shape[0]
        if self.n_batches >= 1:
            batch_3d = (
                batch.unsqueeze(0)
                .expand((n_samples_z, n_examples, self.n_batches))
                .float()
            )
            z_ = torch.cat([z_, batch_3d.float()], -1).float()
        decoder_ = self.decoder(z_)
        x_scale = decoder_["x_scale"]

        x_m = l_map.exp() * x_scale
        px_rate = x_m

        x_v = decoder_["x_var"]
        px_r = torch.clamp(x_v, min=-15, max=15).exp()
        if not self.cell_specific_px:
            px_r = px_r.mean(0)

        if self.constant_pxr:
            px_r = self.px_r_cst.exp()
            # px_r = nn.Softplus()(self.px_r_cst)


        log_pz = HypersphericalUniform(dim=self.n_latent - 1, device="cuda").log_prob(
            z_candidate
        )
        #         log_qz_x = VonMisesFisher(z_post["qz_m"], z_post["qz_v"]).log_prob(z_candidate)
        log_qz_x = PowerSpherical(z_post["qz_m"], z_post["qz_v"].squeeze(-1)).log_prob(
            z_candidate
        )
        log_px_zl = log_nb_positive(x, mu=px_rate, theta=px_r, eps=self.eps)
        log_pxg_zl = log_nb_positive(
            x, mu=px_rate, theta=px_r, eps=self.eps, return_sum=False
        )
        if l_candidate is None:
            log_ql_x = None
            log_pl = None
        else:
            log_pl = (
                db.Normal(local_l_mean, torch.sqrt(local_l_var))
                .log_prob(l_map)
                .sum(dim=-1)
            )
            log_ql_x = db.Normal(ql_m, ql_v.sqrt()).log_prob(l_map).sum(-1)

        return dict(
            ql_m=None,
            ql_v=None,
            qz_m=None,
            qz_v=None,
            log_qz_x=log_qz_x,
            log_pz=log_pz,
            log_px_zl=log_px_zl,
            px_scale=x_scale,
            log_pxg_zl=log_pxg_zl,
            log_ql_x=log_ql_x,
            log_pl=log_pl,
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
    ):

        outputs = self.inference(
            x,
            batch_index,
            y,
            n_samples=n_samples,
            train_library=train_library,
            reparam=True,
        )
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        z = outputs["z"]
        library = outputs["library"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_scale = outputs["px_scale"]
        depth_reg = outputs["depth_reg"]
        px_reg = outputs["px_reg"]
        if not self.cell_specific_px:
            px_r = px_r.mean(0)
        
        if self.constant_pxr:
            px_r = self.px_r_cst.exp()
            # px_r = nn.Softplus()(self.px_r_cst)
        # print(px_r.shape, px_r.max())
        log_px_zl = log_nb_positive(x, mu=px_rate, theta=px_r, eps=self.eps)

        log_pz = self.pz_cst
        log_pz = HypersphericalUniform(dim=self.n_latent - 1, device="cuda").log_prob(z)
        log_pl = (
            db.Normal(local_l_mean, torch.sqrt(local_l_var))
            .log_prob(library)
            .sum(dim=-1)
        )

        log_qz_x = PowerSpherical(
            outputs["qz_m"], outputs["qz_v"].squeeze(-1)
        ).log_prob(z)
        log_qz_x_detach = PowerSpherical(
            outputs["qz_m"].detach(), outputs["qz_v"].detach().squeeze(-1)
        ).log_prob(z)

        log_ql_x = db.Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(-1)
        log_ql_x_detach = db.Normal(ql_m.detach(), ql_v.detach().sqrt()).log_prob(library).sum(-1)

        log_ratio = log_px_zl + log_pz + log_pl - log_qz_x - log_ql_x
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
        elif loss == "DREG":
            # log_qz_x_detach = outputs["log_qz_x_detach"]
            lw = (log_px_zl + log_pz + log_pl) - (log_qz_x_detach - log_ql_x_detach)
            with torch.no_grad():
                reweight = torch.exp(lw - torch.logsumexp(lw, 0))
                z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)

            obj = -(reweight * lw).sum(0).mean(0)
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

        return obj + depth_reg + px_reg

    def get_scale(self, z, library, batch_index):
        n_samples = z.shape[0]
        n_examples = batch_index.shape[0]
        if self.n_batches >= 1:
            batch = F.one_hot(batch_index.squeeze().long(), num_classes=self.n_batches)
        else:
            batch = None
        z_ = z
        # if self.n_batches >= 1:
        #     batch_3d = batch.unsqueeze(0).expand(
        #         (n_samples, n_examples, self.n_batches)
        #     )
        #     z_ = torch.cat([z_, batch_3d], -1).float()
        if self.n_batches >= 1:
            batch_3d = (
                batch.unsqueeze(0)
                .expand((n_samples, n_examples, self.n_batches))
                .float()
            )
            z_ = torch.cat([z_, batch_3d.float()], -1).float()
        decoder_ = self.decoder(z_)
        x_scale = decoder_["x_scale"]
        return x_scale

    def get_scale_(self, z, library, batch_index):
        """Here only working with 2d inputs, and suppose that z and library are aligned

        :param z: [description]
        :type z: [type]
        :param library: [description]
        :type library: [type]
        :param batch_index: [description]
        :type batch_index: [type]
        :return: [description]
        :rtype: [type]
        """
        if self.n_batches >= 1:
            batch = F.one_hot(batch_index.squeeze().long(), num_classes=self.n_batches)
        else:
            batch = None
        z_ = z
        if self.n_batches >= 1:
            z_ = torch.cat([z_, batch.float()], -1).float()
        decoder_ = self.decoder(z_)
        x_scale = decoder_["x_scale"]
        return x_scale


class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if isinstance(val, torch.device) else torch.device(val)

    def __init__(self, dim, validate_args=None, device="cpu"):
        super(HypersphericalUniform, self).__init__(
            torch.Size([dim]), validate_args=validate_args
        )
        self._dim = dim
        self.device = device

    def sample(self, shape=torch.Size()):
        output = (
            torch.distributions.Normal(0, 1)
            .sample(
                (shape if isinstance(shape, torch.Size) else torch.Size([shape]))
                + torch.Size([self._dim + 1])
            )
            .to(self.device)
        )

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

    def __log_surface_area(self):
        if torch.__version__ >= "1.0.0":
            lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self.device))
        else:
            lgamma = torch.lgamma(
                torch.Tensor([(self._dim + 1) / 2], device=self.device)
            )
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma


