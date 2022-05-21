import collections
from typing import Iterable, List
import logging

import torch
from torch import nn as nn
from torch.distributions import Normal, LogNormal
from torch.nn import ModuleList
from torch.nn.utils import weight_norm
from scvi.models.utils import one_hot


logger = logging.getLogger(__name__)


def tril_indices(rows, cols, offset=0):
    return torch.ones(rows, cols, dtype=torch.uint8).tril(offset).nonzero()


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def get_one_hot_cat_list(n_cat_list, cat_list):
    one_hot_cat_list = []  # for generality in this list many indices useless.
    assert len(n_cat_list) <= len(
        cat_list
    ), "nb. categorical args provided doesn't match init. params."
    for n_cat, cat in zip(n_cat_list, cat_list):
        assert not (
            n_cat and cat is None
        ), "cat not provided while n_cat != 0 in init. params."
        if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
            if cat.size(1) != n_cat:
                one_hot_cat = one_hot(cat, n_cat)
            else:
                one_hot_cat = cat  # cat has already been one_hot encoded
            one_hot_cat_list += [one_hot_cat]
    return one_hot_cat_list

class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_cat_list: A list containing, for each category of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: Whether to have `BatchNorm` layers or not
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_weight_norm: bool = False,
        use_layer_norm: bool = False,
        use_layer_normb: bool = False,
        with_activation: bool = nn.ReLU(),
        do_alpha_dropout: bool = False,
    ):
        # assert not use_batch_norm
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        dropout_class = nn.AlphaDropout if do_alpha_dropout else nn.Dropout
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        def layer_definer(my_n_in, my_n_out):
            if use_weight_norm:
                return weight_norm(nn.Linear(my_n_in + sum(self.n_cat_list), my_n_out))
            else:
                return nn.Linear(my_n_in + sum(self.n_cat_list), my_n_out)

        self.activation = with_activation

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            layer_definer(my_n_in=n_in, my_n_out=n_out),
                            # nn.Linear(n_in + sum(self.n_cat_list), n_out),
                            # nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            nn.BatchNorm1d(n_out)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out) if use_layer_norm else None,
                            nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_normb else None,
                            self.activation,
                            # nn.Tanh() if with_activation else None,
                            dropout_class(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :param instance_id: Use a specific conditional instance normalization (batchnorm)
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = get_one_hot_cat_list(self.n_cat_list, cat_list)

        if sum(self.n_cat_list) >= 1:
            batch_features = torch.cat(one_hot_cat_list, dim=-1)
            n_examples, n_batch_cat = batch_features.shape
            if x.ndim == 3:
                n_particles = len(x)
                batch_features = batch_features.unsqueeze(0).expand(
                    [n_particles, n_examples, n_batch_cat]
                )
        else:
            batch_features = torch.tensor([], device=x.device)
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d) and (x.ndim == 3):
                        # xres = []
                        # for xi in x:
                        #     xii = layer(xi)
                        #     # print("xii", xii.shape)
                        #     xres.append(xii.unsqueeze(0))
                        # x = torch.cat(xres, 0)
                        # print("x", x.shape)
                        n_features = x.shape[-1]
                        original_shape = x.shape
                        x2d = x.view(-1, n_features)
                        x2d = layer(x2d)
                        x = x2d.view(*original_shape)

                    else:
                        if isinstance(layer, nn.Linear):
                            x = torch.cat([x, batch_features], -1)
                        x = layer(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_weight_norm: bool = False,
        use_layer_norm: bool = False,
        with_activation: bool = nn.ReLU(),
    ):
        super().__init__()
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.layer1 = FCLayers(
            n_in=n_in,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,  # Should be useless
            with_activation=with_activation,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm,
            dropout_rate=dropout_rate,
        )
        self.layer2 = FCLayers(
            n_in=n_hidden,
            n_out=n_out,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,  # Should be useless
            with_activation=None,
            # with_activation=with_activation,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm,
            dropout_rate=dropout_rate,
        )

        if n_cat_list is None:
            in_features = n_in
        else:
            in_features = n_in + sum(self.n_cat_list)
        if in_features != n_out:
            self.adjust = nn.Linear(in_features=in_features, out_features=n_out)
        else:
            self.adjust = nn.Sequential()

        # self.last_bn = nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
        self.activation = with_activation

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        one_hot_cat_list = get_one_hot_cat_list(self.n_cat_list, cat_list)

        h = self.layer1(x, *cat_list, instance_id=instance_id)
        h = self.layer2(h, *cat_list, instance_id=instance_id)

        # Residual connection adjustments if needed
        if x.dim() == 3:
            one_hot_cat_list = [
                o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                for o in one_hot_cat_list
            ]
        x = torch.cat((x, *one_hot_cat_list), dim=-1)
        x_adj = self.adjust(x)

        h = h + x_adj
        h = self.activation(h)
        return h


class SkipBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        with_activation: bool = nn.ReLU(),
    ):
        super().__init__()
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        if n_cat_list is None:
            ncats = 0
        else:
            ncats = sum(self.n_cat_list)

        self.layer1 = nn.Sequential(
            # nn.Linear(n_in + sum(self.n_cat_list), n_out),
            nn.LayerNorm(n_in + ncats),
            nn.Linear(n_in + ncats, n_hidden),
            # with_activation,
            # nn.Tanh() if with_activation else None,
            # nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
        )

        self.layer2 = nn.Sequential(
            nn.LayerNorm(n_hidden + ncats),
            nn.Linear(n_hidden + ncats, n_out),
        )
        self.dropout_rate = dropout_rate
        self.adjust_middle = nn.Linear(in_features=n_in, out_features=n_hidden + ncats)
        self.adjust = nn.Linear(in_features=n_in, out_features=n_out)
        self.activation = with_activation

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        one_hot_cat_list = get_one_hot_cat_list(self.n_cat_list, cat_list)
        # Residual connection adjustments if needed
        if x.dim() == 3:
            one_hot_cat_list = [
                o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                for o in one_hot_cat_list
            ]
        _x = torch.cat((x, *one_hot_cat_list), dim=-1)

        h = self.layer1(_x)
        resid = self.adjust_middle(x)
        _h = h + resid
        _h = self.activation(_h)
        _h = nn.Dropout(self.dropout_rate)(_h)
        _h = torch.cat((_h, *one_hot_cat_list), dim=-1)
        _h = self.layer2(_h)
        x_adj = self.adjust(x)
        _h = _h + x_adj
        _h = self.activation(_h)
        _h = nn.Dropout(self.dropout_rate)(_h)
        return _h


class DenseResNet(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_blocks: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_weight_norm: bool = False,
        use_layer_norm: bool = False,
        with_activation: bool = nn.ReLU(),
    ):
        super().__init__()

        # First block
        modules = nn.ModuleList()
        modules.append(
            ResNetBlock(
                n_in=n_in,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_weight_norm=use_weight_norm,
                use_layer_norm=use_layer_norm,
                with_activation=with_activation,
            )
        )
        # Intermediary blocks
        for block in range(n_blocks - 2):
            modules.append(
                ResNetBlock(
                    n_in=n_hidden,
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_weight_norm=use_weight_norm,
                    use_layer_norm=use_layer_norm,
                    with_activation=with_activation,
                )
            )
        # Last Block
        modules.append(
            ResNetBlock(
                n_in=n_hidden,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_weight_norm=use_weight_norm,
                use_layer_norm=use_layer_norm,
                with_activation=with_activation,
            )
        )
        self.resnet_layers = modules

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        h = x
        for module in self.resnet_layers:
            h = module(h, *cat_list, instance_id=instance_id)
        return h


class LinearExpLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        use_batch_norm=True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.linear_layer = nn.Sequential(
            nn.Linear(n_in, n_out),
            # nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
        )

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        for layer in self.linear_layer:
            if layer is not None:
                x = layer(x)
        return torch.clamp(x.exp(), max=1e6)


# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_weight_norm: bool = False,
        use_layer_norm: bool = False,
        use_layer_normb: bool = False,
        with_activation=nn.ReLU(),
        prevent_saturation: bool = False,
        sum_feature: bool = False,
        constant_std: bool = False,
        do_alpha_dropout: bool = False,
    ):
        super().__init__()
        self.prevent_saturation = prevent_saturation

        self.sum_feature = sum_feature
        n_inp = n_input
        if sum_feature:
            n_inp += 1
        self.encoder = FCLayers(
            n_in=n_inp,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            with_activation=with_activation,
            use_batch_norm=use_batch_norm,
            use_weight_norm=use_weight_norm,
            use_layer_norm=use_layer_norm,
            use_layer_normb=use_layer_normb,
            dropout_rate=dropout_rate,
            do_alpha_dropout=do_alpha_dropout,
        )

        n_hid = n_hidden
        if sum_feature:
            n_hid += 1
        if use_batch_norm:
            self.mean_encoder = weight_norm(nn.Linear(n_hid, n_output))
        else:
            self.mean_encoder = nn.Linear(n_hid, n_output)

        self.mean_softplus = False
        self.constant_std = constant_std
        if constant_std:
            # parameter chosen such that the standard
            # deviation ~ 0.2
            self._var_encoder = nn.Parameter(torch.tensor(-3.21887), requires_grad=True)
        else:
            if use_batch_norm:
                self._var_encoder = weight_norm(nn.Linear(n_hid, n_output))
            else:
                self._var_encoder = nn.Linear(n_hid, n_output)


    def var_encoder(self, q):
        if self.constant_std:
            return self._var_encoder
        else:
            return self._var_encoder(q)

    def forward(
        self, x: torch.Tensor, *cat_list: int, n_samples=1, reparam=True, squeeze=True, multiplicative_std=None,
    ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution

        x__ = x
        if self.sum_feature:
            x_orig = torch.expm1(x)
            x_sum = x_orig.sum(-1, keepdims=True).log()
            x__ = torch.cat([x__, x_sum], -1)
        q = self.encoder(x__, *cat_list)

        if self.sum_feature:
            q = torch.cat([q, x_sum], -1)
        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = nn.Softplus()(q_v)

        # if self.constant_std:
        #     q_v = q_v * torch.ones_like(q_m)
        if (n_samples > 1) or (not squeeze):
            q_m = q_m.unsqueeze(0).expand((n_samples, q_m.size(0), q_m.size(1)))
            q_v = q_v.unsqueeze(0).expand((n_samples, q_v.size(0), q_v.size(1)))
        
        if multiplicative_std is not None:
            q_v = (multiplicative_std ** 2) * q_v
        dist = Normal(q_m, q_v.sqrt())
        # dist = Normal(q_m, q_v)
        # latent = self.reparameterize(q_m, q_v, reparam=reparam)
        latent = dist.rsample()
        post_density = dist.log_prob(latent).sum(-1)
        return dict(
            q_m=q_m,
            q_v=q_v,
            latent=latent,
            posterior_density=post_density,
            dist=dist,
            sum_last=True,
        )


class LogNormalEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_weight_norm: bool = False,
        sum_feature: bool = False,
    ):
        super().__init__()

        self.sum_feature = sum_feature
        n_inp = n_input
        if sum_feature:
            n_inp += 1
        self.encoder = FCLayers(
            n_in=n_inp,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_weight_norm=use_weight_norm,
            dropout_rate=dropout_rate,
        )

        self.constant_std = True
        n_hid = n_hidden
        if sum_feature:
            n_hid += 1

        self.mean_encoder = nn.Linear(n_hid, n_output)
        self._var_encoder = nn.Parameter(torch.tensor(-3.21887), requires_grad=True)

        self.activation = nn.Softplus()

    def reparameterize(self, mu, var, reparam=True):
        if reparam:
            latent = LogNormal(mu, var.sqrt()).rsample()
        else:
            latent = LogNormal(mu, var.sqrt()).sample()
        return latent

    def var_encoder(self, q):
        return self._var_encoder

    def forward(
        self, x: torch.Tensor, *cat_list: int, n_samples=1, reparam=True, squeeze=True
    ):
        # Parameters for latent distribution

        x__ = x
        if self.sum_feature:
            x_orig = x.exp() - 1.0
            x_sum = (x_orig).sum(-1, keepdims=True)
            x_sum = (x_sum + 1.0).log()
            x__ = torch.cat([x_orig / x_sum, x_sum], -1)
        q = self.encoder(x__, *cat_list)

        if self.sum_feature:
            q = torch.cat([q, x_sum], -1)
        q_m = self.mean_encoder(q)
        q_m = self.activation(q_m)

        q_v = torch.exp(
            self.var_encoder(q)
        )  # (computational stability safeguard)torch.clamp(, -5, 5)
        if self.constant_std:
            q_v = q_v * torch.ones_like(q_m)
        if (n_samples > 1) or (not squeeze):
            q_m = q_m.unsqueeze(0).expand((n_samples, q_m.size(0), q_m.size(1)))
            q_v = q_v.unsqueeze(0).expand((n_samples, q_v.size(0), q_v.size(1)))
        dist = LogNormal(q_m, q_v.sqrt())
        latent = self.reparameterize(q_m, q_v, reparam=reparam)

        post_density = dist.log_prob(latent).sum(-1)
        return dict(
            q_m=q_m,
            q_v=q_v,
            latent=latent,
            posterior_density=post_density,
            dist=dist,
            sum_last=True,
            lognormal=True,
        )


# Decoder
class DecoderSCVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_blocks: int = 0,
        scale_normalize: bool = "softmax",
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        use_layer_normb: bool = False,
        use_weight_norm: bool = False,
        with_activation=nn.ReLU(),
        do_last_skip: bool = False,
        decoder_skip: bool = False,
        dropout_rate: float = 0.0,
        do_alpha_dropout: bool = False,
    ):
        super().__init__()
        self.do_last_skip = do_last_skip
        if n_blocks == 0:
            self.px_decoder = FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                use_batch_norm=use_batch_norm,
                with_activation=with_activation,
                use_layer_norm=use_layer_norm,
                use_layer_normb=use_layer_normb,
                use_weight_norm=use_weight_norm,
                dropout_rate=dropout_rate,
                do_alpha_dropout=do_alpha_dropout,
            )
        else:
            # assert use_batch_norm
            logger.info("Using ResNet structure for the Decoder")
            self.px_decoder = DenseResNet(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_blocks=n_blocks,
                n_hidden=n_hidden,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
            )
        if decoder_skip:
            logger.info("Using Skip connections for the Decoder")
            self.px_decoder = SkipBlock(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                with_activation=nn.ReLU(),
            )


        n_in = n_hidden + n_input if do_last_skip else n_hidden
        # mean gamma
        self.scale_normalize = scale_normalize

        if scale_normalize in ["softmax", "thresh"]:
            logging.info("Scale decoder with Softmax normalization")
            self.px_scale_decoder = nn.Sequential(
                nn.Linear(n_in, n_output),
                # nn.Softmax(dim=-1)
            )
        else:
            raise ValueError

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_in, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
        is_lognormal_library=False,
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)

        last_input = px
        if self.do_last_skip:
            last_input = torch.cat([last_input, z], dim=-1)
        px_scale = self.px_scale_decoder(last_input)

        px_scale = nn.Softmax(dim=-1)(px_scale)
        px_dropout = self.px_dropout_decoder(last_input)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        lib = library
        if not is_lognormal_library:
            lib = lib.exp()

        if self.scale_normalize == "thresh":
            mean = px_scale.log() + lib.log()
            # log_sc = torch.clamp(mean, min=-1.203)
            mean[mean <= -1.203] = -float("inf")
            log_sc = mean
            log_sc = log_sc - lib.log()
            px_scale = log_sc.exp()
        px_rate = lib * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(last_input) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

class LinearDecoderSCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super(LinearDecoderSCVI, self).__init__()

        # mean gamma
        self.n_batches = n_cat_list[0]  # Just try a simple case for now
        if self.n_batches > 1:
            self.batch_regressor = nn.Linear(self.n_batches - 1, n_output, bias=False)
        else:
            self.batch_regressor = None

        self.factor_regressor = nn.Linear(n_input, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_input, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
        is_lognormal_library=False,
    ):
        # The decoder returns values for the parameters of the ZINB distribution
        p1_ = self.factor_regressor(z)
        if self.n_batches > 1:
            one_hot_cat = one_hot(cat_list[0], self.n_batches)[:, :-1]
            p2_ = self.batch_regressor(one_hot_cat)
            raw_px_scale = p1_ + p2_
        else:
            raw_px_scale = p1_

        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z)
        lib = library
        if not is_lognormal_library:
            lib = lib.exp()
        px_rate = lib * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        :param x: tensor with shape ``(n_input,)``
        :param cat_list: list of category membership(s) for this sample
        :return: Mean and variance tensors of shape ``(n_output,)``
        :rtype: 2-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m, p_v

class MultiEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_input_list: List[int],
        n_output: int,
        n_hidden: int = 128,
        n_layers_individual: int = 1,
        n_layers_shared: int = 2,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.encoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, head_id: int, *cat_list: int):
        q = self.encoders[head_id](x, *cat_list)
        q = self.encoder_shared(q, *cat_list)

        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent


class MultiDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_conditioned: int = 32,
        n_hidden_shared: int = 128,
        n_layers_conditioned: int = 1,
        n_layers_shared: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        n_out = n_hidden_conditioned if n_layers_shared else n_hidden_shared
        if n_layers_conditioned:
            self.px_decoder_conditioned = FCLayers(
                n_in=n_input,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_layers=n_layers_conditioned,
                n_hidden=n_hidden_conditioned,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_out
        else:
            self.px_decoder_conditioned = None
            n_in = n_input

        if n_layers_shared:
            self.px_decoder_final = FCLayers(
                n_in=n_in,
                n_out=n_hidden_shared,
                n_cat_list=[],
                n_layers=n_layers_shared,
                n_hidden=n_hidden_shared,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_hidden_shared
        else:
            self.px_decoder_final = None

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_in, n_output), nn.Softmax(dim=-1)
        )
        self.px_r_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: int,
        library: torch.Tensor,
        dispersion: str,
        *cat_list: int,
    ):

        px = z
        if self.px_decoder_conditioned:
            px = self.px_decoder_conditioned(px, *cat_list, instance_id=dataset_id)
        if self.px_decoder_final:
            px = self.px_decoder_final(px, *cat_list)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_r, px_rate, px_dropout
