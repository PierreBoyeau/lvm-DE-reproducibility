import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import dropout3d
from hyperspherical_vae.distributions import VonMisesFisher

from scvi.models.sparsemax import Sparsemax
from scvi.models.scsphere import SphereEncoder, SphereDecoder, SCSphere
from scvi.models import VAE
from scvi.dataset import GeneExpressionDataset
from scvi.inference import UnsupervisedTrainer


def test_full_eb():
    data = torch.randint(0, 100, (256, 500)).float().cuda()
    batch_ids = torch.randint(0, 3, (256,)).cuda()
    # test training procedure
    dataset_pbmc = GeneExpressionDataset()
    dataset_pbmc.populate_from_data(
        X=data.cpu().numpy(),
        batch_indices=batch_ids.cpu().numpy(),
    )

    mdl = VAE(n_input=500, n_batch=3)
    trainer = UnsupervisedTrainer(
        mdl, dataset_pbmc, train_library=True, k=15, loss_type="IWELBO", test_indices=[]
    )
    trainer.train(n_epochs=2, lr=1e-6)

    post = trainer.create_posterior(
        indices=np.arange(50),
    )
    post.get_latents_full(
        n_samples_overall=5000
    )

    # mdl_sph = SCSphere(n_genes=500, n_batches=3, n_latent=25, dropout_rate=0.0)
    mdl_sph = VAE(n_input=500, n_batch=3)

    trainer = UnsupervisedTrainer(
        mdl_sph, dataset_pbmc, train_library=True, k=15, loss_type="IWELBO", test_indices=[]
    )
    trainer.train(n_epochs=2, lr=1e-6)
    post = trainer.create_posterior(
        indices=np.arange(50),
    )
    post.get_latents_full(
        n_samples_overall=5000
    )

def test_scsphere():
    data = torch.randint(0, 100, (256, 500)).float().cuda()
    n_genes = data.shape[1]
    n_ex = data.shape[0]
    batch_ids = torch.randint(0, 3, (256,)).cuda()
    batches = F.one_hot(batch_ids, 3).cuda()
    n_samples = 3
    n_batches = 3

    # Test distribution
    qz = torch.randn(128, 10)
    qv = torch.randn(128, 1).exp().sqrt()
    dist = VonMisesFisher(
        loc=qz,
        scale=qv,
    )
    z = dist.rsample()

    # Test encoder
    enc = SphereEncoder(n_input=n_genes + n_batches, z_dim=10).cuda()
    x_ = torch.cat([data, batches], -1).float()
    z_post = enc.forward(x_, n_samples=n_samples)

    # Test decoder
    z = z_post["z"]
    batch3d = batches.unsqueeze(0).expand((n_samples, n_ex, n_samples))
    z_ = torch.cat([z, batch3d], -1).float()
    dec = SphereDecoder(n_input=10 + n_batches, n_output=n_genes).cuda()
    decoder_ = dec.forward(z_)

    # Test model
    mdl = SCSphere(
        n_genes=n_genes, n_batches=n_batches, n_latent=10, droupout_rate=0.0
    ).cuda()
    outs = mdl.inference(x=data, batch_index=batch_ids, n_samples=5)
    outs = mdl.forward(
        x=data,
        local_l_mean=None,
        local_l_var=None,
        batch_index=batch_ids,
        y=None,
        loss="IWELBO",
        n_samples=5,
    )
    assert outs.shape == (256,)
    print("toto")

    # test training procedure
    dataset_pbmc = GeneExpressionDataset()
    dataset_pbmc.populate_from_data(
        X=data.cpu().numpy(),
        batch_indices=batch_ids.cpu().numpy(),
    )

    trainer = UnsupervisedTrainer(
        mdl, dataset_pbmc, train_library=True, k=15, loss_type="IWELBO", test_indices=[]
    )
    # get latents
    post_a = trainer.train_set.sequential()
    outputs_a = post_a.get_latents(
        n_samples=25,
        other=True,
        device="cpu",
        train_library=True,
    )
    trainer.train()


def test_sparsemax():
    data = torch.randn(20, 3).cuda()
    act = Sparsemax()
    dat = act(data)
    assert ((dat.sum(-1) - 1.0).abs() <= 1e-6).all()

    data2 = torch.randn(200, 128, 3).cuda()
    x = torch.cat([(act(slice_x)).unsqueeze(0) for slice_x in data2], dim=0)
    print(x)


def test_gamma_noise():
    import torch
    import torch.distributions as db
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import anndata
    import seaborn as sns
    import matplotlib.pyplot as plt

    from scvi.dataset import GeneExpressionDataset
    from scvi.inference import UnsupervisedTrainer
    from scvi.models import VAE
    import os

    torch.manual_seed(0)

    adata = anndata.read_h5ad(
        "/data/yosef2/users/pierreboyeau/data/PBMC_updated/pbmc_final.h5ad"
    )

    mat = np.array(adata.X.todense())
    X = mat
    X_cp = np.copy(X)
    new_adata = anndata.AnnData(X=X_cp)
    plt.hist(mat[mat != 0])
    sc.pp.normalize_total(new_adata, target_sum=1e4)
    sc.pp.log1p(new_adata)
    sc.pp.highly_variable_genes(
        adata=new_adata,
        inplace=True,
        n_top_genes=1000,  # inplace=False looks buggy,
    )
    best_5k = new_adata.var["highly_variable"].values
    Xpbmc = adata.X[:, best_5k]

    dataset_pbmc = GeneExpressionDataset()
    dataset_pbmc.populate_from_data(
        X=Xpbmc,
    )
    mdl_gamma_cyc = VAE(
        n_input=dataset_pbmc.nb_genes,
        n_batch=dataset_pbmc.n_batches,
        use_batch_norm=False,
        n_latent=50,
        reconstruction_loss="gamma",
        dispersion="gene-cell",
        use_weight_norm=True,
        n_layers=1,
    )
    trainer_gamma_cyc = UnsupervisedTrainer(
        model=mdl_gamma_cyc,
        beta_policy="cyclic",
        gene_dataset=dataset_pbmc,
        train_library=True,
        k=1,
        loss_type="ELBO",
        test_indices=[],
    )
    trainer_gamma_cyc.train(n_epochs=250, lr=1e-3)
