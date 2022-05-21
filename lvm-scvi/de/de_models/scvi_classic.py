from scvi.models import VAE, MeanVarianceVAE
from scvi.inference import UnsupervisedTrainer
from .de_model import DEModel

import numpy as np


class ScVIClassic(DEModel):
    def __init__(self, dataset, reconstruction_loss, n_latent, full_cov=False,
                 do_mean_variance=False, name=''):
        super().__init__(dataset=dataset, name=name)

        self.reconstruction_loss = reconstruction_loss
        self.n_latent = n_latent
        self.full_cov = full_cov

        if do_mean_variance:
            self.model_type = MeanVarianceVAE
        else:
            self.model_type = VAE

        self.model = None
        self.trainer = None

        self.is_fully_init = False

        # early_stopping_kwargs={'early_stopping_metric': 'll',
        #                        'save_best_state_metric': 'll',
        #                        'patience': 15, 'threshold': 3}

    def full_init(self):
        self.model = self.model_type(n_input=self.dataset.nb_genes, n_batch=self.dataset.n_batches,
                                     reconstruction_loss=self.reconstruction_loss,
                                     n_latent=self.n_latent,
                                     full_cov=self.full_cov)
        self.trainer = UnsupervisedTrainer(model=self.model, gene_dataset=self.dataset,
                                           use_cuda=True,
                                           train_size=0.7, kl=1, frequency=1)
        self.is_fully_init = True

    def train(self, **train_params):
        assert self.is_fully_init
        if len(train_params) == 0:
            train_params = {'n_epochs': 150, 'lr': 1e-3}
        self.trainer.train(**train_params)

    def predict_de(self, n_samples=10000, M_permutation=100000, n_for_each=10,
                   idx1=None, idx2=None, mode='rho'):
        assert mode in ['rho', 'gamma']
        full = self.trainer.create_posterior(self.model, self.dataset,
                                             indices=np.arange(len(self.dataset)))

        if idx1 is None and idx2 is None:
            cell_pos1 = np.where(self.dataset.labels.ravel() == 0)[0][:n_for_each]
            cell_pos2 = np.where(self.dataset.labels.ravel() == 1)[0][:n_for_each]
            cell_idx1 = np.isin(np.arange(len(self.dataset)), cell_pos1)
            cell_idx2 = np.isin(np.arange(len(self.dataset)), cell_pos2)
        else:
            cell_idx1 = idx1
            cell_idx2 = idx2

        de_res = full.differential_expression_score(cell_idx1, cell_idx2, n_samples=n_samples,
                                                    M_permutation=M_permutation)
        de_res_gamma = full.differential_expression_gamma(cell_idx1, cell_idx2, n_samples=n_samples,
                                                          M_permutation=M_permutation)

        de_res.loc[:, 'gamma_bayes1'] = de_res_gamma
        de_res = de_res.sort_index()
        self.de_pred = de_res.bayes1.abs()
        de_res.columns = [self.name+'_'+col for col in de_res.columns]
        return de_res
