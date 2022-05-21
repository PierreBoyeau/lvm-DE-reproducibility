from functools import partial
from scvi.dataset import PowSimSynthetic, SymSimDataset, Dataset10X
import numpy as np

class ConstantLFE:
    def __init__(self, dim, amplitude=1):
        assert dim == 2
        self.amplitude = amplitude
        self.dim = dim

    def sample(self, size):
        size = size[0]
        signs = np.ones((size, 2))
        neg_pos = 2*(np.random.random(size)>= 0.5)- 1
        signs[:, 1] = neg_pos
        signs[:, 0] = 0.0
        return self.amplitude*signs  #  10*signs


my_de_lfc = ConstantLFE(dim=2, amplitude=1)


name_to_dataset = {
    'powsimr': partial(PowSimSynthetic, n_genes=2000, cluster_to_samples=5000*np.ones(2, dtype=int),
                       de_p=0.1, de_lfc=my_de_lfc),
    'symsim': partial(SymSimDataset, save_path='/home/ec2-user/scVI/data/DE/symsim'),
    'mouse_vs_human': partial(Dataset10X, filename='hgmm_5k_v3')
}
