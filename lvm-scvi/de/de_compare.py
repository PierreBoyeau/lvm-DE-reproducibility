import argparse
import os
import matplotlib.pyplot as plt


from utils import name_to_dataset
from de_models import ScVIClassic, Wilcoxon, EdgeR


def parse_args():
    parser = argparse.ArgumentParser(description='Compare methods for a given dataset')
    parser.add_argument('--dataset', type=str, help='Name of considered dataset')
    parser.add_argument('--nb_genes', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    # args = parse_args()
    # dataset_name = args.dataset
    # nb_genes = args.nb_genes

    dataset_name = "powsimr"
    nb_genes = 1200
    save_dir = '.'

    dataset = name_to_dataset[dataset_name]()
    if nb_genes is not None:
        dataset.subsample_genes(new_n_genes=nb_genes)

    models = [
        ScVIClassic(dataset=dataset, reconstruction_loss='zinb', n_latent=5,
                    full_cov=False, do_mean_variance=False,
                    name='scVI_classic'),
        # ScVIClassic(dataset=dataset, reconstruction_loss='zinb', n_latent=5,
        #             full_cov=False, do_mean_variance=True,
        #             name='scVI_mean_variance'),
        # ScVIClassic(dataset=dataset, reconstruction_loss='zinb',
        #             n_latent=5, full_cov=True, do_mean_variance=False, name='scVI_full_covariance'),
        # EdgeR(dataset=dataset, name='EdgeR'),
        # Wilcoxon(dataset=dataset, name='Wilcoxon'),
    ]

    results = {}
    dataframes = {}
    for model in models:
        model_name = model.name
        model.full_init()
        model.train()

        model_perfs = model.predict_de()

        dataframes[model_name] = model_perfs
        results[model_name] = model.precision_recall_curve()

    ###
    import numpy as np
    model_perfs.loc[:, 'scVI_classic_gamma_score'] = np.abs(model_perfs['scVI_classic_gamma_bayes1'])
    results['scVI_classic_gamma'] = model.precision_recall_curve(model_perfs.scVI_classic_gamma_score)

    assert dataset_name == 'powsimr'
    res_df = dataset.gene_properties

    # Precision Recall curve:
    for key in results:
        precision, recall, mAP = results[key]
        plt.plot(recall, precision, alpha=1.0, label='{}@AP={:0.2f}'.format(key, mAP))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall for all models')
    plt.legend()
    plt.savefig(os.path.join(save_dir, '{}_precision_recall.png'.format(dataset_name)))
    plt.show()

    # Saving dataframes if needed for later use
    for key in dataframes:
        res_df = res_df.join(dataframes[key])

    res_df.to_csv(os.path.join(save_dir, '{}_de.tsv'.format(dataset_name)), sep='\t')
