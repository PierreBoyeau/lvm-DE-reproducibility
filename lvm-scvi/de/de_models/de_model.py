from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DEModel:
    def __init__(self, dataset, name=''):
        self.is_fully_init = False
        self.dataset = dataset
        self.name = name
        self.de_gt = pd.Series(np.isin(dataset.gene_names.astype(int), dataset.de_genes_idx),
                               index=dataset.gene_names)
        self.de_pred = None
        pass

    def full_init(self):
        self.is_fully_init = True
        pass

    def train(self):
        pass

    def predict_de(self):
        pass

    def precision_recall_curve(self, de_pred=None, do_plot=False):
        """

        :param model_name:
        :param do_plot:
        :return: precisions, recalls, average precision
        """
        assert self.de_pred is not None
        self.de_gt = self.de_gt.sort_index()
        if de_pred is None:
            self.de_pred = self.de_pred.sort_index()
        else:
            self.de_pred = de_pred.sort_index()
        assert (self.de_pred.index == self.de_gt.index).all()
        precision, recall, _ = precision_recall_curve(y_true=self.de_gt, probas_pred=self.de_pred)
        average_precision = average_precision_score(y_true=self.de_gt, y_score=self.de_pred)

        if do_plot:
            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')
            plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('{} Precision-Recall curve: AP={:0.2f}'.format(self.name, average_precision))
            plt.show()
        return precision, recall, average_precision
