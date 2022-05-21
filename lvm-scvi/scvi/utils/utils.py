import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import arviz as az
import logging
import pickle
from scipy.stats import mannwhitneyu
from statsmodels.stats.weightstats import ttest_ind

logger = logging.getLogger(__name__)


def make_dir_if_necessary(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class IterativeDict:
    """
    Structure useful to save metrics for different models over different trainings

    Saved in a nested dictionnary
    Structure:
    model_name ==> metric_name ==> table [n_trainings, ...]
    """

    def __init__(self, model_names):
        self.values = {key: {} for key in model_names}

    def set_values(self, model_name, metric_name, values):
        if metric_name not in self.values[model_name]:
            self.values[model_name][metric_name] = [values]
        else:
            self.values[model_name][metric_name].append(values)

    def to_df(self):
        return pd.DataFrame(self.values)


def plot_traj(history, x=None, **plot_params):
    """
    :param history: (n_sim, n_x_values) array
    :param x: associated x values used for plotting
    :param plot_params: Plot parameters fed to plt.plot
    :return:
    """
    plot_params = {} if plot_params is None else plot_params
    history_np = np.array(history)
    theta_mean = np.mean(history_np, axis=0)
    theta_std = np.std(history_np, axis=0)
    n_iter = len(theta_mean)

    x = np.arange(n_iter) if x is None else x
    plt.plot(x, theta_mean, **plot_params)

    plt.fill_between(
        x=x, y1=theta_mean - theta_std, y2=theta_mean + theta_std, alpha=0.25
    )


def plot_identity():
    xmin, xmax = plt.xlim()
    vals = np.linspace(xmin, xmax, 50)
    plt.plot(vals, vals, "--", label="identity")


def plot_precision_recall(y_test, y_score, label=""):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = {"step": "post"}
    legend = "{0} PR curve: AP={1:0.2f}".format(label, average_precision)

    plt.step(recall, precision, color="b", alpha=0.2, where="post", label=legend)
    plt.fill_between(recall, precision, alpha=0.2, **step_kwargs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])


def compute_hdi(arr, credible_interval=0.64):
    """
    Given array of (simulations, dimensions) computes Highest Density Intervals
    Sample dimension should be first dimension

    :param arr: Array of shape (n_samples, n_genes)
    :param credible_interval:
    :return:
    """
    return az.hpd(arr, credible_interval=credible_interval)


def demultiply(arr1, arr2, factor=2, weights_a=None, weights_b=None):
    """
    Suppose you have at disposal
        arr1 ~ p(h|x_a)
        arr2 ~ p(h|x_b)

    Then artificially increase the sizes on respective arrays
    so that you can sample from
    p(f(h1, h2) | x_a, x_b) under the right assumptions

    :param arr1:
    :param arr2:
    :param weights_a:
    :param weights_b:
    :param factor:
    :return:
    """
    assert arr1.shape == arr2.shape
    n_original = len(arr1)
    idx_1 = np.random.choice(n_original, size=n_original * factor, p=weights_a, replace=True)
    idx_2 = np.random.choice(n_original, size=n_original * factor, p=weights_b, replace=True)
    return arr1[idx_1], arr2[idx_2]


def predict_de_genes(posterior_probas: np.ndarray, desired_fdr: float):
    """

    :param posterior_probas: Shape (n_samples, n_genes)
    :param desired_fdr:
    :return:
    """
    assert posterior_probas.ndim == 1
    sorted_genes = np.argsort(-posterior_probas)
    sorted_pgs = posterior_probas[sorted_genes]
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))
    d = (cumulative_fdr <= desired_fdr).sum()
    pred_de_genes = sorted_genes[:d]
    is_pred_de = np.zeros_like(cumulative_fdr).astype(bool)
    is_pred_de[pred_de_genes] = True
    return is_pred_de


def save_fig(fig, filename, do_cloud=False):
    from chart_studio.plotly import iplot

    if do_cloud:
        iplot(fig, filename=filename)
    fig.write_image("{}.png".format(filename))


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res


def has_lower_mean(samp_a, samp_b, do_non_parametric=True):
    def nonparametric(x_a, x_b):
        stat, pval = mannwhitneyu(x_a, x_b, alternative="less")
        return pval <= 5e-2

    def parametric(x_a, x_b):
        stat, pval, _ = ttest_ind(x_a, x_b, alternative="smaller")
        return pval <= 5e-2

    if do_non_parametric:
        return nonparametric(samp_a, samp_b)
    else:
        return parametric(samp_a, samp_a)


def softmax(x, axis=None):
    """
    Compute the softmax of each element along an axis of X.
    Parameters
    ----------
    x: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.
    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    y = np.atleast_2d(x)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(x.shape) == 1:
        p = p.flatten()
    return p
