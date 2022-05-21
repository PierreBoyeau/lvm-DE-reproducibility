import logging
import sys
import time

from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.model_selection._split import _validate_shuffle_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from scvi.inference.posterior import Posterior

logger = logging.getLogger(__name__)


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class Trainer:
    r"""The abstract Trainer class for training a PyTorch model and monitoring its statistics. It should be
    inherited at least with a .loss() function to be optimized in the training loop.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :use_cuda: Default: ``True``.
        :metrics_to_monitor: A list of the metrics to monitor. If not specified, will use the
            ``default_metrics_to_monitor`` as specified in each . Default: ``None``.
        :benchmark: if True, prevents statistics computation in the training. Default: ``False``.
        :frequency: The frequency at which to keep track of statistics. Default: ``None``.
        :early_stopping_metric: The statistics on which to perform early stopping. Default: ``None``.
        :save_best_state_metric:  The statistics on which we keep the network weights achieving the best store, and
            restore them at the end of training. Default: ``None``.
        :on: The data_loader name reference for the ``early_stopping_metric`` and ``save_best_state_metric``, that
            should be specified if any of them is. Default: ``None``.
        :show_progbar: If False, disables progress bar.
    """
    default_metrics_to_monitor = []

    def __init__(
        self,
        model,
        gene_dataset,
        use_cuda=True,
        metrics_to_monitor=None,
        grad_clip_value=None,
        benchmark=False,
        frequency=None,
        weight_decay=1e-6,
        early_stopping_kwargs=None,
        pin_memory=True,
        data_loader_kwargs=None,
        show_progbar=True,
        batch_size=128,
        lr_policy=None,
        optimizer_type="adam",
    ):
        # handle mutable defaults
        early_stopping_kwargs = (
            early_stopping_kwargs if early_stopping_kwargs else dict()
        )
        data_loader_kwargs = data_loader_kwargs if data_loader_kwargs else dict()

        self.model = model
        self.grad_clip_value = grad_clip_value
        self.gene_dataset = gene_dataset
        self._posteriors = OrderedDict()

        self.data_loader_kwargs = {"batch_size": batch_size, "pin_memory": pin_memory}
        self.batch_size = batch_size
        self.data_loader_kwargs.update(data_loader_kwargs)

        self.weight_decay = weight_decay
        self.benchmark = benchmark
        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.training_time = 0
        self.lr_policy = lr_policy
        self.optimizer_type = optimizer_type

        if metrics_to_monitor is not None:
            self.metrics_to_monitor = set(metrics_to_monitor)
        else:
            self.metrics_to_monitor = set(self.default_metrics_to_monitor)

        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        if self.early_stopping.early_stopping_metric:
            self.metrics_to_monitor.add(self.early_stopping.early_stopping_metric)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

        self.frequency = frequency if not benchmark else None

        self.history = defaultdict(list)

        self.best_state_dict = self.model.state_dict()
        self.best_epoch = self.epoch

        self.show_progbar = show_progbar

        self.train_losses = []
        self.thresholds = []
        self.test_losses = []
        self.debug_loss = 0

    @torch.no_grad()
    def compute_metrics(self):
        begin = time.time()
        epoch = self.epoch + 1
        if self.frequency and (
            epoch == 0 or epoch == self.n_epochs or (epoch % self.frequency == 0)
        ):
            with torch.set_grad_enabled(False):
                self.model.eval()
                logger.debug("\nEPOCH [%d/%d]: " % (epoch, self.n_epochs))

                for name, posterior in self._posteriors.items():
                    message = " ".join([s.capitalize() for s in name.split("_")[-2:]])
                    if posterior.nb_cells < 5:
                        logging.debug(
                            message + " is too small to track metrics (<5 samples)"
                        )
                        continue
                    if hasattr(posterior, "to_monitor"):
                        for metric in posterior.to_monitor:
                            if metric not in self.metrics_to_monitor:
                                logger.debug(message, end=" : ")
                                result = getattr(posterior, metric)()
                                self.history[metric + "_" + name] += [result]
                    for metric in self.metrics_to_monitor:
                        result = getattr(posterior, metric)()
                        self.history[metric + "_" + name] += [result]
                self.model.train()
        self.compute_metrics_time += time.time() - begin

    def iter_step(self, tensors_list):
        if self.optimizer is not None:
            loss = self.loss(*tensors_list)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_value
                )
            self.optimizer.step()
        else:
            for _ in range(self.n_enc_steps):
                loss = self.loss(*tensors_list)
                self.enc_optimizer.zero_grad()
                loss.backward()
                self.enc_optimizer.step()

            loss = self.loss(*tensors_list)
            self.dec_optimizer.zero_grad()
            loss.backward()
            self.dec_optimizer.step()
        return loss



    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        begin = time.time()
        self.n_epochs = n_epochs
        self.iterations_per_epoch = int(self.n_train_examples / self.batch_size)
        self.iterate = 0

        self.model.train()

        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            library_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.n_enc_steps == 1:
            logging.info("Unique optim")
            optimizer = self.optimizer = torch.optim.Adam(
                params, lr=lr, eps=eps, weight_decay=self.weight_decay
            )
        else:
            logging.info("Asynch optim")
            enc_params = list(self.model.z_encoder.parameters()) + list(self.model.l_encoder.parameters())
            dec_params = list(self.model.decoder.parameters()) + [self.model.px_r]
            encoder_params = filter(lambda p: p.requires_grad, enc_params)
            decoder_params = filter(lambda p: p.requires_grad, dec_params)

            self.enc_optimizer = torch.optim.Adam(
                encoder_params, lr=lr, eps=eps, weight_decay=self.weight_decay
            )
            self.dec_optimizer = torch.optim.Adam(
                decoder_params, lr=lr, eps=eps, weight_decay=self.weight_decay
            )
            self.optimizer = None
        scheduler = None
        if self.lr_policy == "SGDR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50
            )

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        self.compute_metrics()
        self.metrics = []

        n_per_epoch = len(self.train_set.data_loader)
        with trange(
            n_epochs, desc="training", file=sys.stdout, disable=not self.show_progbar
        ) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                loss_mean = []
                ij = 0
                for tensors_list in self.data_loaders_loop():
                    loss  = self.iter_step(tensors_list)
                    loss_mean.append(loss.item())
                    self.debug_loss = loss
                    if torch.isnan(loss).any():
                        raise ValueError("NaN in Loss!")
                    if scheduler is not None:
                        scheduler.step(self.epoch + ij / n_per_epoch)

                    ij += 1
                loss_mean = np.mean(loss_mean)
                self.train_losses.append(loss_mean)

                if (len(self.test_set.indices) > 0) and (self.epoch % 25 == 0):
                    # self.model.eval()
                    self.metrics.append(
                        dict(
                            test_ll=self.test_set.marginal_llb(None, 200).mean().cpu().item(),
                            epoch=self.epoch,
                            # test_log_ratios=self.test_set.sequential().get_key(
                            #     "log_ratio"
                            # ),
                            test_reconstruction_loss=self.test_set.sequential()
                            .get_key("log_px_zl")
                            .mean()
                            .cpu()
                            .item(),
                        )
                    )
                    # self.model.train()

                if self.writer is not None:
                    optimizer.zero_grad()
                    loss = self.loss(*tensors_list)
                    loss.backward()
                    self.writer.add_scalar("Train ELBO", loss_mean, self.epoch)
                    for name, param in self.model.named_parameters():
                        grads = param.grad
                        if grads is not None:
                            grads = grads.view(-1).detach().cpu()
                            self.writer.add_histogram(
                                "{name}_grad".format(name=name), grads, self.epoch
                            )
                        self.writer.add_histogram(
                            "param_{name}".format(name=name), param, self.epoch
                        )
                    # with torch.no_grad():
                    #     latents = self.train_set.get_key("qz_m", n_samples=2,).mean(0)
                    #     rdm_keys = np.random.choice(len(latents), 1000, replace=False)
                    #     self.writer.add_embedding(
                    #         latents[rdm_keys],
                    #         metadata=self.gene_dataset.labels.squeeze()[rdm_keys],
                    #         tag="z",
                    #         global_step=self.epoch,
                    #     )
                    optimizer.zero_grad()

                self.iterate += 1
                if not self.on_epoch_end():
                    break

        if self.writer is not None:
            self.writer.close()

        if self.early_stopping.save_best_state_metric is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.compute_metrics()

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time
        if self.frequency:
            logger.debug(
                "\nTraining time:  %i s. / %i epochs"
                % (int(self.training_time), self.n_epochs)
            )

    def find_lr(self, lr=1e-10, eps=0.01):
        self.model.train()
        self.history = {"lr": [], "loss": []}

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.n_enc_steps == 1:
            logging.info("Unique optim")
            optimizer = self.optimizer = torch.optim.Adam(
                params, lr=lr, eps=eps, weight_decay=self.weight_decay
            )
            optimizer_schedule = ExponentialLR(
                optimizer=optimizer, end_lr=10.0, num_iter=100
            )
        else:
            logging.info("Asynch optim")
            enc_params = list(self.model.z_encoder.parameters()) + list(self.model.l_encoder.parameters())
            dec_params = list(self.model.decoder.parameters()) + [self.model.px_r]
            encoder_params = filter(lambda p: p.requires_grad, enc_params)
            decoder_params = filter(lambda p: p.requires_grad, dec_params)

            self.enc_optimizer = torch.optim.Adam(
                encoder_params, lr=lr, eps=eps, weight_decay=self.weight_decay
            )
            self.dec_optimizer = torch.optim.Adam(
                decoder_params, lr=lr, eps=eps, weight_decay=self.weight_decay
            )
            self.optimizer = None
            enc_schedule = ExponentialLR(
                optimizer=self.enc_optimizer, end_lr=10.0, num_iter=100
            )
            dec_schedule = ExponentialLR(
                optimizer=self.dec_optimizer, end_lr=10.0, num_iter=100
            )

        iter = 0
        do_continue = True
        while do_continue:
            for tensors_list in self.data_loaders_loop():
                # loss = self.loss(*tensors_list)
                loss  = self.iter_step(tensors_list)

                if torch.isnan(loss).any():
                    do_continue = False
                iter += 1
                if self.n_enc_steps == 1:
                    self.history["lr"].append(optimizer_schedule.get_lr()[0])
                    optimizer_schedule.step()
                else:
                    self.history["lr"].append(enc_schedule.get_lr()[0])
                    enc_schedule.step()
                    dec_schedule.step()  

                self.history["loss"].append(loss)
            if iter >= 10000:
                do_continue = False
        losses = np.array([ite.detach().item() for ite in self.history["loss"]])
        losses = np.nan_to_num(losses, nan=1e16)
        max_val = losses[:3].mean()
        min_val = losses.min()
        mval = 0.5 * (max_val + min_val)
        thresh = mval
        idx = np.where(losses <= thresh)[0][0]
        lr = self.history["lr"][idx]

        # mval = np.exp(np.mean(np.log(np.array([min_val, max_val]))))
        # thresh = mval
        # thresh = mval
        # idx = np.where(losses <= thresh)[0][0]
        # lr = self.history["lr"][idx]
        if lr <= 1e-5:
            logging.info("Problem with automatic LR choice, using 1e-4 as default")
            return 1e-4
        logging.info("Automatic LR choice {}".format(lr))
        return lr

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        self.compute_metrics()
        on = self.early_stopping.on
        early_stopping_metric = self.early_stopping.early_stopping_metric
        save_best_state_metric = self.early_stopping.save_best_state_metric
        if save_best_state_metric is not None and on is not None:
            if self.early_stopping.update_state(
                self.history[save_best_state_metric + "_" + on][-1]
            ):
                self.best_state_dict = self.model.state_dict()
                self.best_epoch = self.epoch

        continue_training = True
        if early_stopping_metric is not None and on is not None:
            continue_training, reduce_lr = self.early_stopping.update(
                self.history[early_stopping_metric + "_" + on][-1]
            )
            if reduce_lr:
                logger.info("Reducing LR.")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= self.early_stopping.lr_factor

        return continue_training

    @property
    @abstractmethod
    def posteriors_loop(self):
        pass

    def data_loaders_loop(
        self,
    ):  # returns an zipped iterable corresponding to loss signature
        data_loaders_loop = [self._posteriors[name] for name in self.posteriors_loop]
        return zip(
            data_loaders_loop[0],
            *[cycle(data_loader) for data_loader in data_loaders_loop[1:]]
        )

    def register_posterior(self, name, value):
        name = name.strip("_")
        self._posteriors[name] = value

    def corrupt_posteriors(
        self, rate=0.1, corruption="uniform", update_corruption=True
    ):
        if not hasattr(self.gene_dataset, "corrupted") and update_corruption:
            self.gene_dataset.corrupt(rate=rate, corruption=corruption)
        for name, posterior in self._posteriors.items():
            self.register_posterior(name, posterior.corrupted())

    def uncorrupt_posteriors(self):
        for name_, posterior in self._posteriors.items():
            self.register_posterior(name_, posterior.uncorrupted())

    def __getattr__(self, name):
        if "_posteriors" in self.__dict__:
            _posteriors = self.__dict__["_posteriors"]
            if name.strip("_") in _posteriors:
                return _posteriors[name.strip("_")]
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        if name.strip("_") in self._posteriors:
            del self._posteriors[name.strip("_")]
        else:
            object.__delattr__(self, name)

    def __setattr__(self, name, value):
        if isinstance(value, Posterior):
            name = name.strip("_")
            self.register_posterior(name, value)
        else:
            object.__setattr__(self, name, value)

    def train_test(
        self,
        model=None,
        gene_dataset=None,
        train_size=0.1,
        test_size=None,
        seed=0,
        test_indices=None,
        type_class=Posterior,
    ):
        """
        :param train_size: float, int, or None (default is 0.1)
        :param test_size: float, int, or None (default is None)
        :param model:
        :param gene_dataset:
        :param seed:
        :param test_indices:
        :param type_class:
        """
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )

        n = len(gene_dataset)
        if test_indices is None:
            n_train, n_test = _validate_shuffle_split(n, test_size, train_size)
            np.random.seed(seed=seed)
            permutation = np.random.permutation(n)
            indices_test = permutation[:n_test]
            indices_train = permutation[n_test : (n_test + n_train)]
        else:
            indices_test = np.array(test_indices)
            all_indices = np.arange(len(gene_dataset))
            indices_train = ~np.isin(all_indices, indices_test)
            indices_train = all_indices[indices_train]
            assert len(np.intersect1d(indices_train, indices_test)) == 0

        return (
            self.create_posterior(
                model, gene_dataset, indices=indices_train, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, indices=indices_test, type_class=type_class
            ),
        )

    def create_posterior(
        self,
        model=None,
        gene_dataset=None,
        shuffle=False,
        indices=None,
        type_class=Posterior,
    ):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )
        return type_class(
            model,
            gene_dataset,
            shuffle=shuffle,
            indices=indices,
            use_cuda=self.use_cuda,
            data_loader_kwargs=self.data_loader_kwargs,
        )


class SequentialSubsetSampler(SubsetRandomSampler):
    def __init__(self, indices):
        self.indices = np.sort(indices)

    def __iter__(self):
        return iter(self.indices)


class EarlyStopping:
    def __init__(
        self,
        early_stopping_metric: str = None,
        save_best_state_metric: str = None,
        on: str = "test_set",
        patience: int = 15,
        threshold: int = 3,
        benchmark: bool = False,
        reduce_lr_on_plateau: bool = False,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
    ):
        self.benchmark = benchmark
        self.patience = patience
        self.threshold = threshold
        self.epoch = 0
        self.wait = 0
        self.wait_lr = 0
        self.mode = (
            getattr(Posterior, early_stopping_metric).mode
            if early_stopping_metric is not None
            else None
        )
        # We set the best to + inf because we're dealing with a loss we want to minimize
        self.current_performance = np.inf
        self.best_performance = np.inf
        self.best_performance_state = np.inf
        # If we want to maximize, we start at - inf
        if self.mode == "max":
            self.best_performance *= -1
            self.current_performance *= -1
        self.mode_save_state = (
            getattr(Posterior, save_best_state_metric).mode
            if save_best_state_metric is not None
            else None
        )
        if self.mode_save_state == "max":
            self.best_performance_state *= -1

        self.early_stopping_metric = early_stopping_metric
        self.save_best_state_metric = save_best_state_metric
        self.on = on
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

    def update(self, scalar):
        self.epoch += 1
        if self.benchmark:
            continue_training = True
            reduce_lr = False
        elif self.wait >= self.patience:
            continue_training = False
            reduce_lr = False
        else:
            # Check if we should reduce the learning rate
            if not self.reduce_lr_on_plateau:
                reduce_lr = False
            elif self.wait_lr >= self.lr_patience:
                reduce_lr = True
                self.wait_lr = 0
            else:
                reduce_lr = False
            # Shift
            self.current_performance = scalar

            # Compute improvement
            if self.mode == "max":
                improvement = self.current_performance - self.best_performance
            elif self.mode == "min":
                improvement = self.best_performance - self.current_performance
            else:
                raise NotImplementedError("Unknown optimization mode")

            # updating best performance
            if improvement > 0:
                self.best_performance = self.current_performance

            if improvement < self.threshold:
                self.wait += 1
                self.wait_lr += 1
            else:
                self.wait = 0
                self.wait_lr = 0

            continue_training = True
        if not continue_training:
            # FIXME: log total number of epochs run
            logger.info(
                "\nStopping early: no improvement of more than "
                + str(self.threshold)
                + " nats in "
                + str(self.patience)
                + " epochs"
            )
            logger.info(
                "If the early stopping criterion is too strong, "
                "please instantiate it with different parameters in the train method."
            )
        return continue_training, reduce_lr

    def update_state(self, scalar):
        improved = (
            self.mode_save_state == "max" and scalar - self.best_performance_state > 0
        ) or (
            self.mode_save_state == "min" and self.best_performance_state - scalar > 0
        )
        if improved:
            self.best_performance_state = scalar
        return improved
