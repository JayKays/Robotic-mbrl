
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

from mbrl.models.model import Ensemble
from .bayesian_utils import BayesianLinearEnsembleLayer

class BNN(Ensemble):
    """
    Implements a linear Bayesian Ensemble
    Som of the functionality is re-purposed from the GaussianMLP in mbrl-lib:
    https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/models/gaussian_mlp.py
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        prior_sigma1: float = 1,
        prior_sigma2: float = 0.1,
        prior_pi: float = 0.8,
        propagation_method: Optional[str] = None,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )

        self.in_size = in_size
        self.out_size = out_size
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_pi = prior_pi

        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        def create_linear_layer(l_in, l_out):
            return BayesianLinearEnsembleLayer(
                                ensemble_size, 
                                l_in, 
                                l_out,
                                prior_pi=self.prior_pi, 
                                prior_sigma_1=self.prior_sigma1, 
                                prior_sigma_2=self.prior_sigma2
                            )

        hidden_layers = [
            nn.Sequential(create_linear_layer(in_size, hid_size), create_activation())
        ]

        for _ in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = create_linear_layer(hid_size, out_size)

        self.freeze = deterministic
        if self.freeze: self.freeze_model()
        
        #Num_batches must be updated externally before training to use KL reweighting for Mini-Batch Optimization
        self.num_batches = None
        self.batch_idx  = 0

        self.to(self.device)
        

        self.elite_models: List[int] = None


    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.output_layer.set_elite(self.elite_models)
            self.output_layer.toggle_use_only_elite()

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)

        if self.freeze:
            self.freeze_model()

        x = self.hidden_layers(x)
        output = self.output_layer(x)

        self._maybe_toggle_layers_use_only_elite(only_elite)

        return output

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        pred = self._default_forward(shuffled_x, only_elite=True)
        # note that pred is shuffled
        pred = pred.view(batch_size, -1)
        pred[model_shuffle_indices] = pred.clone()  # invert the shuffle

        return pred

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
            return mean
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            pred = self._default_forward(x, only_elite=True)
            return pred.mean(dim=0)

        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predictions for the given input.
        """
        
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor):
        assert model_in.ndim == target.ndim
        
        pred_mean = self.forward(model_in, use_propagation=False)
        return F.nll_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the total loss of the Network,
        If model is frozen, the mse loss is computed
        If not frozen, the estimated ELBO loss is computed
        
        This function returns no metadata, so the second output is set to an empty dict.
        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Od``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.
        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        if self.freeze:
            self.freeze_model()
            loss = self._mse_loss(model_in, target)
        else:
            loss = self.sample_elbo(model_in, target)
        
        return loss, {}

    def nn_kl_divergence(self):
        """Returns the sum of the KL divergence of each of the BayesianModules of the model, which are from
            their posterior current distribution of weights relative to a scale-mixtured prior (and simpler) distribution of weights
            Parameters:
                N/a
            Returns torch.tensor with 0 dim.      
        
        """
        kl_divergence = 0

        for module in self.modules():
            if isinstance(module, (BayesianLinearEnsembleLayer)):
                kl_divergence += module.log_variational_posterior - module.log_prior

        return kl_divergence

    def get_complexity_cost(self):
        '''
        Calculates the complexity cost for the current minibatch,
        if num_batches is set before training.
        '''
        if self.num_batches is None:
            return 1
        else:
            return 2**(self.num_batches - self.batch_idx)/(2**self.num_batches -1)

    def gaussian_nll(self, pred_mean, pred_var, targets):
        '''
        Computes the guassian nll of predictions
        '''
        l2 = F.mse_loss(pred_mean, targets)
        inv_var = 1/(pred_var + 1e-12)
        loss = l2 * inv_var + torch.log(pred_var)

        return loss.sum(dim = 1).mean()

    def sample_elbo(self,
                    inputs,
                    targets,
                    sample_nbr = 5,
                    complexity_cost_weight=1):

        """ Samples the ELBO Loss for a batch of data, consisting of inputs and corresponding-by-index labels
                The ELBO Loss consists of the sum of the KL Divergence of the model
                 (explained above, interpreted as a "complexity part" of the loss)
                 with the actual criterion - (loss function) of optimization of our model
                 (the performance part of the loss).
                As we are using variational inference, it takes several (quantified by the parameter sample_nbr) Monte-Carlo
                 samples of the weights in order to gather a better approximation for the loss.
            Parameters:
                inputs(torch.tensor) -> the input data to the model
                labels(torch.tensor) -> label data for the performance-part of the loss calculation
                        The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
                sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to
                            gather the loss to be .backwarded in the optimization of the model.
            Returns:
                (tensor): Estimated ELBO loss for the model
        """

        if self.num_batches is not None:
            complexity_cost_weight = self.get_complexity_cost()
            self.batch_idx += 1
            self.batch_idx = self.batch_idx % self.num_batches

        loss = 0
        complexity_cost_weight *= 1/inputs.shape[-2]
        for _ in range(sample_nbr):
            loss += self._mse_loss(inputs, targets)
            loss += self.nn_kl_divergence() * complexity_cost_weight

        loss/= sample_nbr

        return loss
    
    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.
        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.
        This function returns no metadata, so the second output is set to an empty dict.
        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.
        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            self.freeze_model()
            pred = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            self.unfreeze_model()
            return F.mse_loss(pred, target, reduction="none"), {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                f"To use ensemble propagation, the batch size [{batch_size}] must "
                f"be a multiple of the number of models [{model_len}] in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Samples an output from the model using .

        This method will be used by :class:`ModelEnv` to simulate a transition of the form.
            outputs_t+1, s_t+1 = sample(model_input_t, s_t), where

            - model_input_t: observation and action at time t, concatenated across axis=1.
            - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
            - outputs_t+1: observation and reward at time t+1, concatenated across axis=1.

        The default implementation returns `s_t+1=s_t`.

        Args:
            model_input (tensor): the observation and action at.
            model_state (tensor): the model state st. Must contain a key
                "propagation_indices" to use for uncertainty propagation.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary. Everything but the observation is optional, and can
                be returned with value ``None``.
        """
        if deterministic or self.freeze:
            self.freeze_model()
            pred = self.forward(
                model_input, rng=rng, propagation_indices=model_state["propagation_indices"]
            )
            self.unfreeze_model()
            return pred, model_state
        assert rng is not None
        pred = self.forward(
            model_input, rng=rng, propagation_indices=model_state["propagation_indices"]
        )
        return pred, model_state


    def set_batch_count(self, num_batches):
        self.num_batches = num_batches
        self.batch_idx = 0

    def set_elite(self, elite_indices: Sequence[int]):
        #Elite models not supported by bayesian linear layer currently, 
        #So this will allways keep elite_models = None
        if self.elite_models is None: return
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def freeze_model(self):
        """
        Freezes the model by making predictions only using the expected values of the weight distributions
        """
        for module in self.modules():
            if isinstance(module, (BayesianLinearEnsembleLayer)):
                module.freeze()

        self.freeze = True
    
    def unfreeze_model(self):
        """
        Unfreezes the model by letting it draw its weights with uncertanity from their corresponding distributions
        """
        for module in self.modules():
            if isinstance(module, (BayesianLinearEnsembleLayer)):
                module.unfreeze()

        self.freeze = False

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
        # self.elite_models = model_dict["elite_models"]
        self.elite_models = None