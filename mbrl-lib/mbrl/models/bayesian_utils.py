
from typing import List, Sequence, Tuple, Union, Optional

import numpy as np
import torch
from torch import nn as nn

class BayesianLinearEnsembleLayer(nn.Module):
    """
    Bayesian Linear Ensemble layer, implements a linear layer as proposed in the Bayes by Backprop paper. 
    This is also capable of acting as an efficient linear layer in a bayesian network ensemble, given num_members > 1.
    
    parameters:
        num_members(int): size of the network ensemble
        in_fetaures(int): number  of input features for the layer
        out_features(int): number of output features for the layer
        bias(bool): whether the layer should use bias or not, defaults to True
        prior_sigma_1 (float): sigma1 of the  gaussian mixture prior distribution
        prior_sigma_2 (float): sigma2 of the gaussian mixture prior distribution
        prior_pi(float): scaling factor of the guassian mixture prior (must be between 0-1)
        posterior_mu_init (float): mean of the mu parameter init
        posterior_rho_init (float): mean of the rho parameter init
        freeze(bool): wheter the model will start with frozen(deterministic) weights, or not
        prior_dist: A potential prior distribution of weghts
        truncated_init(bool): Wether to use a truncated normal distribution for parameter init or not
        moped (bool): Wether or not to use the moped initialization of weight std (rho = delta*|w|)
        delta(float): The delta value in moped init, not used if moped = False 
    """
    def __init__(self,
                 num_members: int,
                 in_features: int,
                 out_features: int,
                 bias: bool =True,
                 prior_sigma_1: float = 0.9,
                 prior_sigma_2: float = 0.001,
                 prior_pi: float = 0.7,
                 posterior_mu_init: Union[float, int] = 0,
                 posterior_rho_init: Union[float, int] = -7.0,
                 freeze: bool = False,
                 prior_dist: Optional[torch.distributions.distribution.Distribution] = None,
                 truncated_init: bool = True
                 ):

        super().__init__()

        #main layer parameters
        self.num_members = num_members
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.freeze = freeze

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.elite_models: List[int] = None
        self.use_only_elite = False

        #parameters for the scale mixture gaussian prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        #Parameters of truncated normal distribution for parameter initialization
        init_std = 1 / (2*np.sqrt(self.in_features))
        init_std = 0.1
        mu_init_max, mu_init_min = posterior_mu_init + 2*init_std, posterior_mu_init-2*init_std

        # Variational parameters and sampler for weights and biases
        if truncated_init:
            self.weight_mu = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_mu_init, init_std).clamp(mu_init_min, mu_init_max))
            self.weight_rho = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_rho_init, init_std))
            if self.use_bias:
                self.bias_mu = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_mu_init, init_std).clamp(mu_init_min, mu_init_max))
                self.bias_rho = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_rho_init, init_std))
        else:
            self.weight_mu = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_mu_init, init_std))
            self.weight_rho = nn.Parameter(torch.Tensor(num_members, in_features, out_features).normal_(posterior_rho_init, init_std))
            if self.use_bias:
                self.bias_mu = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_mu_init, init_std))
                self.bias_rho = nn.Parameter(torch.Tensor(num_members, 1, out_features).normal_(posterior_rho_init, init_std))


        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)
        
        if self.use_bias:
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
            self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)

        # Prior distributions
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        '''
        Computes the forward pass of the Bayesian layer by sampling weights and biases
        and computes the output y = W*x + b with the sampled paramters.

        returns:
            torch.tensor with shape [num_members, out_features]
        '''
        
        #if the model is frozen, return deterministic forward pass
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()


        if self.use_bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros(self.bias_sampler.mu.shape)
            b_log_posterior = 0
            b_log_prior = 0

        if self.use_only_elite:
            w = w[self.elite_models,...]
            b = b[self.elite_models,...]

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        #Calculate forward pass output whith sampled weights (and biases)
        xw = x.matmul(w) + b

        return xw

    def forward_frozen(self, x):
        '''
        Computes the deterministic forward pass using only the means of the weight distributions
        
        returns:
            torch.tensor with shape [num_members, out_features]
        '''
        if self.use_only_elite:
            xw = x.matmul(self.weight_sampler.mu[self.elite_models,...])

            if self.use_bias:
                xw += self.bias_sampler.mu[self.elite_models,...]

        else:
            xw = x.matmul(self.weight_sampler.mu)

            if self.use_bias:
                xw += self.bias_sampler.mu
        
        return xw
    
    def freeze(self):
        self.freeze = True
    
    def unfreeze(self):
        self.freeze = False

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite

class TrainableRandomDistribution(nn.Module):
    '''
    Samples weights for variational inference as in Weights Uncertainity on Neural Networks (Bayes by backprop paper)
    Calculates the variational posterior part of the complexity part of the loss
    '''

    def __init__(self, mu: Union[float, int], rho: Union[float, int]):
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is 
        a function from a trainable parameter, and adding a mean
        sets those weights as the current ones
        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w

    def sample_elites(self, elite_models):

        self.eps_w.data.normal_()
        self.sigma[elite_models,...] = torch.log1p(torch.exp(self.rho[elite_models,...]))
        self.w[elite_models,...] = self.mu[elite_models,...] + self.sigma[elite_models,...] * self.eps_w[elite_models,...]
        
        return self.w[elite_models,...]

    
    def log_posterior(self, w = None):

        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost
        returns:
            torch.tensor with shape []
        """

        assert (self.w is not None), "You can only have a log posterior for W if you've already sampled it"
        if w is None:
            w = self.w
        
        log_sqrt2pi = np.log(np.sqrt(2*self.pi))
        log_posteriors =  -log_sqrt2pi - torch.log(self.sigma) - (((w - self.mu) ** 2)/(2 * self.sigma ** 2)) - 0.5
        return log_posteriors.sum()

class PriorWeightDistribution(nn.Module):
    '''
    Calculates a Scale Mixture Prior distribution for the prior 
    part of the complexity cost on Bayes by Backprop paper
    '''

    def __init__(self,
                 pi: Union[float, int] = 1,
                 sigma1: Union[float, int] = 0.1,
                 sigma2: Union[float, int] = 0.001,
                 dist: torch.distributions.distribution.Distribution = None):
        super().__init__()


        if (dist is None):
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0, sigma1)
            self.dist2 = torch.distributions.Normal(0, sigma2)

        if (dist is not None):
            self.pi = 1
            self.dist1 = dist
            self.dist2 = None

        

    def log_prior(self, w):
        """
        Calculates the log_likelihood for each of the weights sampled relative to a prior distribution as a part of the complexity cost
        returns:
            torch.tensor with shape []
        """
        prob_n1 = torch.exp(self.dist1.log_prob(w))

        if self.dist2 is not None:
            prob_n2 = torch.exp(self.dist2.log_prob(w))
        if self.dist2 is None:
            prob_n2 = 0
        
        # Prior of the mixture distribution, adding 1e-6 prevents numeric problems with log(p) for small p
        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2) + 1e-6

        return (torch.log(prior_pdf) - 0.5).sum()