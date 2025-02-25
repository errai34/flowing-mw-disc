#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uncertainty-aware training components for normalizing flow models.
Fixed version with proper tensor type handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation


class RecognitionNetwork(nn.Module):
    """
    Recognition network for amortized variational inference.
    Implements q(v|w) as an Inverse Autoregressive Flow (IAF).

    Parameters:
    -----------
    input_dim : int
        Dimension of the data
    n_transforms : int
        Number of transforms in the flow
    hidden_dims : list
        Dimensions of hidden layers
    """

    def __init__(self, input_dim=5, n_transforms=8, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Base distribution (standard normal)
        self.base_dist = StandardNormal(shape=[input_dim])

        # Create conditioning network
        self.conditioning_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dims[0]),  # observed data + error info
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        # Create transforms - use IAF for efficient sampling
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=input_dim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=input_dim,
                    hidden_features=hidden_dims[0],
                    context_features=hidden_dims[
                        0
                    ],  # context from conditioning network
                    num_bins=16,
                    tails="linear",
                    tail_bound=5.0,
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=F.relu,
                )
            )

        self.transform = CompositeTransform(transforms)

    def forward(self, observed_data, uncertainties):
        """
        Process observed data and uncertainties to create context.

        Parameters:
        -----------
        observed_data : torch.Tensor
            Observed data (batch_size, input_dim)
        uncertainties : torch.Tensor
            Uncertainties for observed data (batch_size, input_dim)

        Returns:
        --------
        torch.Tensor
            Context for the normalizing flow (batch_size, hidden_dim)
        """
        # Concatenate observed data and uncertainties
        context_input = torch.cat([observed_data, uncertainties], dim=1)
        return self.conditioning_net(context_input)

    def sample(self, observed_data, uncertainties, n_samples=1):
        """
        Sample from q(v|w).

        Parameters:
        -----------
        observed_data : torch.Tensor
            Observed data points (batch_size, input_dim)
        uncertainties : torch.Tensor
            Measurement uncertainties (batch_size, input_dim)
        n_samples : int
            Number of samples to draw per observed data point

        Returns:
        --------
        torch.Tensor
            Samples from q(v|w) (batch_size * n_samples, input_dim)
        """
        batch_size = observed_data.shape[0]
        input_dim = observed_data.shape[1]

        # Ensure n_samples is an integer
        n_samples = int(n_samples)

        # Get context from conditioning network
        context = self.forward(observed_data, uncertainties)

        # Draw samples from base distribution
        eps = self.base_dist.sample(batch_size * n_samples).reshape(
            batch_size * n_samples, -1
        )

        # Create batched context by repeating for each sample
        batched_context = context.repeat_interleave(n_samples, dim=0)

        # Transform samples using context
        samples, _ = self.transform.inverse(eps, batched_context)

        return samples.reshape(batch_size * n_samples, -1)

    def log_prob(self, latent_samples, observed_data, uncertainties):
        """
        Compute log q(v|w).

        Parameters:
        -----------
        latent_samples : torch.Tensor
            Latent variable samples (batch_size, input_dim)
        observed_data : torch.Tensor
            Observed data points (batch_size, input_dim)
        uncertainties : torch.Tensor
            Measurement uncertainties (batch_size, input_dim)

        Returns:
        --------
        torch.Tensor
            Log probability of samples under q(v|w) (batch_size,)
        """
        # Get context from conditioning network
        context = self.forward(observed_data, uncertainties)

        # Transform samples to base space
        noise, logabsdet = self.transform(latent_samples, context)

        # Compute log probability
        log_prob = self.base_dist.log_prob(noise) + logabsdet

        return log_prob


def compute_log_noise_pdf(w, v, e):
    """
    Given a measurement model w = v + noise, where noise ~ N(0, diag(e²)),
    compute the log probability log p_noise(w | v) in a numerically stable way.

    Parameters:
    -----------
    w : torch.Tensor
        Observed data points (N, D)
    v : torch.Tensor
        True latent data points (N, D)
    e : torch.Tensor
        Measurement uncertainties (N, D)

    Returns:
    --------
    torch.Tensor
        Log probability of the noise model (N,)
    """
    # Clamp the error so that the variance never vanishes
    e_safe = e.clamp(min=1e-8)

    # The noise model is: p(w|v) = N(w; v, e_safe^2)
    # Compute log probability in a numerically stable way
    log_prob = -0.5 * torch.sum(
        torch.log(2 * torch.pi * e_safe.pow(2)) + ((w - v) / e_safe).pow(2), dim=-1
    )

    return log_prob


def uncertainty_aware_elbo(flow, recognition_net, observed_data, uncertainties, K=10):
    """
    Compute uncertainty-aware ELBO for deconvolution using a dedicated recognition network.
    ELBO = E_q(v|w)[log p(w|v) + log p(v) - log q(v|w)]

    Parameters:
    -----------
    flow : Flow5D
        Normalizing flow model for the prior p(v)
    recognition_net : RecognitionNetwork
        Recognition network for approximate posterior q(v|w)
    observed_data : torch.Tensor
        Observed data points (batch_size, D)
    uncertainties : torch.Tensor
        Measurement uncertainties (batch_size, D)
    K : int
        Number of Monte Carlo samples

    Returns:
    --------
    torch.Tensor
        ELBO value (scalar)
    """
    batch_size = observed_data.shape[0]

    # Ensure K is an integer
    K = int(K)

    # Sample from recognition network q(v|w)
    samples = recognition_net.sample(observed_data, uncertainties, n_samples=K)

    # Repeat observed data and uncertainties for each sample
    repeated_observed = observed_data.repeat_interleave(K, dim=0)
    repeated_uncertainties = uncertainties.repeat_interleave(K, dim=0)

    # Compute log probabilities
    log_p_v = flow.log_prob(samples)  # Prior
    log_p_w_given_v = compute_log_noise_pdf(
        repeated_observed, samples, repeated_uncertainties
    )  # Likelihood
    log_q_v_given_w = recognition_net.log_prob(
        samples, repeated_observed, repeated_uncertainties
    )  # Posterior

    # Compute ELBO for each sample
    elbo_components = log_p_w_given_v + log_p_v - log_q_v_given_w
    elbo_components = elbo_components.reshape(batch_size, K)

    # Average over MC samples
    elbo = torch.mean(elbo_components, dim=1)

    return torch.mean(elbo)


def importance_weighted_elbo(flow, recognition_net, observed_data, uncertainties, K=50):
    """
    Compute Importance Weighted ELBO (IWAE) for tighter bound.
    IW-ELBO = E[ log(1/K sum_{k=1}^K [p(w|v_k)p(v_k)/q(v_k|w)]) ]

    Parameters:
    -----------
    flow : Flow5D
        Normalizing flow model for the prior p(v)
    recognition_net : RecognitionNetwork
        Recognition network for approximate posterior q(v|w)
    observed_data : torch.Tensor
        Observed data points (batch_size, D)
    uncertainties : torch.Tensor
        Measurement uncertainties (batch_size, D)
    K : int
        Number of importance samples

    Returns:
    --------
    torch.Tensor
        IW-ELBO value (scalar)
    """
    batch_size = observed_data.shape[0]

    # Ensure K is an integer
    K = int(K)

    # Sample from recognition network q(v|w)
    samples = recognition_net.sample(observed_data, uncertainties, n_samples=K)

    # Repeat observed data and uncertainties for each sample
    repeated_observed = observed_data.repeat_interleave(K, dim=0)
    repeated_uncertainties = uncertainties.repeat_interleave(K, dim=0)

    # Compute log probabilities
    log_p_v = flow.log_prob(samples)  # Prior
    log_p_w_given_v = compute_log_noise_pdf(
        repeated_observed, samples, repeated_uncertainties
    )  # Likelihood
    log_q_v_given_w = recognition_net.log_prob(
        samples, repeated_observed, repeated_uncertainties
    )  # Posterior

    # Compute log weights: log p(w|v) + log p(v) - log q(v|w)
    log_weights = log_p_w_given_v + log_p_v - log_q_v_given_w
    log_weights = log_weights.reshape(batch_size, K)

    # Compute IW-ELBO: log(1/K sum_k exp(log_weights_k))
    iw_elbo = torch.logsumexp(log_weights, dim=1) - torch.log(
        torch.tensor(K, dtype=torch.float32, device=observed_data.device)
    )

    return torch.mean(iw_elbo)


def compute_diagnostics(
    flow, recognition_net, observed_data, uncertainties, n_samples=50
):
    """
    Compute diagnostic metrics to monitor training progress.

    Parameters:
    -----------
    flow : Flow5D
        Normalizing flow model for the prior p(v)
    recognition_net : RecognitionNetwork
        Recognition network for approximate posterior q(v|w)
    observed_data : torch.Tensor
        Observed data points (batch_size, D)
    uncertainties : torch.Tensor
        Measurement uncertainties (batch_size, D)
    n_samples : int
        Number of samples for Monte Carlo estimation

    Returns:
    --------
    dict
        Dictionary of diagnostic metrics
    """
    batch_size = observed_data.shape[0]

    # Ensure n_samples is an integer
    n_samples = int(n_samples)

    # Sample from recognition network q(v|w)
    samples = recognition_net.sample(observed_data, uncertainties, n_samples=n_samples)

    # Repeat observed data and uncertainties for each sample
    repeated_observed = observed_data.repeat_interleave(n_samples, dim=0)
    repeated_uncertainties = uncertainties.repeat_interleave(n_samples, dim=0)

    # Compute log probabilities
    log_p_v = flow.log_prob(samples)  # Prior
    log_p_w_given_v = compute_log_noise_pdf(
        repeated_observed, samples, repeated_uncertainties
    )  # Likelihood
    log_q_v_given_w = recognition_net.log_prob(
        samples, repeated_observed, repeated_uncertainties
    )  # Posterior

    # Reshape
    log_p_v = log_p_v.reshape(batch_size, n_samples)
    log_p_w_given_v = log_p_w_given_v.reshape(batch_size, n_samples)
    log_q_v_given_w = log_q_v_given_w.reshape(batch_size, n_samples)

    # Compute ELBO components
    reconstruction_term = torch.mean(log_p_w_given_v, dim=1).mean()
    kl_term = torch.mean(log_q_v_given_w - log_p_v, dim=1).mean()

    # Compute log weights for importance sampling
    log_weights = log_p_w_given_v + log_p_v - log_q_v_given_w

    # Compute effective sample size
    normalized_weights = F.softmax(log_weights, dim=1)
    ess = 1.0 / torch.sum(normalized_weights**2, dim=1)
    avg_ess = ess.mean() / n_samples

    return {
        "reconstruction_term": reconstruction_term.item(),
        "kl_term": kl_term.item(),
        "effective_sample_size": avg_ess.item(),
    }
