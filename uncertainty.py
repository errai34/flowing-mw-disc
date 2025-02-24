#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uncertainty-aware training components for normalizing flow models.
Properly incorporates measurement uncertainties into the model training.
"""

import torch


def compute_log_noise_pdf(w, x, e):
    """
    Given a measurement model w = x + noise, where noise ~ N(0, diag(e²)),
    compute the log probability log p_noise(w | x) in a numerically stable way.

    Parameters:
    -----------
    w : torch.Tensor
        Observed data points (N, D)
    x : torch.Tensor
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

    # The noise model is: p(w|x) = N(w; x, e_safe^2)
    # Compute log probability in a numerically stable way
    log_prob = -0.5 * torch.sum(
        torch.log(2 * torch.pi * e_safe.pow(2)) + ((w - x) / e_safe).pow(2), dim=-1
    )

    return log_prob


def uncertainty_aware_elbo(flow, observed_data, uncertainties, K=10):
    """
    Compute uncertainty-aware ELBO for deconvolution.
    ELBO = E_q(x|w)[log p(w|x) + log p(x) - log q(x|w)]

    Parameters:
    -----------
    flow : Flow5D or Flow3D
        Normalizing flow model from flow_model module
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

    # Sample from perturbed normal distribution around observed data
    # This approximates the posterior q(x|w)
    input_dim = observed_data.shape[1]  # Get actual input dimension instead of hardcoding 5
    eps = torch.randn(K, batch_size, input_dim, device=observed_data.device)
    perturbed_data = observed_data.unsqueeze(0) + eps * uncertainties.unsqueeze(0)
    perturbed_data = perturbed_data.reshape(K * batch_size, -1)

    # Compute log probabilities
    log_p_x = flow.log_prob(perturbed_data)
    log_p_w_given_x = compute_log_noise_pdf(
        observed_data.repeat(K, 1), perturbed_data, uncertainties.repeat(K, 1)
    )

    # Compute log probability of the approximate posterior
    log_q_x_given_w = -0.5 * torch.sum(
        torch.log(2 * torch.pi * uncertainties.pow(2).repeat(K, 1))
        + eps.reshape(K * batch_size, -1).pow(2),
        dim=-1,
    )

    # Compute ELBO
    elbo_samples = log_p_w_given_x + log_p_x - log_q_x_given_w
    elbo_samples = elbo_samples.reshape(K, batch_size)

    # Average over Monte Carlo samples
    elbo = torch.mean(elbo_samples, dim=0)

    return torch.mean(elbo)
