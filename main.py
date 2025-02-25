#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for training and analyzing normalizing flow models of Galactic evolution.
Implements the analysis pipeline with improved density deconvolution based on the paper.
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")

from flow_model import Flow5D  # noqa: E402
from uncertainty import (  # noqa: E402
    RecognitionNetwork,
    uncertainty_aware_elbo,
)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain_flow(data, n_epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-5):
    """
    Pretrain flow model using maximum likelihood on data.

    Parameters:
    -----------
    data : np.ndarray
        Data array of shape (N, D)
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer

    Returns:
    --------
    tuple
        (trained flow model, scaler, training stats)
    """
    # Ensure n_epochs is an integer
    n_epochs = int(n_epochs)

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize flow model with enhanced configuration
    flow = Flow5D(
        n_transforms=12,  # Reduced from 16
        hidden_dims=[128, 128],  # Reduced from [256, 256]
        num_bins=24,  # Reduced from 32
        use_residual_blocks=True,
        dropout_probability=0.0,  # No dropout during pretraining
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    train_stats = {"log_likelihood": [], "time": []}
    start_time = time.time()

    print(f"Pretraining flow on {len(data)} data points for {n_epochs} epochs")
    for epoch in range(n_epochs):
        epoch_start = time.time()
        flow.train()
        epoch_lls = []

        # Progress bar
        batch_progress = tqdm(
            loader,
            desc=f"Pretraining Epoch {epoch+1}/{n_epochs}",
            leave=False,
            ncols=100,
        )

        for (batch_data,) in batch_progress:
            optimizer.zero_grad()

            # Compute log likelihood
            log_likelihood = flow.log_prob(batch_data)
            loss = -torch.mean(log_likelihood)

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            optimizer.step()

            epoch_lls.append(-loss.item())
            batch_progress.set_postfix({"LL": f"{-loss.item():.4f}"})

        # Track stats
        avg_ll = np.mean(epoch_lls)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        train_stats["log_likelihood"].append(avg_ll)
        train_stats["time"].append(total_time)

        print(
            f"Pretraining Epoch {epoch+1}/{n_epochs}, Log-Likelihood: {avg_ll:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

    print(
        f"Pretraining complete. Final Log-Likelihood: {train_stats['log_likelihood'][-1]:.4f}"
    )
    return flow, scaler, train_stats


def train_flow_model(
    data,
    errors,
    n_transforms=12,
    hidden_dims=None,
    n_epochs=50,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    mc_samples=20,
    use_importance_weighted=False,
    output_dir=None,
    pretrain_epochs=10,
):
    """
    Minimalist version of the flow model training function.
    """
    print("Starting flow model training...")

    # Ensure integer parameters
    n_epochs = int(n_epochs)
    mc_samples = int(mc_samples)
    pretrain_epochs = int(pretrain_epochs)

    # Use default hidden dimensions if not provided
    if hidden_dims is None:
        hidden_dims = [128, 128]

    # Get data dimensions
    input_dim = data.shape[1]

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    errors_scaled = errors / scaler.scale_

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    errors_tensor = torch.tensor(errors_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor, errors_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Stage 1: Pretrain the flow model
    print(f"\n=== Stage 1: Pretraining flow model for {pretrain_epochs} epochs ===")
    flow, _, _ = pretrain_flow(
        data=data,
        n_epochs=pretrain_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Initialize recognition network
    recognition_net = RecognitionNetwork(
        input_dim=input_dim, n_transforms=8, hidden_dims=hidden_dims
    ).to(device)

    # Stage 2: Train with uncertainty-aware ELBO
    print(
        f"\n=== Stage 2: Training with uncertainty-aware ELBO for {n_epochs} epochs ==="
    )

    # Optimizer
    optimizer = optim.AdamW(
        list(flow.parameters()) + list(recognition_net.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 ** (epoch / 20)  # Halve the learning rate every 20 epochs

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Training stats
    train_stats = {
        "elbo": [],
        "lr": [],
    }

    best_elbo = -float("inf")
    best_state = {"flow": None, "recognition": None}

    print(f"Training on {len(data)} data points for {n_epochs} epochs")

    # Main training loop
    for epoch in range(n_epochs):
        flow.train()
        recognition_net.train()

        epoch_elbo_values = []

        # Batch loop with progress bar
        batch_progress = tqdm(
            loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False, ncols=100
        )

        for batch_data, batch_errors in batch_progress:
            optimizer.zero_grad()

            # Calculate objective
            elbo = uncertainty_aware_elbo(
                flow, recognition_net, batch_data, batch_errors, K=mc_samples
            )
            batch_elbo = elbo.item()
            epoch_elbo_values.append(batch_elbo)

            # Backward pass
            loss = -elbo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(flow.parameters()) + list(recognition_net.parameters()), 10.0
            )
            optimizer.step()

            # Update progress bar
            batch_progress.set_postfix({"ELBO": f"{batch_elbo:.4f}"})

        # Update learning rate
        scheduler.step()

        # Record epoch stats
        epoch_avg_elbo = (
            sum(epoch_elbo_values) / len(epoch_elbo_values) if epoch_elbo_values else 0
        )
        train_stats["elbo"].append(epoch_avg_elbo)
        train_stats["lr"].append(optimizer.param_groups[0]["lr"])

        # Save best model
        if epoch_avg_elbo > best_elbo:
            best_elbo = epoch_avg_elbo
            best_state = {
                "flow": {k: v.cpu().clone() for k, v in flow.state_dict().items()},
                "recognition": {
                    k: v.cpu().clone() for k, v in recognition_net.state_dict().items()
                },
            }
            print(f"New best objective: {best_elbo:.4f}")

        # Print status
        print(
            f"Epoch {epoch+1}/{n_epochs}, ELBO: {epoch_avg_elbo:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Create simple visualization every 5 epochs
        if epoch % 5 == 0 and output_dir:
            bin_name = (
                os.path.basename(output_dir).split("_")[0] if output_dir else "unknown"
            )
            plot_path = os.path.join(plots_dir, f"training_progress_{bin_name}.png")

            plt.figure(figsize=(10, 5))
            plt.plot(range(epoch + 1), train_stats["elbo"], "b-", marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("ELBO")
            plt.title(f"Training Progress for {bin_name}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            print(f"Training progress plot saved with {epoch+1} points")

    # Load best model
    flow.load_state_dict(best_state["flow"])
    recognition_net.load_state_dict(best_state["recognition"])
    flow.to(device)
    recognition_net.to(device)

    print(f"Training complete. Best objective: {best_elbo:.4f}")
    return flow, recognition_net, scaler, train_stats
