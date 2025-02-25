#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for training and analyzing normalizing flow models of Galactic evolution.
Implements the analysis pipeline with improved density deconvolution based on the paper.
"""

import os
import time
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")

from flow_model import Flow5D
from uncertainty import (
    RecognitionNetwork,
    compute_diagnostics,
    importance_weighted_elbo,
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
    pretrain_epochs=10,  # Added parameter
):
    """
    Train a normalizing flow model on data with uncertainty-aware training.
    Uses the improved implementation with a dedicated recognition network.

    Parameters:
    -----------
    data : np.ndarray
        Data array of shape (N, D)
    errors : np.ndarray
        Error array of shape (N, D)
    n_transforms : int
        Number of transforms in the flow
    hidden_dims : list
        Dimensions of hidden layers
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    mc_samples : int
        Number of Monte Carlo samples for ELBO estimation
    use_importance_weighted : bool
        Whether to use importance weighted ELBO
    output_dir : str
        Directory to save progress plots
    pretrain_epochs : int
        Number of epochs for pretraining

    Returns:
    --------
    tuple
        (trained flow model, recognition network, scaler, training stats)
    """
    print(
        "Starting density deconvolution training with dedicated recognition network..."
    )

    # Ensure these are integers
    n_epochs = int(n_epochs)
    mc_samples = int(mc_samples)
    pretrain_epochs = int(pretrain_epochs)

    # Use default hidden dimensions if not provided
    if hidden_dims is None:
        hidden_dims = [128, 128]  # Reduced from [256, 256]

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
        n_epochs=pretrain_epochs,  # Use parameter here
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

    # Optimizer for both models
    optimizer = optim.AdamW(
        list(flow.parameters()) + list(recognition_net.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5  # Reduced from 10
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 ** (epoch / 20)  # Halve the learning rate every 20 epochs

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create plots directory within the run directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Training loop
    train_stats = {
        "elbo": [],
        "iwae": [],
        "reconstruction": [],
        "kl": [],
        "ess": [],
        "lr": [],
        "time": [],
    }

    best_elbo = -float("inf")
    best_state = {"flow": None, "recognition": None}
    start_time = time.time()

    print(f"Training on {len(data)} data points for {n_epochs} epochs")
    for epoch in range(n_epochs):
        epoch_start = time.time()
        flow.train()
        recognition_net.train()

        epoch_metrics = defaultdict(list)

        # Create progress bar for batches
        batch_progress = tqdm(
            loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False, ncols=100
        )

        for batch_data, batch_errors in batch_progress:
            optimizer.zero_grad()

            # Compute ELBO or IWAE
            if use_importance_weighted and epoch >= 10:  # Start with regular ELBO
                elbo = importance_weighted_elbo(
                    flow, recognition_net, batch_data, batch_errors, K=mc_samples
                )
                epoch_metrics["iwae"].append(elbo.item())
            else:
                elbo = uncertainty_aware_elbo(
                    flow, recognition_net, batch_data, batch_errors, K=mc_samples
                )
                epoch_metrics["elbo"].append(elbo.item())

            loss = -elbo

            # Compute diagnostics occasionally
            if np.random.rand() < 0.1:  # 10% of batches
                diagnostics = compute_diagnostics(
                    flow,
                    recognition_net,
                    batch_data,
                    batch_errors,
                    n_samples=mc_samples,
                )
                for k, v in diagnostics.items():
                    epoch_metrics[k].append(v)

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(flow.parameters()) + list(recognition_net.parameters()), 10.0
            )
            optimizer.step()

            # Update progress bar
            objective_val = elbo.item()
            batch_progress.set_postfix({"ELBO": f"{objective_val:.4f}"})

        # Update learning rate
        scheduler.step()

        # Compute average metrics
        train_metrics = {}
        for k, v in epoch_metrics.items():
            if v:  # Only include metrics that have values
                train_metrics[k] = np.mean(v)

        # Track stats
        for k, v in train_metrics.items():
            if k in train_stats:
                train_stats[k].append(v)

        # Fill in missing metrics to maintain consistent history
        for k in train_stats.keys():
            if k not in train_metrics and k != "time" and k != "lr":
                train_stats[k].append(None)

        # Always record time and learning rate
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        train_stats["time"].append(total_time)
        train_stats["lr"].append(optimizer.param_groups[0]["lr"])

        # Determine current objective value (IWAE if available, else ELBO)
        current_objective = train_metrics.get(
            "iwae", train_metrics.get("elbo", -float("inf"))
        )

        # Save best model
        if current_objective > best_elbo:
            best_elbo = current_objective
            best_state = {
                "flow": {k: v.cpu().clone() for k, v in flow.state_dict().items()},
                "recognition": {
                    k: v.cpu().clone() for k, v in recognition_net.state_dict().items()
                },
            }
            print(f"New best objective: {best_elbo:.4f}")

        # Print status
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        print(
            f"Epoch {epoch+1}/{n_epochs}, {metric_str}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Visualize progress every 10 epochs
        if epoch % 10 == 0 and output_dir:
            # Generate and save a diagnostic plot
            bin_name = (
                os.path.basename(output_dir).split("_")[0] if output_dir else "unknown"
            )
            training_plot_path = os.path.join(
                plots_dir, f"training_progress_{bin_name}_epoch_{epoch}.png"
            )

            plt.figure(figsize=(15, 10))

            # Plot ELBO/IWAE
            plt.subplot(2, 2, 1)
            if any(v is not None for v in train_stats["elbo"]):
                plt.plot(
                    [i for i, v in enumerate(train_stats["elbo"]) if v is not None],
                    [v for v in train_stats["elbo"] if v is not None],
                    "b-",
                    marker="o",
                    label="ELBO",
                )
            if any(v is not None for v in train_stats["iwae"]):
                plt.plot(
                    [i for i, v in enumerate(train_stats["iwae"]) if v is not None],
                    [v for v in train_stats["iwae"] if v is not None],
                    "g-",
                    marker="o",
                    label="IWAE",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Objective")
            plt.title("Training Progress: Objective")
            plt.legend()
            plt.grid(True)

            # Plot Reconstruction vs KL terms
            plt.subplot(2, 2, 2)
            if any(v is not None for v in train_stats["reconstruction"]):
                plt.plot(
                    [
                        i
                        for i, v in enumerate(train_stats["reconstruction"])
                        if v is not None
                    ],
                    [v for v in train_stats["reconstruction"] if v is not None],
                    "r-",
                    marker="o",
                    label="Reconstruction",
                )
            if any(v is not None for v in train_stats["kl"]):
                plt.plot(
                    [i for i, v in enumerate(train_stats["kl"]) if v is not None],
                    [v for v in train_stats["kl"] if v is not None],
                    "m-",
                    marker="o",
                    label="KL",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title("ELBO Components")
            plt.legend()
            plt.grid(True)

            # Plot Effective Sample Size
            plt.subplot(2, 2, 3)
            if any(v is not None for v in train_stats["ess"]):
                plt.plot(
                    [i for i, v in enumerate(train_stats["ess"]) if v is not None],
                    [v for v in train_stats["ess"] if v is not None],
                    "c-",
                    marker="o",
                )
                plt.xlabel("Epoch")
                plt.ylabel("ESS / K")
                plt.title("Effective Sample Size Ratio")
                plt.grid(True)

            # Plot Learning Rate
            plt.subplot(2, 2, 4)
            plt.plot(train_stats["lr"], "k-", marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(training_plot_path)
            plt.close()

            print(f"Training progress plot saved to {training_plot_path}")

    # Load best model
    flow.load_state_dict(best_state["flow"])
    recognition_net.load_state_dict(best_state["recognition"])
    flow.to(device)
    recognition_net.to(device)

    print(f"Training complete. Best objective: {best_elbo:.4f}")
    return flow, recognition_net, scaler, train_stats
