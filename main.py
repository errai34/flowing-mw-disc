#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for training and analyzing normalizing flow models of Galactic evolution.
Implements the analysis pipeline described in the paper.
"""

import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
import yaml  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from data_handler import StellarDataHandler, prepare_data_for_radial_bins  # noqa: E402

# Import the new flow model implementation instead of the old one
from flow_model import Flow5D  # noqa: E402
from gradient_analysis import analyze_metallicity_gradients  # noqa: E402
from uncertainty import uncertainty_aware_elbo  # noqa: E402
from visualization import (  # noqa: E402
    create_master_gradient_plot,
    plot_bin_chemical_distribution,
    plot_gradient_analysis,
    plot_radial_bin_comparison,
)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_flow_model(
    data,
    errors,
    n_transforms=16,
    hidden_dims=None,
    n_epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    output_dir=None,
):
    """
    Train a normalizing flow model on data with uncertainty-aware training.

    Parameters:
    -----------
    data : np.ndarray
        Data array of shape (N, 5)
    errors : np.ndarray
        Error array of shape (N, 5)
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

    Returns:
    --------
    tuple
        (trained flow model, scaler, training stats)
    """
    # Use default hidden dimensions if not provided
    if hidden_dims is None:
        hidden_dims = [128, 128]

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

    # Initialize flow model
    # Use the new nflows-based implementation
    flow = Flow5D(n_transforms=n_transforms, hidden_dims=hidden_dims).to(device)

    # Optimizer
    optimizer = optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    # We'll manually print learning rate changes instead of using verbose=True

    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Training loop
    train_stats = {"elbo": [], "lr": [], "time": []}
    best_elbo = -float("inf")
    best_state = None
    start_time = time.time()

    print(f"Training on {len(data)} data points for {n_epochs} epochs")
    for epoch in range(n_epochs):
        epoch_start = time.time()
        flow.train()
        epoch_elbos = []

        # Create progress bar for batches
        batch_progress = tqdm(
            loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False, ncols=100
        )

        for batch_data, batch_errors in batch_progress:
            optimizer.zero_grad()

            # Compute ELBO with uncertainty
            elbo = uncertainty_aware_elbo(flow, batch_data, batch_errors, K=10)
            loss = -elbo

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            optimizer.step()

            epoch_elbos.append(elbo.item())
            # Update progress bar
            batch_progress.set_postfix({"ELBO": f"{elbo.item():.4f}"})

        # Track stats
        avg_elbo = np.mean(epoch_elbos)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        train_stats["elbo"].append(avg_elbo)
        train_stats["lr"].append(optimizer.param_groups[0]["lr"])
        train_stats["time"].append(total_time)

        # Update scheduler
        scheduler.step(avg_elbo)

        # Save best model
        if avg_elbo > best_elbo:
            best_elbo = avg_elbo
            best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            print(f"New best ELBO: {best_elbo:.4f}")

        # Print status more frequently for enhanced progress display
        print(
            f"Epoch {epoch+1}/{n_epochs}, ELBO: {avg_elbo:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )

    # Plot training progress
    plt.figure(figsize=(15, 5))

    # Plot ELBO
    plt.subplot(1, 3, 1)
    plt.plot(train_stats["elbo"], "b-", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.title("Training Progress: ELBO")
    plt.grid(True)

    # Plot Learning Rate
    plt.subplot(1, 3, 2)
    plt.plot(train_stats["lr"], "r-", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)

    # Plot Training Time
    plt.subplot(1, 3, 3)
    plt.plot(range(len(train_stats["time"])), train_stats["time"], "g-", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Time (s)")
    plt.title("Training Time")
    plt.grid(True)

    # Save the plot
    bin_name = os.path.basename(output_dir).split("_")[0] if output_dir else "unknown"
    training_plot_path = os.path.join(plots_dir, f"training_progress_{bin_name}.png")
    plt.tight_layout()
    plt.savefig(training_plot_path)
    plt.close()

    print(f"Training plot saved to {training_plot_path}")

    # Load best model
    flow.load_state_dict(best_state)
    flow.to(device)

    print(f"Training complete. Best ELBO: {best_elbo:.4f}")
    return flow, scaler, train_stats


def train_models_for_all_bins(
    bin_data, output_dir, config=None, n_epochs=100, lr=1e-3, force_retrain=False
):
    """
    Train models for all radial bins and save results.

    Parameters:
    -----------
    bin_data : dict
        Dictionary with bin data from prepare_data_for_radial_bins
    output_dir : str
        Directory to save models and results
    config : dict
        Configuration dictionary
    n_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    force_retrain : bool
        Whether to force retraining of models

    Returns:
    --------
    tuple
        (flows_dict, scalers_dict, stats_dict)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    flows_dict = {}
    scalers_dict = {}
    stats_dict = {}

    # Extract training parameters from config if available
    if config is not None:
        flow_config = config.get("flow", {})
        training_config = config.get("training", {})

        n_transforms = flow_config.get("n_transforms", 16)
        hidden_dims = flow_config.get("hidden_dims", [128, 128])
        batch_size = training_config.get("batch_size", 256)
        weight_decay = float(training_config.get("weight_decay", 1e-5))
    else:
        n_transforms = 16
        hidden_dims = [128, 128]
        batch_size = 256
        weight_decay = 1e-5

    # Extract visualization parameters
    if config is not None:
        analysis_config = config.get("analysis", {})
        grid_resolution = analysis_config.get("grid_resolution", 50)
    else:
        grid_resolution = 50

    for bin_name, bin_info in bin_data.items():
        print(f"\nTraining model for bin {bin_name}...")
        model_path = os.path.join(output_dir, f"{bin_name}_model.pt")

        # Skip if model already exists and not force_retrain
        if os.path.exists(model_path) and not force_retrain:
            print(f"Loading existing model for {bin_name} from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)

            # Use the new nflows-based implementation
            flow = Flow5D().to(device)
            flow.load_state_dict(checkpoint["model_state"])

            flows_dict[bin_name] = flow
            scalers_dict[bin_name] = checkpoint["scaler"]
            stats_dict[bin_name] = checkpoint["stats"]
            continue

        # Train new model
        flow, scaler, stats = train_flow_model(
            bin_info["data"],
            bin_info["err"],
            n_transforms=n_transforms,
            hidden_dims=hidden_dims,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            output_dir=os.path.join(output_dir, bin_name),
        )

        # Save model
        torch.save(
            {"model_state": flow.state_dict(), "scaler": scaler, "stats": stats},
            model_path,
        )

        flows_dict[bin_name] = flow
        scalers_dict[bin_name] = scaler
        stats_dict[bin_name] = stats

        print(f"Model for {bin_name} saved to {model_path}")

    # Visual distributions are created in the analyze_all_bins function instead
    return flows_dict, scalers_dict, stats_dict


def analyze_all_bins(flows_dict, scalers_dict, output_dir, config=None):
    """
    Run analysis for all bins and create visualizations.

    Parameters:
    -----------
    flows_dict : dict
        Dictionary mapping bin names to flow models
    scalers_dict : dict
        Dictionary mapping bin names to scalers
    output_dir : str
        Directory to save results
    config : dict
        Configuration dictionary
    """
    # Create output directories
    plots_dir = os.path.join(output_dir, "plots")
    gradient_dir = os.path.join(output_dir, "gradients")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(gradient_dir, exist_ok=True)

    # Extract analysis parameters from config if available
    if config is not None:
        analysis_config = config.get("analysis", {})
        age_values = analysis_config.get("age_values", [4, 6, 8, 10, 12])
        feh_min, feh_max, feh_steps = analysis_config.get("feh_range", [-1.0, 0.5, 100])
        age_range = (0, 14)
        feh_range = (-1.0, 0.5)
        mgfe_range = (0.0, 0.4)
        grid_resolution = analysis_config.get("grid_resolution", 50)
    else:
        age_values = [4, 6, 8, 10, 12]
        feh_min, feh_max, feh_steps = -1.0, 0.5, 100
        age_range = (0, 14)
        feh_range = (-1.0, 0.5)
        mgfe_range = (0.0, 0.4)
        grid_resolution = 50

    # Create individual bin plots
    for bin_name, flow in flows_dict.items():
        scaler = scalers_dict[bin_name]

        # Chemical distribution plots
        print(f"Generating chemical distribution plots for {bin_name}...")
        plot_bin_chemical_distribution(
            flow,
            scaler,
            bin_name,
            save_dir=plots_dir,
            age_range=age_range,
            feh_range=feh_range,
            mgfe_range=mgfe_range,
            grid_resolution=grid_resolution,
        )

        # Gradient analysis
        gradient_results = analyze_metallicity_gradients(
            flow, scaler, age_values=age_values, feh_range=(feh_min, feh_max, feh_steps)
        )

        # Plot gradient analysis
        plot_gradient_analysis(
            gradient_results,
            bin_name,
            save_path=os.path.join(gradient_dir, f"{bin_name}_gradients.png"),
        )

    # Create comparison plots across bins
    print("Generating radial bin comparison plot...")
    plot_radial_bin_comparison(
        flows_dict,
        scalers_dict,
        save_path=os.path.join(plots_dir, "radial_bin_comparison.png"),
    )

    # Create master gradient plot
    print("Generating master gradient plot...")
    gradient_results_dict = {}

    for bin_name, flow in flows_dict.items():
        scaler = scalers_dict[bin_name]
        gradient_results_dict[bin_name] = analyze_metallicity_gradients(
            flow, scaler, age_values=age_values, feh_range=(feh_min, feh_max, feh_steps)
        )

    create_master_gradient_plot(
        gradient_results_dict,
        save_path=os.path.join(gradient_dir, "master_gradient_plot.png"),
    )


def main(args):
    """Main function to run the full analysis pipeline"""
    print(f"Using device: {device}")

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    data_handler = StellarDataHandler.from_config(args.config)

    try:
        # Try loading from CSV first
        print("Attempting to load APOGEE data from CSV...")
        mw_data = data_handler.load_apogee_csv()
    except FileNotFoundError:
        # If CSV fails, try FITS
        print("CSV not found, attempting to load from FITS...")
        try:
            mw_data = data_handler.load_apogee_fits()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not find APOGEE data files. Please check your paths in config.yaml."
            )

    print(f"Successfully loaded {len(mw_data)} stars")

    # Apply quality filters using the data handler
    filtered_mw = data_handler.apply_quality_filters(mw_data)
    print(f"After filtering, we have {len(filtered_mw)} stars")

    # Prepare data for radial bins - use bins directly from config
    bin_data = prepare_data_for_radial_bins(filtered_mw, config.get("radial_bins"))

    # Create output directory
    output_dir = os.path.join(args.output_dir, "models")
    os.makedirs(
        args.output_dir, exist_ok=True
    )  # Create main output dir if it doesn't exist

    # Get training parameters, with command line arguments taking precedence
    training_config = config.get("training", {})
    n_epochs = args.epochs or training_config.get("n_epochs", 100)
    lr = args.lr or float(training_config.get("lr", 1e-3))

    # Train models for all bins
    flows_dict, scalers_dict, stats_dict = train_models_for_all_bins(
        bin_data,
        output_dir,
        config=config,
        n_epochs=n_epochs,
        lr=lr,
        force_retrain=args.force_retrain,
    )

    # Run analysis
    analyze_all_bins(flows_dict, scalers_dict, args.output_dir, config=config)

    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and analyze normalizing flow models for Galactic evolution"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--force_retrain", action="store_true", help="Force retraining of models"
    )

    args = parser.parse_args()
    main(args)
