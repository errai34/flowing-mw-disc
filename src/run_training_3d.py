#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execution script for 3D normalizing flow model training.
Simplified to focus only on age, [Fe/H], and [Mg/Fe].
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data_handler import StellarDataHandler
from src.flow_model import Flow3D

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def format_bin_name(bin_range):
    """
    Format bin name to ensure consistent naming with decimal points.
    Converts '0-6' to 'R0.0-6.0'
    """
    r_min, r_max = map(float, bin_range.split("-"))
    return f"R{r_min:.1f}-{r_max:.1f}"


def prepare_data_for_radial_bins_3d(df, radial_bins=None):
    """
    Prepare 3D data (age, [Fe/H], [Mg/Fe]) for analysis in radial bins.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with necessary stellar parameters
    radial_bins : list
        List of (min_R, max_R) tuples defining radial bins

    Returns:
    --------
    dict
        Dictionary mapping bin names to data and errors
    """
    # Define default radial bins if not provided
    if radial_bins is None:
        radial_bins = [(0, 6), (6, 8), (8, 10), (10, 15)]

    # Create a proper copy to avoid SettingWithCopyWarning
    df = df.copy()

    bin_data = {}

    for r_min, r_max in radial_bins:
        bin_name = f"R{r_min:.1f}-{r_max:.1f}"
        mask = (df["rmean"] >= r_min) & (df["rmean"] < r_max)
        bin_df = df[mask]

        # Skip if too few stars
        if len(bin_df) < 100:
            print(f"Warning: Bin {bin_name} has only {len(bin_df)} stars, skipping")
            continue

        # Extract data and errors for the 3D model only (age, [Fe/H], [Mg/Fe])
        data_3d = bin_df[["pred_logAge", "FE_H", "MG_FE"]].values
        err_3d = bin_df[["pred_logAge_std", "FE_H_ERR", "MG_FE_ERR"]].values

        bin_data[bin_name] = {"data": data_3d, "err": err_3d, "count": len(bin_df)}

        print(f"Bin {bin_name}: {len(bin_df)} stars")

    return bin_data


def train_flow_model_3d(
    data,
    errors,
    n_transforms=12,
    hidden_dims=None,
    n_epochs=50,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    output_dir=None,
):
    """
    Train a 3D flow model on the data.

    Parameters:
    -----------
    data : np.ndarray
        Training data of shape (N, 3) for [age, [Fe/H], [Mg/Fe]]
    errors : np.ndarray
        Measurement errors for training data
    n_transforms : int
        Number of transforms in the flow
    hidden_dims : list
        List of hidden layer dimensions
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    output_dir : str
        Directory to save training progress

    Returns:
    --------
    tuple
        (trained flow model, scaler, training stats)
    """
    print("Starting 3D flow model training...")

    # Ensure integer parameters
    n_epochs = int(n_epochs)

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
    flow = Flow3D(
        n_transforms=n_transforms,
        hidden_dims=hidden_dims,
        num_bins=24,
        tail_bound=5.0,
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
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
        "log_likelihood": [],
        "lr": [],
    }

    best_log_likelihood = -float("inf")
    best_state = None

    print(f"Training on {len(data)} data points for {n_epochs} epochs")

    # Main training loop
    for epoch in range(n_epochs):
        flow.train()
        epoch_log_likelihood = []

        # Batch loop with progress bar
        batch_progress = tqdm(
            loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False, ncols=100
        )

        for batch_data, _ in batch_progress:
            optimizer.zero_grad()

            # Compute log likelihood
            log_likelihood = flow.log_prob(batch_data)
            loss = -torch.mean(log_likelihood)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            optimizer.step()

            # Record batch statistics
            epoch_log_likelihood.append(-loss.item())
            batch_progress.set_postfix({"LL": f"{-loss.item():.4f}"})

        # Update learning rate
        scheduler.step()

        # Record epoch stats
        avg_log_likelihood = np.mean(epoch_log_likelihood)
        train_stats["log_likelihood"].append(avg_log_likelihood)
        train_stats["lr"].append(optimizer.param_groups[0]["lr"])

        # Save best model
        if avg_log_likelihood > best_log_likelihood:
            best_log_likelihood = avg_log_likelihood
            best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            print(f"New best log likelihood: {best_log_likelihood:.4f}")

        # Print status
        print(
            f"Epoch {epoch+1}/{n_epochs}, Log-Likelihood: {avg_log_likelihood:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    # Load best model
    flow.load_state_dict(best_state)
    flow.to(device)

    print(f"Training complete. Best log likelihood: {best_log_likelihood:.4f}")
    return flow, scaler, train_stats


def sample_flow_and_visualize(flow, scaler, bin_name, save_dir, n_samples=5000):
    """
    Sample from the flow model and create visualizations.

    Parameters:
    -----------
    flow : Flow3D
        Trained flow model
    scaler : StandardScaler
        Scaler used to normalize the data
    bin_name : str
        Name of the radial bin
    save_dir : str
        Directory to save visualizations
    n_samples : int
        Number of samples to draw from the model
    """
    import matplotlib.pyplot as plt

    # Set evaluation mode
    flow.eval()

    # Sample from the flow
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    # Inverse transform to get original scale
    samples_original = scaler.inverse_transform(samples)

    # Extract age, [Fe/H], and [Mg/Fe]
    log_ages = samples_original[:, 0]
    fehs = samples_original[:, 1]
    mgfes = samples_original[:, 2]

    # Convert log age to linear age
    ages = 10**log_ages

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Age vs [Fe/H]
    ax1.scatter(ages, fehs, s=3, alpha=0.6, color="blue")
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Fe/H]")
    ax1.set_title(f"{bin_name}: Age vs. [Fe/H]")
    ax1.set_xlim(14, 0)  # Reversed to show oldest stars on left
    ax1.set_ylim(-1.5, 0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Age vs [Mg/Fe]
    ax2.scatter(ages, mgfes, s=3, alpha=0.6, color="green")
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_ylabel("[Mg/Fe]")
    ax2.set_title(f"{bin_name}: Age vs. [Mg/Fe]")
    ax2.set_xlim(14, 0)  # Reversed to show oldest stars on left
    ax2.set_ylim(-0.2, 0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, f"{bin_name}_3d_chemical_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(args):
    """Main function to run the training."""
    print("Starting 3D flow model training...")

    # Create models directory if it doesn't exist
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Create run directory for logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_3d_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Format bin name consistently
    formatted_bin_name = args.bin_name
    if "-" in args.bin_name and not args.bin_name.startswith("R"):
        formatted_bin_name = format_bin_name(args.bin_name)
    elif "-" in args.bin_name and args.bin_name.startswith("R"):
        range_part = args.bin_name[1:]
        formatted_bin_name = format_bin_name(range_part)

    # Save configuration
    config = {
        "data_config": {"apogee_path": args.data_path, "n_samples": args.n_samples},
        "model_config": {
            "n_transforms": args.n_transforms,
            "hidden_dims": [args.hidden_dim, args.hidden_dim],
            "num_bins": args.num_bins,
            "model_type": "3D",  # Explicitly note we're using a 3D model
        },
        "training_config": {
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
        },
        "bin_name": formatted_bin_name,
    }

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Load data
    data_handler = StellarDataHandler(args.data_path, "")

    try:
        print("Attempting to load data...")
        if args.data_format == "csv":
            mw_data = data_handler.load_apogee_csv()
        else:
            mw_data = data_handler.load_apogee_fits()

        print(f"Successfully loaded {len(mw_data)} stars")

        # Apply quality filters
        filtered_mw = data_handler.apply_quality_filters(mw_data)
        print(f"After filtering, we have {len(filtered_mw)} stars")

        # If requested, take a subset of the data
        if args.n_samples > 0 and args.n_samples < len(filtered_mw):
            filtered_mw = filtered_mw.sample(args.n_samples, random_state=42)
            print(f"Using subset of {len(filtered_mw)} stars for training")

        # Convert ranges to actual numeric values
        bin_ranges = []
        formatted_bin_ranges = []

        for bin_range in args.bin_ranges:
            formatted = format_bin_name(bin_range)
            formatted_bin_ranges.append(formatted)

            # Extract min and max from formatted name
            r_min, r_max = map(float, formatted[1:].split("-"))
            bin_ranges.append((r_min, r_max))

        # Prepare 3D data for radial bins
        bin_data = prepare_data_for_radial_bins_3d(filtered_mw, bin_ranges)

        # Train model for selected bin
        if formatted_bin_name not in bin_data:
            print(
                f"Bin {formatted_bin_name} not found. Available bins: {list(bin_data.keys())}"
            )
            return

        bin_info = bin_data[formatted_bin_name]
        print(
            f"Training model for bin {formatted_bin_name} with {bin_info['count']} stars"
        )

        # Get data and errors
        data = bin_info["data"]
        errors = bin_info["err"]

        # Train the 3D flow model
        flow, scaler, stats = train_flow_model_3d(
            data=data,
            errors=errors,
            n_transforms=args.n_transforms,
            hidden_dims=[args.hidden_dim, args.hidden_dim],
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            output_dir=run_dir,
        )

        # Save trained model
        model_path = os.path.join(models_dir, f"{formatted_bin_name}_3d_model.pt")
        torch.save(
            {
                "model_state": flow.state_dict(),
                "scaler": scaler,
                "stats": stats,
                "model_config": {
                    "n_transforms": args.n_transforms,
                    "hidden_dims": [args.hidden_dim, args.hidden_dim],
                    "num_bins": args.num_bins,
                    "model_type": "3D",
                },
            },
            model_path,
        )

        print(f"Model saved to {model_path}")

        # Generate visualization
        sample_flow_and_visualize(
            flow=flow,
            scaler=scaler,
            bin_name=formatted_bin_name,
            save_dir=plots_dir,
            n_samples=5000,
        )

        print("Training and visualization complete!")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3D normalizing flow model training"
    )

    # Data parameters
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="fits",
        choices=["fits", "csv"],
        help="Data format (fits or csv)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="Number of samples to use (-1 for all)",
    )
    parser.add_argument(
        "--bin_name", type=str, default="R8-10", help="Radial bin to train on"
    )
    parser.add_argument(
        "--bin_ranges",
        type=str,
        nargs="+",
        default=["0-6", "6-8", "8-10", "10-15"],
        help="Radial bin ranges",
    )

    # Model parameters
    parser.add_argument(
        "--n_transforms",
        type=int,
        default=12,
        help="Number of transforms in flow",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=24,
        help="Number of bins in spline transforms",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")

    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )

    args = parser.parse_args()
    main(args)
