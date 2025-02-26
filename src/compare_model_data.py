#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to compare real data with model samples
This helps diagnose issues with model outputs
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_handler import StellarDataHandler
from src.flow_model import Flow3D


def load_trained_model(model_path):
    """
    Load a trained model from a checkpoint file.

    Parameters:
    -----------
    model_path : str
        Path to the model checkpoint file

    Returns:
    --------
    tuple
        (flow_model, scaler)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Check if it's the standard format or uncertainty model format
    if "model_state" in checkpoint:
        model_config = checkpoint.get(
            "model_config",
            {
                "n_transforms": 8,
                "hidden_dims": [64, 64],
                "num_bins": 12,
            },
        )

        # Initialize model
        flow = Flow3D(
            n_transforms=model_config.get("n_transforms", 8),
            hidden_dims=model_config.get("hidden_dims", [64, 64]),
            num_bins=model_config.get("num_bins", 12),
        )

        # Load state dict
        flow.load_state_dict(checkpoint["model_state"])
        scaler = checkpoint["scaler"]
    else:
        # Older model format
        flow = Flow3D(n_transforms=8, hidden_dims=[64, 64])
        flow.load_state_dict(checkpoint["flow_state"])
        scaler = checkpoint["scaler"]

    return flow, scaler


def load_data(data_path, bin_name, n_samples=-1):
    """
    Load and filter data for a specific radial bin.

    Parameters:
    -----------
    data_path : str
        Path to the data directory
    bin_name : str
        Name of the radial bin to analyze
    n_samples : int
        Number of samples to use (-1 for all)

    Returns:
    --------
    pandas.DataFrame
        Filtered data for the specified bin
    """
    data_handler = StellarDataHandler(data_path, "")

    # Try to load data
    try:
        mw_data = data_handler.load_apogee_fits()
    except FileNotFoundError:
        try:
            mw_data = data_handler.load_apogee_csv()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find APOGEE data in {data_path}")

    print(f"Successfully loaded {len(mw_data)} stars")

    # Apply quality filters
    filtered_mw = data_handler.apply_quality_filters(mw_data)
    print(f"After filtering, we have {len(filtered_mw)} stars")

    # If requested, take a subset of the data
    if n_samples > 0 and n_samples < len(filtered_mw):
        filtered_mw = filtered_mw.sample(n_samples, random_state=42)
        print(f"Using subset of {len(filtered_mw)} stars")

    # Extract the radial bin range from bin_name
    if bin_name.startswith("R"):
        range_part = bin_name[1:]
    else:
        range_part = bin_name

    try:
        r_min, r_max = map(float, range_part.split("-"))
    except ValueError:
        print(f"Invalid bin name format: {bin_name}. Using default 0-6 kpc.")
        r_min, r_max = 0.0, 6.0

    # Filter for the specified radial bin
    mask = (filtered_mw["rmean"] >= r_min) & (filtered_mw["rmean"] < r_max)
    bin_df = filtered_mw[mask]

    print(f"Bin {bin_name} has {len(bin_df)} stars")
    return bin_df


def compare_real_and_model_data(
    data_path,
    model_path,
    bin_name,
    n_samples=2000,
    n_model_samples=5000,
    output_dir="comparison_plots",
):
    """
    Compare real data with samples from a trained model.

    Parameters:
    -----------
    data_path : str
        Path to the data directory
    model_path : str
        Path to the model checkpoint file
    bin_name : str
        Name of the radial bin to analyze
    n_samples : int
        Number of data samples to use (-1 for all)
    n_model_samples : int
        Number of samples to generate from the model
    output_dir : str
        Directory to save the comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    bin_df = load_data(data_path, bin_name, n_samples)

    # Extract variables from real data
    real_log_ages = bin_df["pred_logAge"].values
    real_fehs = bin_df["FE_H"].values
    real_mgfes = bin_df["MG_FE"].values

    # Convert log age to linear age for plotting
    real_ages = 10**real_log_ages

    # Load model
    print(f"Loading model from {model_path}...")
    flow, scaler = load_trained_model(model_path)
    flow.eval()  # Set to evaluation mode

    # Generate samples from the model
    print(f"Generating {n_model_samples} samples from the model...")
    with torch.no_grad():
        model_samples = flow.sample(n_model_samples).cpu().numpy()

    # Inverse transform the samples to original scale
    model_samples_original = scaler.inverse_transform(model_samples)

    # Extract variables from model samples
    model_log_ages = model_samples_original[:, 0]
    model_fehs = model_samples_original[:, 1]
    model_mgfes = model_samples_original[:, 2]

    # Convert log age to linear age for plotting
    model_ages = 10**model_log_ages

    # Create comparison plots
    print("Creating comparison plots...")

    # 1. Age vs [Fe/H] comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    # Real data
    scatter1 = ax1.scatter(
        real_ages, real_fehs, c=real_mgfes, cmap="viridis", alpha=0.7, s=15
    )
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Fe/H]")
    ax1.set_title(f"Real Data: Age vs [Fe/H] for {bin_name} kpc")
    ax1.set_xlim(0, 14)
    ax1.set_ylim(-1.5, 0.5)
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="[Mg/Fe]")

    # Model samples
    scatter2 = ax2.scatter(
        model_ages, model_fehs, c=model_mgfes, cmap="viridis", alpha=0.7, s=15
    )
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_title(f"Model Samples: Age vs [Fe/H] for {bin_name} kpc")
    ax2.set_xlim(0, 14)
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label="[Mg/Fe]")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{bin_name}_age_feh_comparison.png"), dpi=300)
    plt.close()

    # 2. Age vs [Mg/Fe] comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    # Real data
    scatter1 = ax1.scatter(
        real_ages, real_mgfes, c=real_fehs, cmap="plasma", alpha=0.7, s=15
    )
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Mg/Fe]")
    ax1.set_title(f"Real Data: Age vs [Mg/Fe] for {bin_name} kpc")
    ax1.set_xlim(0, 14)
    ax1.set_ylim(-0.2, 0.5)
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="[Fe/H]")

    # Model samples
    scatter2 = ax2.scatter(
        model_ages, model_mgfes, c=model_fehs, cmap="plasma", alpha=0.7, s=15
    )
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_title(f"Model Samples: Age vs [Mg/Fe] for {bin_name} kpc")
    ax2.set_xlim(0, 14)
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label="[Fe/H]")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{bin_name}_age_mgfe_comparison.png"), dpi=300
    )
    plt.close()

    # 3. [Fe/H] vs [Mg/Fe] comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True, sharex=True)

    # Real data
    scatter1 = ax1.scatter(
        real_fehs, real_mgfes, c=real_ages, cmap="viridis", alpha=0.7, s=15
    )
    ax1.set_xlabel("[Fe/H]")
    ax1.set_ylabel("[Mg/Fe]")
    ax1.set_title(f"Real Data: [Fe/H] vs [Mg/Fe] for {bin_name} kpc")
    ax1.set_xlim(-1.5, 0.5)
    ax1.set_ylim(-0.2, 0.5)
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="Age (Gyr)")

    # Model samples
    scatter2 = ax2.scatter(
        model_fehs, model_mgfes, c=model_ages, cmap="viridis", alpha=0.7, s=15
    )
    ax2.set_xlabel("[Fe/H]")
    ax2.set_title(f"Model Samples: [Fe/H] vs [Mg/Fe] for {bin_name} kpc")
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label="Age (Gyr)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{bin_name}_feh_mgfe_comparison.png"), dpi=300
    )
    plt.close()

    # 4. Distribution comparison
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Log Age distribution
    axs[0].hist(real_log_ages, bins=30, alpha=0.5, label="Real data", density=True)
    axs[0].hist(model_log_ages, bins=30, alpha=0.5, label="Model samples", density=True)
    axs[0].set_xlabel("Log Age")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Log Age Distribution")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # [Fe/H] distribution
    axs[1].hist(real_fehs, bins=30, alpha=0.5, label="Real data", density=True)
    axs[1].hist(model_fehs, bins=30, alpha=0.5, label="Model samples", density=True)
    axs[1].set_xlabel("[Fe/H]")
    axs[1].set_ylabel("Density")
    axs[1].set_title("[Fe/H] Distribution")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # [Mg/Fe] distribution
    axs[2].hist(real_mgfes, bins=30, alpha=0.5, label="Real data", density=True)
    axs[2].hist(model_mgfes, bins=30, alpha=0.5, label="Model samples", density=True)
    axs[2].set_xlabel("[Mg/Fe]")
    axs[2].set_ylabel("Density")
    axs[2].set_title("[Mg/Fe] Distribution")
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{bin_name}_distribution_comparison.png"), dpi=300
    )
    plt.close()

    # 5. 3D comparison (projection views)
    fig = plt.figure(figsize=(18, 12))

    # 3D plot for real data
    ax1 = fig.add_subplot(231, projection="3d")
    sc1 = ax1.scatter(
        real_log_ages,
        real_fehs,
        real_mgfes,
        c=real_ages,
        cmap="viridis",
        s=15,
        alpha=0.5,
    )
    ax1.set_xlabel("Log Age")
    ax1.set_ylabel("[Fe/H]")
    ax1.set_zlabel("[Mg/Fe]")
    ax1.set_title("Real Data: 3D Distribution")

    # 3D plot for model samples
    ax2 = fig.add_subplot(232, projection="3d")
    sc2 = ax2.scatter(
        model_log_ages,
        model_fehs,
        model_mgfes,
        c=model_ages,
        cmap="viridis",
        s=15,
        alpha=0.5,
    )
    ax2.set_xlabel("Log Age")
    ax2.set_ylabel("[Fe/H]")
    ax2.set_zlabel("[Mg/Fe]")
    ax2.set_title("Model Samples: 3D Distribution")
    plt.colorbar(sc2, ax=ax2, label="Age (Gyr)")

    # Difference in distribution
    # Heat map or contour plots of differences in distributions
    ax3 = fig.add_subplot(233)
    ax3.hist2d(real_fehs, real_mgfes, bins=30, cmap="Blues", alpha=0.7, density=True)
    ax3.contour(
        *np.histogram2d(model_fehs, model_mgfes, bins=30, density=True)[1::-1],
        np.histogram2d(model_fehs, model_mgfes, bins=30, density=True)[0].T,
        colors="red",
        alpha=0.7,
    )
    ax3.set_xlabel("[Fe/H]")
    ax3.set_ylabel("[Mg/Fe]")
    ax3.set_title("[Fe/H] vs [Mg/Fe] Distribution Comparison")

    # Age-[Fe/H] comparison (histograms)
    ax4 = fig.add_subplot(234)
    ax4.hist2d(real_ages, real_fehs, bins=30, cmap="Blues", alpha=0.7, density=True)
    ax4.contour(
        *np.histogram2d(model_ages, model_fehs, bins=30, density=True)[1::-1],
        np.histogram2d(model_ages, model_fehs, bins=30, density=True)[0].T,
        colors="red",
        alpha=0.7,
    )
    ax4.set_xlabel("Age (Gyr)")
    ax4.set_ylabel("[Fe/H]")
    ax4.set_title("Age vs [Fe/H] Distribution Comparison")

    # Age-[Mg/Fe] comparison (histograms)
    ax5 = fig.add_subplot(235)
    ax5.hist2d(real_ages, real_mgfes, bins=30, cmap="Blues", alpha=0.7, density=True)
    ax5.contour(
        *np.histogram2d(model_ages, model_mgfes, bins=30, density=True)[1::-1],
        np.histogram2d(model_ages, model_mgfes, bins=30, density=True)[0].T,
        colors="red",
        alpha=0.7,
    )
    ax5.set_xlabel("Age (Gyr)")
    ax5.set_ylabel("[Mg/Fe]")
    ax5.set_title("Age vs [Mg/Fe] Distribution Comparison")

    # Legend/guide
    ax6 = fig.add_subplot(236)
    ax6.text(
        0.1, 0.7, "Distribution Comparison Legend:", fontsize=12, fontweight="bold"
    )
    ax6.text(0.1, 0.5, "Blue shaded areas = Real data", fontsize=10)
    ax6.text(0.1, 0.3, "Red contour lines = Model samples", fontsize=10)
    ax6.text(
        0.1,
        0.1,
        f"Model samples: {n_model_samples}, Real data: {len(bin_df)}",
        fontsize=10,
    )
    ax6.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{bin_name}_3d_comparison.png"), dpi=300)
    plt.close()

    print(f"Comparison plots saved to {output_dir}/")


if __name__ == "__main__":
    # Use argparse to handle command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Compare real data with model samples")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--bin_name", type=str, default="R0-6", help="Radial bin to analyze"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of data samples to use (-1 for all)",
    )
    parser.add_argument(
        "--n_model_samples",
        type=int,
        default=5000,
        help="Number of samples to generate from the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_plots",
        help="Directory to save plots",
    )

    args = parser.parse_args()
    compare_real_and_model_data(
        args.data_path,
        args.model_path,
        args.bin_name,
        args.n_samples,
        args.n_model_samples,
        args.output_dir,
    )
