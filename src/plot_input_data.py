#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to plot the input data that goes into the normalizing flow model.
This will help visualize what the model is actually learning from.
"""

import os

import matplotlib.pyplot as plt

from src.data_handler import StellarDataHandler


def plot_data_distributions(
    data_path, bin_name="R0-6", n_samples=2000, output_dir="plots"
):
    """
    Plot the distributions of the data that goes into the model.

    Parameters:
    -----------
    data_path : str
        Path to the data directory
    bin_name : str
        Name of the radial bin to analyze
    n_samples : int
        Number of samples to use (-1 for all)
    output_dir : str
        Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...")
    data_handler = StellarDataHandler(data_path, "")

    try:
        # Try to load data
        mw_data = data_handler.load_apogee_fits()
        print(f"Successfully loaded {len(mw_data)} stars")

        # Apply quality filters
        filtered_mw = data_handler.apply_quality_filters(mw_data)
        print(f"After filtering, we have {len(filtered_mw)} stars")

        # If requested, take a subset of the data
        if n_samples > 0 and n_samples < len(filtered_mw):
            filtered_mw = filtered_mw.sample(n_samples, random_state=42)
            print(f"Using subset of {len(filtered_mw)} stars for visualization")

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

        if len(bin_df) < 50:
            print(
                f"Warning: Bin {bin_name} has only {len(bin_df)} stars, which may not be enough for visualization"
            )

        print(f"Plotting data for bin {bin_name} with {len(bin_df)} stars")

        # Extract the variables we're interested in
        log_ages = bin_df["pred_logAge"].values
        fehs = bin_df["FE_H"].values
        mgfes = bin_df["MG_FE"].values
        age_errs = bin_df["pred_logAge_std"].values
        feh_errs = bin_df["FE_H_ERR"].values
        mgfe_errs = bin_df["MG_FE_ERR"].values

        # Convert log age to linear age
        ages = 10**log_ages

        # Create figures
        # 1. Age-[Fe/H] scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(ages, fehs, c=mgfes, cmap="viridis", alpha=0.7, s=15)
        plt.colorbar(scatter, label="[Mg/Fe]")
        plt.xlabel("Age (Gyr)")
        plt.ylabel("[Fe/H]")
        plt.title(f"Age vs [Fe/H] for {bin_name} kpc")
        plt.xlim(0, 14)
        plt.ylim(-1.5, 0.5)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{bin_name}_age_feh.png"), dpi=300)
        plt.close()

        # 2. Age-[Mg/Fe] scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(ages, mgfes, c=fehs, cmap="plasma", alpha=0.7, s=15)
        plt.colorbar(scatter, label="[Fe/H]")
        plt.xlabel("Age (Gyr)")
        plt.ylabel("[Mg/Fe]")
        plt.title(f"Age vs [Mg/Fe] for {bin_name} kpc")
        plt.xlim(0, 14)
        plt.ylim(-0.2, 0.5)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{bin_name}_age_mgfe.png"), dpi=300)
        plt.close()

        # 3. [Fe/H]-[Mg/Fe] scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(fehs, mgfes, c=ages, cmap="viridis", alpha=0.7, s=15)
        plt.colorbar(scatter, label="Age (Gyr)")
        plt.xlabel("[Fe/H]")
        plt.ylabel("[Mg/Fe]")
        plt.title(f"[Fe/H] vs [Mg/Fe] for {bin_name} kpc")
        plt.xlim(-1.5, 0.5)
        plt.ylim(-0.2, 0.5)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{bin_name}_feh_mgfe.png"), dpi=300)
        plt.close()

        # 4. Error distributions
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].scatter(ages, age_errs, c=ages, cmap="viridis", alpha=0.7, s=15)
        axs[0].set_xlabel("Age (Gyr)")
        axs[0].set_ylabel("Log Age Error")
        axs[0].set_title("Age vs Log Age Error")
        axs[0].grid(alpha=0.3)

        axs[1].scatter(ages, feh_errs, c=ages, cmap="viridis", alpha=0.7, s=15)
        axs[1].set_xlabel("Age (Gyr)")
        axs[1].set_ylabel("[Fe/H] Error")
        axs[1].set_title("Age vs [Fe/H] Error")
        axs[1].grid(alpha=0.3)

        axs[2].scatter(ages, mgfe_errs, c=ages, cmap="viridis", alpha=0.7, s=15)
        axs[2].set_xlabel("Age (Gyr)")
        axs[2].set_ylabel("[Mg/Fe] Error")
        axs[2].set_title("Age vs [Mg/Fe] Error")
        axs[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{bin_name}_errors.png"), dpi=300)
        plt.close()

        # 5. Combined 3D data visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            log_ages, fehs, mgfes, c=ages, cmap="viridis", s=15, alpha=0.7
        )
        plt.colorbar(scatter, label="Age (Gyr)")
        ax.set_xlabel("Log Age")
        ax.set_ylabel("[Fe/H]")
        ax.set_zlabel("[Mg/Fe]")
        ax.set_title(f"3D Data Distribution for {bin_name} kpc")
        plt.savefig(os.path.join(output_dir, f"{bin_name}_3d_data.png"), dpi=300)
        plt.close()

        print(f"Plots saved to {output_dir}/")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Use argparse to handle command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot distributions of input data for flow model"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--bin_name", type=str, default="R0-6", help="Radial bin to analyze"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="Number of samples to use (-1 for all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Directory to save plots"
    )

    args = parser.parse_args()
    plot_data_distributions(
        args.data_path, args.bin_name, args.n_samples, args.output_dir
    )
