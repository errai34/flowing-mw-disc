#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to generate age-metallicity scatter plots from existing trained models.
"""

import argparse
import os

import torch
import yaml

from flow_model import Flow5D
from visualization import (
    plot_age_metallicity_scatter,
    plot_multiple_bin_age_metallicity,
)


def load_models(models_dir):
    """
    Load trained flow models from a directory.

    Parameters:
    -----------
    models_dir : str
        Directory containing model files

    Returns:
    --------
    tuple
        (flows_dict, scalers_dict)
    """
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    flows_dict = {}
    scalers_dict = {}

    # Find all model files
    for filename in os.listdir(models_dir):
        if filename.endswith("_model.pt"):
            bin_name = filename.split("_model.pt")[0]
            model_path = os.path.join(models_dir, filename)

            print(f"Loading model for bin {bin_name} from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)

            # Initialize flow model
            flow = Flow5D().to(device)
            flow.load_state_dict(checkpoint["model_state"])
            flow.eval()

            flows_dict[bin_name] = flow
            scalers_dict[bin_name] = checkpoint["scaler"]

    print(f"Loaded {len(flows_dict)} models")
    return flows_dict, scalers_dict


def generate_scatter_plots(
    flows_dict,
    scalers_dict,
    output_dir,
    n_samples=2000,
    age_range=(0, 20),
    feh_range=(-1.5, 0.5),
):
    """
    Generate scatter plots of age vs. [Fe/H] for all models.

    Parameters:
    -----------
    flows_dict : dict
        Dictionary mapping bin names to flow models
    scalers_dict : dict
        Dictionary mapping bin names to scalers
    output_dir : str
        Directory to save results
    n_samples : int
        Number of samples to draw per model
    age_range : tuple
        (min, max) for age range
    feh_range : tuple
        (min, max) for [Fe/H] range
    """
    # Create output directory
    scatter_dir = os.path.join(output_dir, "scatter_plots")
    os.makedirs(scatter_dir, exist_ok=True)

    # Generate individual scatter plots
    for bin_name, flow in flows_dict.items():
        scaler = scalers_dict[bin_name]
        print(f"Generating scatter plot for {bin_name}...")

        plot_age_metallicity_scatter(
            flow,
            scaler,
            n_samples=n_samples,
            save_path=os.path.join(scatter_dir, f"{bin_name}_age_feh_scatter.png"),
            age_range=age_range,
            feh_range=feh_range,
            flip_age_axis=True,
        )

    # Generate combined scatter plot
    print("Generating combined scatter plot...")
    plot_multiple_bin_age_metallicity(
        flows_dict,
        scalers_dict,
        n_samples=n_samples // 2,  # Use fewer samples per bin for the combined plot
        save_path=os.path.join(scatter_dir, "all_bins_age_feh_scatter.png"),
        age_range=age_range,
        feh_range=feh_range,
        flip_age_axis=True,
    )

    print(f"Scatter plots saved to {scatter_dir}")


def main(args):
    """Main function."""
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Load models
    flows_dict, scalers_dict = load_models(args.models_dir)

    # Generate scatter plots
    generate_scatter_plots(
        flows_dict,
        scalers_dict,
        args.output_dir,
        n_samples=args.samples,
        age_range=(args.min_age, args.max_age),
        feh_range=(args.min_feh, args.max_feh),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate age-metallicity scatter plots from trained models"
    )
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Directory containing model files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--samples", type=int, default=2000, help="Number of samples per model"
    )
    parser.add_argument("--min_age", type=float, default=0, help="Minimum age to plot")
    parser.add_argument("--max_age", type=float, default=20, help="Maximum age to plot")
    parser.add_argument(
        "--min_feh", type=float, default=-1.5, help="Minimum [Fe/H] to plot"
    )
    parser.add_argument(
        "--max_feh", type=float, default=0.5, help="Maximum [Fe/H] to plot"
    )

    args = parser.parse_args()
    main(args)
