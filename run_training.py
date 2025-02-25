#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execution script for the improved density deconvolution flow model training.
Reduced epochs for faster execution.
"""

import argparse
import os
from datetime import datetime

import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_handler import StellarDataHandler, prepare_data_for_radial_bins
from flow_model import Flow5D
from uncertainty import RecognitionNetwork
from visualization import plot_bin_chemical_distribution

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main(args):
    """Main function to run the training."""
    print("Starting improved density deconvolution training...")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    config = {
        "data_config": {"apogee_path": args.data_path, "n_samples": args.n_samples},
        "model_config": {
            "n_transforms": args.n_transforms,
            "hidden_dims": [args.hidden_dim, args.hidden_dim],
            "num_bins": args.num_bins,
        },
        "training_config": {
            "pretraining_epochs": args.pretraining_epochs,
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "mc_samples": args.mc_samples,
            "use_importance_weighted": args.use_iwae,
        },
    }

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
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

        # Apply quality filters using the data handler
        filtered_mw = data_handler.apply_quality_filters(mw_data)
        print(f"After filtering, we have {len(filtered_mw)} stars")

        # If requested, take a subset of the data
        if args.n_samples > 0 and args.n_samples < len(filtered_mw):
            filtered_mw = filtered_mw.sample(args.n_samples, random_state=42)
            print(f"Using subset of {len(filtered_mw)} stars for training")

        # Prepare data for radial bins
        bin_ranges = []
        for bin_range in args.bin_ranges:
            r_min, r_max = map(float, bin_range.split("-"))
            bin_ranges.append((r_min, r_max))

        bin_data = prepare_data_for_radial_bins(filtered_mw, bin_ranges)

        # Train model for selected bin
        if args.bin_name not in bin_data:
            print(
                f"Bin {args.bin_name} not found. Available bins: {list(bin_data.keys())}"
            )
            return

        bin_info = bin_data[args.bin_name]
        print(f"Training model for bin {args.bin_name} with {bin_info['count']} stars")

        # Get data and errors
        data = bin_info["data"]
        errors = bin_info["err"]

        # Scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        errors_scaled = errors / scaler.scale_

        # Convert to tensors
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
        errors_tensor = torch.tensor(errors_scaled, dtype=torch.float32).to(device)

        # Create dataset and loader
        dataset = TensorDataset(data_tensor, errors_tensor)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Initialize models
        flow = Flow5D(
            n_transforms=args.n_transforms,
            hidden_dims=[args.hidden_dim, args.hidden_dim],
            num_bins=args.num_bins,
            tail_bound=5.0,
            use_residual_blocks=True,
            dropout_probability=0.1,
        ).to(device)

        recognition_net = RecognitionNetwork(
            input_dim=5,  # Assuming 5D data
            n_transforms=8,
            hidden_dims=[args.hidden_dim, args.hidden_dim],
        ).to(device)

        # Set up optimizer
        optimizer = torch.optim.AdamW(
            list(flow.parameters()) + list(recognition_net.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Training loop
        bin_output_dir = os.path.join(output_dir, args.bin_name)
        os.makedirs(bin_output_dir, exist_ok=True)

        # Import the custom training function with modifications
        from main import train_flow_model

        flow, recognition_net, scaler, stats = train_flow_model(
            data=data,
            errors=errors,
            n_transforms=args.n_transforms,
            hidden_dims=[args.hidden_dim, args.hidden_dim],
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mc_samples=args.mc_samples,
            use_importance_weighted=args.use_iwae,
            output_dir=bin_output_dir,
            pretrain_epochs=args.pretraining_epochs,  # Add this parameter
        )

        # Save trained model
        model_path = os.path.join(bin_output_dir, f"{args.bin_name}_model.pt")
        torch.save(
            {
                "flow_state": flow.state_dict(),
                "recognition_state": recognition_net.state_dict(),
                "scaler": scaler,
                "stats": stats,
            },
            model_path,
        )

        print(f"Model saved to {model_path}")

        # Generate visualization
        plot_bin_chemical_distribution(
            flow=flow,
            scaler=scaler,
            bin_name=args.bin_name,
            save_dir=os.path.join(output_dir, "plots"),
            age_range=(0, 14),
            feh_range=(-1.0, 0.5),
            mgfe_range=(0.0, 0.4),
        )

        print("Training and visualization complete!")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run improved density deconvolution training"
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
        default=12,  # Reduced from 16
        help="Number of transforms in flow",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,  # Reduced from 256
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=24,  # Reduced from 32
        help="Number of bins in spline transforms",
    )

    # Training parameters
    parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=10,  # Reduced from 20
        help="Number of pretraining epochs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,  # Reduced from 100
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=20,  # Reduced from 50
        help="Number of Monte Carlo samples",
    )
    parser.add_argument(
        "--use_iwae", action="store_true", help="Use importance weighted ELBO"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )

    args = parser.parse_args()
    main(args)
