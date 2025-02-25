import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from flow_model import Flow5D


def plot_age_metallicity_kde(
    flow,
    scaler,
    n_samples=5000,
    save_path=None,
    age_range=(0, 20),
    feh_range=(-1.5, 0.5),
    figsize=(10, 8),
    flip_age_axis=True,
    cmap="viridis",
    point_size=0.5,
    bin_name=None,
):
    """
    Create a KDE-based visualization of Age vs. [Fe/H] using samples from the flow model.

    Parameters:
    -----------
    flow : Flow5D
        Trained normalizing flow model
    scaler : StandardScaler
        Scaler used to normalize the data
    n_samples : int
        Number of samples to draw
    save_path : str
        Path to save the figure
    age_range : tuple
        (min, max) for age range in plot
    feh_range : tuple
        (min, max) for [Fe/H] range in plot
    figsize : tuple
        Figure size
    flip_age_axis : bool
        If True, plot age from high to low (oldest to youngest on the left)
    cmap : str
        Colormap to use for density visualization
    point_size : float
        Size of scatter points
    bin_name : str
        Name of the bin for title

    Returns:
    --------
    tuple
        (fig, ax) - Figure and axis objects
    """
    # Set device and evaluation mode
    device = next(flow.parameters()).device
    flow.eval()

    # Sample from the flow
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    # Inverse transform to get original scale
    samples_original = scaler.inverse_transform(samples)

    # Extract age and [Fe/H]
    log_ages = samples_original[:, 0]  # First dimension is log(age)
    fehs = samples_original[:, 1]  # Second dimension is [Fe/H]

    # Convert log age to linear age
    ages = 10**log_ages

    # Filter points within the specified ranges
    mask = (
        (ages >= age_range[0])
        & (ages <= age_range[1])
        & (fehs >= feh_range[0])
        & (fehs <= feh_range[1])
    )

    ages_filtered = ages[mask]
    fehs_filtered = fehs[mask]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate KDE on filtered data - swap order to have age on x-axis and [Fe/H] on y-axis
    xy = np.vstack([ages_filtered, fehs_filtered])
    kde = gaussian_kde(xy)

    # Create grid for KDE evaluation - note the swap of age to x-axis
    x_grid = np.linspace(age_range[0], age_range[1], 100)
    y_grid = np.linspace(feh_range[0], feh_range[1], 100)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Evaluate KDE on grid
    zz = kde(np.vstack([xx.ravel(), yy.ravel()]))
    zz = zz.reshape(xx.shape)

    # Plot KDE as contours with filled colors
    contour = ax.contourf(xx, yy, zz, levels=20, cmap=cmap, alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Density")

    # Add scatter points (with very small size for detail) - swap for age on x-axis
    ax.scatter(ages_filtered, fehs_filtered, s=point_size, color="k", alpha=0.1)

    # Set labels and title - swap axes labels
    ax.set_xlabel("Age (Gyr)")
    ax.set_ylabel("[Fe/H]")
    if bin_name:
        ax.set_title(f"Age-Metallicity Relation - {bin_name}")
    else:
        ax.set_title("Age-Metallicity Relation")

    # Set axis ranges - swap ranges
    ax.set_xlim(age_range)
    ax.set_ylim(feh_range)
    if flip_age_axis:
        ax.invert_xaxis()  # Flip x-axis to show oldest at left

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.5)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_multiple_bin_kde(
    flows_dict,
    scalers_dict,
    n_samples=2000,
    save_path=None,
    age_range=(0, 20),
    feh_range=(-1.5, 0.5),
    flip_age_axis=True,
):
    """
    Create KDE-based visualizations of Age vs. [Fe/H] for multiple radial bins.

    Parameters:
    -----------
    flows_dict : dict
        Dictionary mapping bin names to flow models
    scalers_dict : dict
        Dictionary mapping bin names to scalers
    n_samples : int
        Number of samples per bin
    save_path : str
        Path to save the figure
    age_range : tuple
        (min, max) for age range in plot
    feh_range : tuple
        (min, max) for [Fe/H] range in plot
    flip_age_axis : bool
        If True, plot age from high to low (oldest to youngest on the left)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Check if we have any models
    if not flows_dict:
        print("No models available for combined visualization")
        return None
        
    # Define bin order to ensure correct ordering from inner to outer
    radial_bin_order = ["R0.0-6.0", "R6.0-8.0", "R8.0-10.0", "R10.0-15.0"]
    
    # Filter bin order to only include available models
    available_bins = []
    for bin_name in radial_bin_order:
        if bin_name in flows_dict and bin_name in scalers_dict:
            available_bins.append(bin_name)
    
    # If no bins are available, return None
    if not available_bins:
        print("No valid models available for comparison")
        return None
        
    # Set up figure
    n_bins = len(available_bins)
    fig, axes = plt.subplots(
        1, n_bins, figsize=(5 * n_bins, 5), sharex=True, sharey=True
    )

    # Handle the case of a single bin
    if n_bins == 1:
        axes = [axes]

    # Plot each bin in the correct order
    for i, bin_name in enumerate(available_bins):
        flow = flows_dict[bin_name]
        device = next(flow.parameters()).device
        flow.eval()
        scaler = scalers_dict[bin_name]

        # Sample from the flow
        with torch.no_grad():
            samples = flow.sample(n_samples).cpu().numpy()

        # Inverse transform to get original scale
        samples_original = scaler.inverse_transform(samples)

        # Extract age and [Fe/H]
        log_ages = samples_original[:, 0]
        fehs = samples_original[:, 1]

        # Convert log age to linear age
        ages = 10**log_ages

        # Filter points within the specified ranges
        mask = (
            (ages >= age_range[0])
            & (ages <= age_range[1])
            & (fehs >= feh_range[0])
            & (fehs <= feh_range[1])
        )

        ages_filtered = ages[mask]
        fehs_filtered = fehs[mask]

        # Calculate KDE - swap for age on x-axis
        xy = np.vstack([ages_filtered, fehs_filtered])
        kde = gaussian_kde(xy)

        # Create grid for KDE evaluation - swap axes
        x_grid = np.linspace(age_range[0], age_range[1], 100)
        y_grid = np.linspace(feh_range[0], feh_range[1], 100)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Evaluate KDE on grid
        zz = kde(np.vstack([xx.ravel(), yy.ravel()]))
        zz = zz.reshape(xx.shape)

        # Plot KDE as contours with filled colors
        contour = axes[i].contourf(xx, yy, zz, levels=20, cmap="viridis", alpha=0.8)

        # Add scatter points with small size - swap axes
        axes[i].scatter(ages_filtered, fehs_filtered, s=0.3, color="k", alpha=0.1)

        # Set title and labels - swap axes labels
        axes[i].set_title(f"Bin: {bin_name}")
        axes[i].set_xlabel("Age (Gyr)")
        if i == 0:
            axes[i].set_ylabel("[Fe/H]")

        # Set axis ranges - swap ranges
        axes[i].set_xlim(age_range)
        axes[i].set_ylim(feh_range)
        if flip_age_axis:
            axes[i].invert_xaxis()  # Flip x-axis instead of y-axis

        # Add grid
        axes[i].grid(True, linestyle="--", alpha=0.5)

    # Add colorbar for the last plot
    cbar = fig.colorbar(contour, ax=axes[-1])
    cbar.set_label("Density")

    plt.suptitle("Age-Metallicity Relation Across Radial Bins", fontsize=16)
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_age_metallicity_heatmap(
    flow,
    scaler,
    n_samples=10000,
    save_path=None,
    age_range=(0, 20),
    feh_range=(-1.5, 0.5),
    nbins=(100, 100),
    figsize=(10, 8),
    flip_age_axis=True,
    bin_name=None,
):
    """
    Create a 2D histogram heatmap of Age vs. [Fe/H] using samples from the flow model.

    Parameters:
    -----------
    flow : Flow5D
        Trained normalizing flow model
    scaler : StandardScaler
        Scaler used to normalize the data
    n_samples : int
        Number of samples to draw
    save_path : str
        Path to save the figure
    age_range : tuple
        (min, max) for age range in plot
    feh_range : tuple
        (min, max) for [Fe/H] range in plot
    nbins : tuple
        Number of bins in each dimension (feh_bins, age_bins)
    figsize : tuple
        Figure size
    flip_age_axis : bool
        If True, plot age from high to low (oldest to youngest on the left)
    bin_name : str
        Name of the bin for title

    Returns:
    --------
    tuple
        (fig, ax) - Figure and axis objects
    """
    # Set device and evaluation mode
    device = next(flow.parameters()).device
    flow.eval()

    # Sample from the flow
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    # Inverse transform to get original scale
    samples_original = scaler.inverse_transform(samples)

    # Extract age and [Fe/H]
    log_ages = samples_original[:, 0]  # First dimension is log(age)
    fehs = samples_original[:, 1]  # Second dimension is [Fe/H]

    # Convert log age to linear age
    ages = 10**log_ages

    # Filter points within the specified ranges
    mask = (
        (ages >= age_range[0])
        & (ages <= age_range[1])
        & (fehs >= feh_range[0])
        & (fehs <= feh_range[1])
    )

    ages_filtered = ages[mask]
    fehs_filtered = fehs[mask]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create 2D histogram - swap axes for age on x-axis
    hist, xedges, yedges = np.histogram2d(
        ages_filtered, fehs_filtered, bins=nbins, range=[age_range, feh_range]
    )

    # Apply logarithmic scaling to better visualize the full range of counts
    # Add small constant to avoid log(0)
    hist_log = np.log1p(hist.T)  # log(1+x) and transpose for imshow

    # Plot heatmap - swap axes in extent
    extent = [age_range[0], age_range[1], feh_range[0], feh_range[1]]
    im = ax.imshow(
        hist_log,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log(1 + count)")

    # Set labels and title - swap axes labels
    ax.set_xlabel("Age (Gyr)")
    ax.set_ylabel("[Fe/H]")
    if bin_name:
        ax.set_title(f"Age-Metallicity Relation - {bin_name}")
    else:
        ax.set_title("Age-Metallicity Relation")

    # Set axis ranges - swap ranges
    ax.set_xlim(age_range)
    ax.set_ylim(feh_range)
    if flip_age_axis:
        ax.invert_xaxis()  # Flip x-axis to show oldest at left

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.5)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def generate_visualizations(
    flows_dict,
    scalers_dict,
    output_dir,
    n_samples=10000,
    age_range=(0, 20),
    feh_range=(-1.5, 0.5),
):
    """
    Generate multiple visualizations of age vs. [Fe/H] for all models.

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
    # Create output directories
    viz_dir = os.path.join(output_dir, "age_metallicity_viz")
    kde_dir = os.path.join(viz_dir, "kde")
    heatmap_dir = os.path.join(viz_dir, "heatmap")

    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(kde_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    # Define bin order to ensure correct ordering from inner to outer
    radial_bin_order = ["R0.0-6.0", "R6.0-8.0", "R8.0-10.0", "R10.0-15.0"]

    # Generate individual visualizations in the correct order
    for bin_name in radial_bin_order:
        if bin_name in flows_dict and bin_name in scalers_dict:
            flow = flows_dict[bin_name]
            scaler = scalers_dict[bin_name]
            print(f"Generating visualizations for {bin_name}...")

            # KDE plots
            plot_age_metallicity_kde(
                flow,
                scaler,
                n_samples=n_samples,
                save_path=os.path.join(kde_dir, f"{bin_name}_age_feh_kde.png"),
                age_range=age_range,
                feh_range=feh_range,
                flip_age_axis=True,
                bin_name=bin_name,
            )

            # Heatmap plots
            plot_age_metallicity_heatmap(
                flow,
                scaler,
                n_samples=n_samples,
                save_path=os.path.join(heatmap_dir, f"{bin_name}_age_feh_heatmap.png"),
                age_range=age_range,
                feh_range=feh_range,
                nbins=(100, 100),
                flip_age_axis=True,
                bin_name=bin_name,
            )
        else:
            print(f"Skipping bin {bin_name}: model or scaler not available")

    # Generate combined comparison plots
    plot_multiple_bin_kde(
        flows_dict,
        scalers_dict,
        n_samples=n_samples // 2,
        save_path=os.path.join(viz_dir, "all_bins_age_feh_kde.png"),
        age_range=age_range,
        feh_range=feh_range,
        flip_age_axis=True,
    )

    print(f"Visualizations saved to {viz_dir}")


def load_models(models_dir):
    """
    Load trained flow models from a directory.
    """
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    flows_dict = {}
    scalers_dict = {}

    # Define the order of radial bins
    # This will ensure bins are processed in the correct order from inner to outer
    radial_bin_order = ["R0.0-6.0", "R6.0-8.0", "R8.0-10.0", "R10.0-15.0"]
    
    # Create a set for faster lookups
    radial_bins_set = set(radial_bin_order)

    # Find all model files
    model_files = {}
    for filename in os.listdir(models_dir):
        if filename.endswith("_model.pt"):
            bin_name = filename.split("_model.pt")[0]
            # Ensure bin name follows our standard format and is one we're interested in
            if bin_name in radial_bins_set:
                model_path = os.path.join(models_dir, filename)
                model_files[bin_name] = model_path

    # Load models in the specified order
    for bin_name in radial_bin_order:
        if bin_name in model_files:
            model_path = model_files[bin_name]
            print(f"Loading model for bin {bin_name} from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)

            # Initialize flow model with matching parameters from the model
            if "model_config" in checkpoint:
                # If we saved the configuration
                config = checkpoint["model_config"] 
                flow = Flow5D(
                    n_transforms=config.get("n_transforms", 12),
                    hidden_dims=config.get("hidden_dims", [128, 128]),
                    num_bins=config.get("num_bins", 24)
                ).to(device)
            else:
                # Use default configuration matching what we used in training
                flow = Flow5D(
                    n_transforms=12,
                    hidden_dims=[128, 128],
                    num_bins=24
                ).to(device)

            try:
                # Check which format the model was saved in
                if "flow_state" in checkpoint:
                    flow.load_state_dict(checkpoint["flow_state"])
                elif "model_state" in checkpoint:
                    flow.load_state_dict(checkpoint["model_state"])
                else:
                    print(f"Warning: Unknown model format in {model_path}")
                    continue
                
                flow.eval()
                flows_dict[bin_name] = flow
                scalers_dict[bin_name] = checkpoint["scaler"]
                print(f"Successfully loaded model for {bin_name}")
                
            except RuntimeError as e:
                print(f"Error loading model for {bin_name}: {e}")
                print("The model architecture in the file may not match the current Flow5D implementation.")
                print("Skipping this model.")
                continue

    print(f"Loaded {len(flows_dict)} models")
    return flows_dict, scalers_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate age-metallicity visualizations from trained models"
    )
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="outputs/models", 
        help="Directory containing model files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/visualizations", 
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=10000, 
        help="Number of samples per model"
    )
    parser.add_argument(
        "--min_age", 
        type=float, 
        default=0, 
        help="Minimum age to plot"
    )
    parser.add_argument(
        "--max_age", 
        type=float, 
        default=20, 
        help="Maximum age to plot"
    )
    parser.add_argument(
        "--min_feh", 
        type=float, 
        default=-1.5, 
        help="Minimum [Fe/H] to plot"
    )
    parser.add_argument(
        "--max_feh", 
        type=float, 
        default=0.5, 
        help="Maximum [Fe/H] to plot"
    )

    args = parser.parse_args()

    # Ensure models directory exists
    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory '{args.models_dir}' does not exist.")
        print("Please specify the correct directory with --models_dir or run training first.")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    flows_dict, scalers_dict = load_models(args.models_dir)
    
    if not flows_dict:
        print("No valid models found. Please check the models directory.")
        exit(1)
        
    print(f"Successfully loaded {len(flows_dict)} models: {list(flows_dict.keys())}")

    # Generate visualizations
    generate_visualizations(
        flows_dict,
        scalers_dict,
        args.output_dir,
        n_samples=args.samples,
        age_range=(args.min_age, args.max_age),
        feh_range=(args.min_feh, args.max_feh),
    )
