import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KernelDensity


def generate_gaussian_mixture_data(n_samples=50000, random_seed=42):
    """
    Generate 2D mixture of Gaussians data as used in the Density Deconvolution paper.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (v, w) - Arrays of shape (n_samples, 2) for clean and noisy data
    """
    np.random.seed(random_seed)

    # Equal mixture weights for 3 components
    n_per_component = n_samples // 3
    # Ensure total samples is exactly n_samples
    n_samples_actual = n_per_component * 3

    # Component means
    means = [
        np.array([-2.0, 0.0]),  # Component 1
        np.array([0.0, -2.0]),  # Component 2
        np.array([0.0, 2.0]),  # Component 3
    ]

    # Component covariances
    covariances = [
        np.array([[0.3**2, 0.0], [0.0, 1.0]]),  # Component 1
        np.array([[1.0, 0.0], [0.0, 0.3**2]]),  # Component 2
        np.array([[1.0, 0.0], [0.0, 0.3**2]]),  # Component 3
    ]

    # Noise covariance
    noise_cov = np.array([[0.1, 0.0], [0.0, 1.0]])

    # Generate samples from each component
    samples = []
    for i in range(3):
        component_samples = np.random.multivariate_normal(
            mean=means[i], cov=covariances[i], size=n_per_component
        )
        samples.append(component_samples)

    # Concatenate samples from all components
    v = np.vstack(samples)

    # Add noise to create the observed data w
    noise = np.random.multivariate_normal(
        mean=np.zeros(2), cov=noise_cov, size=n_samples_actual
    )
    w = v + noise

    return v, w


def visualize_data(v, w, output_dir="mixture_gaussian_plots"):
    """
    Create visualizations of the clean and noisy data.

    Parameters:
    -----------
    v : np.ndarray
        Clean data array of shape (n_samples, 2)
    w : np.ndarray
        Noisy data array of shape (n_samples, 2)
    output_dir : str
        Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot clean data
    sns.histplot(
        x=v[:, 0],
        y=v[:, 1],
        bins=50,
        pthresh=0.02,
        cmap="Blues",
        ax=ax1,
        cbar=True,
        stat="density",
    )

    # Add contour lines for clean data
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    values = np.vstack([v[:, 0], v[:, 1]]).T
    kernel = KernelDensity(bandwidth=0.2).fit(values)
    zz = np.exp(kernel.score_samples(positions))
    zz = np.reshape(zz, xx.shape)
    ax1.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)

    ax1.set_title("Latent Data p(v)")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")

    # Plot noisy data
    sns.histplot(
        x=w[:, 0],
        y=w[:, 1],
        bins=50,
        pthresh=0.02,
        cmap="Reds",
        ax=ax2,
        cbar=True,
        stat="density",
    )

    # Add contour lines for noisy data
    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    values = np.vstack([w[:, 0], w[:, 1]]).T
    kernel = KernelDensity(bandwidth=0.2).fit(values)
    zz = np.exp(kernel.score_samples(positions))
    zz = np.reshape(zz, xx.shape)
    ax2.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)

    ax2.set_title("Observed Data p(w)")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gaussian_mixture_data.png"), dpi=300)
    plt.close()

    # Save data as numpy arrays
    np.save(os.path.join(output_dir, "clean_data.npy"), v)
    np.save(os.path.join(output_dir, "noisy_data.npy"), w)

    print(f"Data visualization saved to {output_dir}/")
    print(f"Data saved as numpy arrays in {output_dir}/")


def generate_train_test_split(
    v, w, test_fraction=0.2, validation_fraction=0.05, random_seed=42
):
    """
    Split data into training, validation, and test sets.

    Parameters:
    -----------
    v : np.ndarray
        Clean data array
    w : np.ndarray
        Noisy data array
    test_fraction : float
        Fraction of data to use for testing
    validation_fraction : float
        Fraction of data to use for validation
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with train, validation, and test data
    """
    np.random.seed(random_seed)

    n_samples = v.shape[0]
    indices = np.random.permutation(n_samples)

    test_size = int(n_samples * test_fraction)
    val_size = int(n_samples * validation_fraction)
    train_size = n_samples - test_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    data_splits = {
        "train": {"v": v[train_indices], "w": w[train_indices]},
        "val": {"v": v[val_indices], "w": w[val_indices]},
        "test": {"v": v[test_indices], "w": w[test_indices]},
    }

    return data_splits


def save_data_splits(data_splits, output_dir="mixture_gaussian_data"):
    """
    Save data splits to disk.

    Parameters:
    -----------
    data_splits : dict
        Dictionary with train, validation, and test data
    output_dir : str
        Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in data_splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, "v.npy"), split_data["v"])
        np.save(os.path.join(split_dir, "w.npy"), split_data["w"])

    print(f"Data splits saved to {output_dir}/")


def get_exact_posterior(w, prior_means, prior_covs, prior_weights, noise_cov):
    """
    Calculate the exact posterior p(v|w) for a Gaussian mixture model.

    For each observation w, the posterior is also a Gaussian mixture.

    Parameters:
    -----------
    w : np.ndarray
        A single observed point w
    prior_means : list
        List of prior component means
    prior_covs : list
        List of prior component covariances
    prior_weights : list
        List of prior component weights
    noise_cov : np.ndarray
        Noise covariance matrix

    Returns:
    --------
    tuple
        (posterior_means, posterior_covs, posterior_weights)
    """
    posterior_means = []
    posterior_covs = []
    posterior_weights = []

    for mean, cov, weight in zip(prior_means, prior_covs, prior_weights):
        # Combined covariance
        combined_cov = cov + noise_cov
        combined_cov_inv = np.linalg.inv(combined_cov)

        # Posterior covariance
        posterior_cov = np.linalg.inv(np.linalg.inv(cov) + np.linalg.inv(noise_cov))

        # Posterior mean
        posterior_mean = posterior_cov @ (
            np.linalg.inv(cov) @ mean + np.linalg.inv(noise_cov) @ w
        )

        # Component likelihood
        dist = w - mean
        exponent = -0.5 * dist.T @ combined_cov_inv @ dist
        det_term = np.sqrt(np.linalg.det(2 * np.pi * combined_cov))
        component_likelihood = np.exp(exponent) / det_term

        posterior_means.append(posterior_mean)
        posterior_covs.append(posterior_cov)
        posterior_weights.append(weight * component_likelihood)

    # Normalize weights
    posterior_weights = np.array(posterior_weights)
    posterior_weights = posterior_weights / np.sum(posterior_weights)

    return posterior_means, posterior_covs, posterior_weights


def visualize_posterior(w_point, output_dir="mixture_gaussian_plots"):
    """
    Visualize the exact posterior for a given observation.

    Parameters:
    -----------
    w_point : np.ndarray
        A single observed point w
    output_dir : str
        Directory to save visualization
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define the prior parameters
    prior_means = [
        np.array([-2.0, 0.0]),  # Component 1
        np.array([0.0, -2.0]),  # Component 2
        np.array([0.0, 2.0]),  # Component 3
    ]

    prior_covs = [
        np.array([[0.3**2, 0.0], [0.0, 1.0]]),  # Component 1
        np.array([[1.0, 0.0], [0.0, 0.3**2]]),  # Component 2
        np.array([[1.0, 0.0], [0.0, 0.3**2]]),  # Component 3
    ]

    prior_weights = [1 / 3, 1 / 3, 1 / 3]  # Equal weights

    noise_cov = np.array([[0.1, 0.0], [0.0, 1.0]])

    # Calculate the exact posterior
    posterior_means, posterior_covs, posterior_weights = get_exact_posterior(
        w_point, prior_means, prior_covs, prior_weights, noise_cov
    )

    # Create a grid of points
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.flatten(), Y.flatten()])

    # Evaluate the posterior density on the grid
    posterior_density = np.zeros(grid.shape[0])

    for mean, cov, weight in zip(posterior_means, posterior_covs, posterior_weights):
        # Calculate the Gaussian density for this component
        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        for i in range(grid.shape[0]):
            diff = grid[i] - mean
            exponent = -0.5 * diff @ cov_inv @ diff
            density = np.exp(exponent) / np.sqrt((2 * np.pi) ** 2 * det_cov)
            posterior_density[i] += weight * density

    # Reshape for plotting
    posterior_density = posterior_density.reshape(X.shape)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot the posterior density
    plt.contourf(X, Y, posterior_density, levels=50, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Posterior Density")

    # Plot the observation point
    plt.scatter(
        w_point[0], w_point[1], color="red", s=100, marker="x", label="Observation w"
    )

    # Plot the posterior means
    for i, (mean, weight) in enumerate(zip(posterior_means, posterior_weights)):
        plt.scatter(
            mean[0],
            mean[1],
            color="black",
            s=100 * weight,
            marker="+",
            label=f"Component {i+1} Mean (weight={weight:.2f})",
        )

    # Plot the 1-sigma level of the noise around the observation
    noise_1sigma = np.sqrt(np.diag(noise_cov))
    ellipse = plt.matplotlib.patches.Ellipse(
        xy=w_point,
        width=2 * noise_1sigma[0],
        height=2 * noise_1sigma[1],
        fill=False,
        edgecolor="red",
        linestyle="--",
        label="1-σ Noise Level",
    )
    plt.gca().add_patch(ellipse)

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Exact Posterior p(v|w)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "exact_posterior.png"), dpi=300)
    plt.close()

    print(f"Posterior visualization saved to {output_dir}/")


if __name__ == "__main__":
    # Generate data
    v, w = generate_gaussian_mixture_data(n_samples=50000)

    # Create visualizations
    visualize_data(v, w)

    # Split data
    data_splits = generate_train_test_split(v, w)

    # Save data splits
    save_data_splits(data_splits)

    # Visualize posterior for a specific point
    # Choose a point near the middle to see all three components
    example_w = np.array([-0.5, 0.0])
    visualize_posterior(example_w)
