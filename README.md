# flowing-mw-disc

This repository contains code for training and visualizing normalizing flow models for the Milky Way disc stellar populations.

## Project Structure

- `outputs/models/` - Central storage location for trained models
- `outputs/plots/` - Central storage location for visualization plots
- `outputs/run_YYYYMMDD_HHMMSS/` - Individual training run directories (contain logs, configs)
- `outputs/visualizations/` - Advanced visualizations generated from trained models

## Usage

### Training a Model

Train a model for a specific radial bin:

```bash
python run_training.py --data_path /path/to/data --bin_name "R8-10"
```

This will:
1. Create a timestamped run directory under `outputs/run_YYYYMMDD_HHMMSS/`
2. Save the trained model to the central `outputs/models/` directory
3. Generate basic visualizations in `outputs/plots/`

### Generating Visualizations

Generate detailed age-metallicity visualizations for all trained models:

```bash
python generate_density_plots.py
```

By default, this looks for models in `outputs/models/` and saves visualizations to `outputs/visualizations/`.

## Bin Naming Convention

Models are organized by galactic radial bins with a standardized naming convention:
- `R0.0-6.0` - Inner disc (0 to 6 kpc)
- `R6.0-8.0` - Inner-middle disc (6 to 8 kpc)
- `R8.0-10.0` - Solar neighborhood (8 to 10 kpc)
- `R10.0-15.0` - Outer disc (10 to 15 kpc)