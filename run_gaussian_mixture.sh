#!/bin/bash

# First make sure we generate the Gaussian mixture data if it doesn't exist
if [ ! -d "mixture_gaussian_data" ]; then
    echo "Generating Gaussian mixture data..."
    python src/generate_mixture_gaussians.py
fi

# Install required packages if missing
echo "Checking dependencies..."
pip install scikit-learn seaborn nflows tqdm matplotlib --quiet

# Run the deconvolution model on the Gaussian mixture data
echo "Running deconvolution on Gaussian mixture data..."
python src/deconvolution_flow.py \
    --dataset_type gaussian_mixture \
    --data_path mixture_gaussian_data \
    --use_known_noise \
    --use_clean_for_pretraining \
    --output_dir results_gaussian_mixture \
    --pretraining_epochs 15 \
    --epochs 10 \
    --batch_size 128 \
    --n_transforms 4 \
    --recognition_n_transforms 6 \
    --hidden_dim 64 \
    --num_bins 16 \
    --mc_samples 5 \
    --use_neural_spline \
    --use_enhanced_recognition \
    --temperature_annealing \
    --n_samples 10000

echo "Deconvolution complete! Check results in results_gaussian_mixture/ directory."
