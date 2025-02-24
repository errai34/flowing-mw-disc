import torch
import torch.nn as nn
from uncertainty import compute_log_noise_pdf, uncertainty_aware_elbo
from flow_model import Flow5D, Flow3D

# Create a mock Flow class for testing
class MockFlow(nn.Module):
    """Mock Flow class for testing uncertainty functions"""
    
    def __init__(self):
        super().__init__()
        # Add some dummy parameters to make this a proper nn.Module
        self.dummy_param = nn.Parameter(torch.randn(1))
    
    def log_prob(self, x):
        """Return dummy log probabilities for testing"""
        # Return a random but consistent log probability for each sample
        batch_size = x.shape[0]
        # Generate deterministic but varying log probs based on the input
        return -torch.sum(x**2, dim=1) / 10.0

def test_compute_log_noise_pdf():
    # Create sample observed data, latent data, and uncertainties
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [2.0, 3.0, 4.0, 5.0, 6.0]])
    x = torch.tensor([[0.9, 2.1, 3.2, 3.8, 5.1],
                      [2.1, 2.9, 4.1, 5.2, 5.7]])
    e = torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.3],
                      [0.2, 0.1, 0.1, 0.3, 0.2]])
    
    try:
        # Test log noise PDF computation
        log_prob = compute_log_noise_pdf(w, x, e)
        print("Log noise PDF shape:", log_prob.shape)
        print("Log noise PDF values:", log_prob)
        
        # Test with small uncertainties
        e_small = torch.ones_like(e) * 1e-10
        log_prob_small_e = compute_log_noise_pdf(w, x, e_small)
        print("\nLog noise PDF with small uncertainties:", log_prob_small_e)
        
        print("\ncompute_log_noise_pdf test passed successfully!")
    except Exception as e:
        print(f"Error in compute_log_noise_pdf test: {str(e)}")

def test_uncertainty_aware_elbo():
    # Create a mock flow model
    flow = MockFlow()
    
    # Test with 5D data
    print("Testing with 5D data:")
    observed_data_5d = torch.randn(8, 5)  # 8 samples, 5 dimensions
    uncertainties_5d = torch.abs(torch.randn(8, 5)) * 0.1  # Random positive uncertainties
    
    try:
        # Test ELBO computation with different K values
        elbo_k5 = uncertainty_aware_elbo(flow, observed_data_5d, uncertainties_5d, K=5)
        print("\nELBO with K=5:", elbo_k5.item())
        
        elbo_k10 = uncertainty_aware_elbo(flow, observed_data_5d, uncertainties_5d, K=10)
        print("ELBO with K=10:", elbo_k10.item())
        
        # Test with very small uncertainties
        small_uncertainties = torch.ones_like(uncertainties_5d) * 1e-4
        elbo_small_e = uncertainty_aware_elbo(flow, observed_data_5d, small_uncertainties, K=5)
        print("ELBO with small uncertainties:", elbo_small_e.item())
        
        # Test with 3D data
        print("\nTesting with 3D data:")
        observed_data_3d = torch.randn(8, 3)  # 8 samples, 3 dimensions
        uncertainties_3d = torch.abs(torch.randn(8, 3)) * 0.1  # Random positive uncertainties
        
        elbo_3d = uncertainty_aware_elbo(flow, observed_data_3d, uncertainties_3d, K=5)
        print("ELBO with 3D data:", elbo_3d.item())
        
        print("\nuncertainty_aware_elbo test passed successfully!")
    except Exception as e:
        print(f"Error in uncertainty_aware_elbo test: {str(e)}")

if __name__ == "__main__":
    print("Testing compute_log_noise_pdf function...")
    test_compute_log_noise_pdf()
    
    print("\nTesting uncertainty_aware_elbo function...")
    test_uncertainty_aware_elbo()
    
    print("\nAll uncertainty tests completed!")