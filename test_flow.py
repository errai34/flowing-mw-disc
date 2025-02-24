import torch
from normflows import Flow5D

def test_flow():
    # Create a flow model
    flow = Flow5D(n_transforms=2)  # Using fewer transforms for testing
    
    # Create some dummy data (batch of 10 samples, 5 dimensions each)
    x = torch.randn(10, 5)
    
    try:
        # Test log probability computation
        log_prob = flow.log_prob(x)
        print("Log probability shape:", log_prob.shape)
        print("Log probability values:", log_prob)
        
        # Test sampling
        samples = flow.sample(10)
        print("\nSamples shape:", samples.shape)
        print("Sample values:", samples)
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_flow()
