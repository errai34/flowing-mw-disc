import torch
import sys

def inspect_model(model_path):
    print(f"Inspecting model: {model_path}")
    
    # Load the model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Successfully loaded checkpoint")
        
        # Print the keys in the checkpoint
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"- {key}")
        
        # If there's a scaler, print its info
        if 'scaler' in checkpoint:
            print("\nScaler is present in the checkpoint")
        
        # Check model state
        if 'model_state' in checkpoint:
            print("\nModel state details:")
            state_dict = checkpoint['model_state']
            
            # Count total parameters
            param_count = sum(p.numel() for p in checkpoint['model_state'].values())
            print(f"Total parameters: {param_count:,}")
            
            # Print some key names to understand structure
            print("\nSample of model state keys:")
            for i, key in enumerate(state_dict.keys()):
                if i < 10:  # Print first 10 keys
                    print(f"- {key}")
                else:
                    break
                    
            if 'flow_state' in checkpoint:
                print("\nFlow state is present in the checkpoint")
                
        # Check if there's configuration info
        if 'model_config' in checkpoint:
            print("\nModel configuration:")
            for key, value in checkpoint['model_config'].items():
                print(f"- {key}: {value}")
        else:
            print("\nNo model configuration found in checkpoint")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        inspect_model(model_path)
    else:
        print("Please provide a model path")
        print("Usage: python inspect_model.py path/to/model.pt")