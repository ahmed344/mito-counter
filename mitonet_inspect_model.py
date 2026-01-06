import torch
import os

model_path = 'MitoNet_v1.pth'
if not os.path.exists(model_path):
    print("Model not found")
    exit()

try:
    print(f"Loading {model_path} as TorchScript...")
    # Load as a ScriptModule
    model = torch.jit.load(model_path)
    print("Model loaded successfully!")
    
    # Inspect named parameters (weights)
    print("\n--- Model Parameters (Top 10) ---")
    count = 0
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
        count += 1
        if count >= 20:
            break
            
    # Check for specific layers by iterating through named modules/parameters
    print("\n--- Architecture Check ---")
    all_names = [name for name, _ in model.named_parameters()]
    
    has_pr = any('semantic_pr' in k for k in all_names)
    print(f"Has PointRend keys (semantic_pr): {has_pr}")
    
    has_bc = any('boundary_head' in k for k in all_names)
    print(f"Has Boundary keys (boundary_head): {has_bc}")

    # You can also print the code/graph
    # print(model.code)

except Exception as e:
    print(f"Error: {e}")