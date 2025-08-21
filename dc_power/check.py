#!/usr/bin/env python3
"""
DC Power Flow template for EET 109 – Autumn 2025
Roll No: <replace_with_your_rollno>

Note: Keep load_pglib_opf.py in the SAME folder as this script,
      then import as shown below. Do not rename or move it.
"""

## Do Not Change the above this. 

import time
import numpy as np
from load_pglib_opf import load_pglib_opf

# GPU support check and imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) detected - will use GPU acceleration")
except ImportError:
    try:
        import torch
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            print("GPU (PyTorch CUDA) detected - will use GPU acceleration")
        else:
            GPU_AVAILABLE = False
            print("PyTorch available but no CUDA GPU detected - using CPU")
    except ImportError:
        GPU_AVAILABLE = False
        print("No GPU libraries detected - using CPU")

def test(case_path):
    """
    Runs DCPF on the given PGLib case file.
    Returns:
      {
        "angles": [θ1, θ2, ...]    # float radians for each bus, in bus order,
        "time": <float>            # elapsed seconds (DCPF solve only)
      }
    """
    t0 = time.perf_counter()
    bus_df, gen_df, branch_df = load_pglib_opf(case_path)
    # Do not change the above line.

    # === STUDENT IMPLEMENTATION STARTS HERE ===
    # Solve the DC Power Flow problem here.
    
    # Get the number of buses
    bus_numbers = sorted(bus_df['bus_i'].unique())
    n = len(bus_numbers)
    
    # Create mapping from bus number to index
    bus_to_idx = {bus_num: idx for idx, bus_num in enumerate(bus_numbers)}
    idx_to_bus = {idx: bus_num for idx, bus_num in enumerate(bus_numbers)}
    
    print(f"Number of buses: {n}")
    print(f"Bus numbers: {bus_numbers[:10]}...")  # Show first 10
    
    # Choose array library based on GPU availability
    if GPU_AVAILABLE:
        try:
            # Try CuPy first (more NumPy-like)
            import cupy as cp
            xp = cp
            device = "GPU (CuPy)"
        except ImportError:
            # Fall back to PyTorch
            import torch
            xp = torch
            device = "GPU (PyTorch)"
            torch.cuda.empty_cache()  # Clear GPU memory
    else:
        xp = np
        device = "CPU"
    
    print(f"Using: {device}")
    
    # Build admittance matrix B
    if GPU_AVAILABLE and 'cupy' in str(xp):
        # Using CuPy
        B = xp.zeros((n, n), dtype=xp.float64)
    elif GPU_AVAILABLE and 'torch' in str(xp):
        # Using PyTorch
        B = xp.zeros((n, n), dtype=xp.float64, device='cuda')
    else:
        # Using NumPy
        B = xp.zeros((n, n), dtype=np.float64)
    
    for _, branch in branch_df.iterrows():
        fbus_num = int(branch["fbus"])
        tbus_num = int(branch["tbus"])
        
        # Map to indices
        i = bus_to_idx[fbus_num]
        j = bus_to_idx[tbus_num]
        
        x = branch["x"]
        if x != 0:  # Avoid division by zero
            b = 1.0 / x
            
            # Build B matrix
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b
    
    # Calculate net power injection for each bus
    if GPU_AVAILABLE and 'cupy' in str(xp):
        P = xp.zeros(n, dtype=xp.float64)
    elif GPU_AVAILABLE and 'torch' in str(xp):
        P = xp.zeros(n, dtype=xp.float64, device='cuda')
    else:
        P = xp.zeros(n, dtype=np.float64)
    
    # Add generation (positive injection)
    for _, gen in gen_df.iterrows():
        bus_num = int(gen["bus"])
        if bus_num in bus_to_idx:  # Check if bus exists
            idx = bus_to_idx[bus_num]
            P[idx] += gen["Pg"] / 100.0  # Convert to per unit (base 100 MVA)
    
    # Subtract load (negative injection)
    for _, bus in bus_df.iterrows():
        bus_num = int(bus["bus_i"])
        idx = bus_to_idx[bus_num]
        P[idx] -= bus["Pd"] / 100.0  # Convert to per unit (base 100 MVA)
    
    # Convert to CPU for printing if on GPU
    if GPU_AVAILABLE:
        if 'cupy' in str(xp):
            print(f"Power injections (first 10): {xp.asnumpy(P[:10])}")
        elif 'torch' in str(xp):
            print(f"Power injections (first 10): {P[:10].cpu().numpy()}")
    else:
        print(f"Power injections (first 10): {P[:10]}")
    
    # Find slack bus (typically bus with type 3, or first bus if none)
    slack_idx = 0
    for _, bus in bus_df.iterrows():
        if bus["type"] == 3:  # Reference/slack bus
            slack_bus_num = int(bus["bus_i"])
            if slack_bus_num in bus_to_idx:
                slack_idx = bus_to_idx[slack_bus_num]
                break
    
    print(f"Slack bus index: {slack_idx}, bus number: {idx_to_bus[slack_idx]}")
    
    # Remove slack bus from B matrix and P vector
    if GPU_AVAILABLE and 'torch' in str(xp):
        # PyTorch approach
        indices = list(range(n))
        indices.remove(slack_idx)
        indices_tensor = xp.tensor(indices, device='cuda')
        B_reduced = B[indices_tensor][:, indices_tensor]
        P_reduced = P[indices_tensor]
    else:
        # CuPy/NumPy approach
        indices = [i for i in range(n) if i != slack_idx]
        B_reduced = B[np.ix_(indices, indices)]
        P_reduced = P[indices]
    
    # Solve the linear system
    try:
        if GPU_AVAILABLE and 'cupy' in str(xp):
            # CuPy solve
            theta_reduced = xp.linalg.solve(B_reduced, P_reduced)
        elif GPU_AVAILABLE and 'torch' in str(xp):
            # PyTorch solve
            theta_reduced = xp.linalg.solve(B_reduced, P_reduced)
        else:
            # NumPy solve
            theta_reduced = xp.linalg.solve(B_reduced, P_reduced)
        
        # Reconstruct full angle vector
        if GPU_AVAILABLE and 'cupy' in str(xp):
            theta = xp.zeros(n, dtype=xp.float64)
        elif GPU_AVAILABLE and 'torch' in str(xp):
            theta = xp.zeros(n, dtype=xp.float64, device='cuda')
        else:
            theta = xp.zeros(n, dtype=np.float64)
            
        theta[slack_idx] = 0.0  # Slack bus angle is reference (0)
        
        # Fill in other angles
        reduced_indices = [i for i in range(n) if i != slack_idx]
        for i, reduced_idx in enumerate(reduced_indices):
            theta[reduced_idx] = theta_reduced[i]
            
    except Exception as e:
        print(f"Warning: Linear solve failed ({e}), using pseudo-inverse")
        
        if GPU_AVAILABLE and 'cupy' in str(xp):
            theta_reduced = xp.linalg.pinv(B_reduced) @ P_reduced
            theta = xp.zeros(n, dtype=xp.float64)
        elif GPU_AVAILABLE and 'torch' in str(xp):
            theta_reduced = xp.linalg.pinv(B_reduced) @ P_reduced
            theta = xp.zeros(n, dtype=xp.float64, device='cuda')
        else:
            theta_reduced = xp.linalg.pinv(B_reduced) @ P_reduced
            theta = xp.zeros(n, dtype=np.float64)
            
        theta[slack_idx] = 0.0
        
        reduced_indices = [i for i in range(n) if i != slack_idx]
        for i, reduced_idx in enumerate(reduced_indices):
            theta[reduced_idx] = theta_reduced[i]
    
    # Convert result back to CPU/NumPy for output
    if GPU_AVAILABLE:
        if 'cupy' in str(xp):
            voltage_angle = xp.asnumpy(theta).tolist()
        elif 'torch' in str(xp):
            voltage_angle = theta.cpu().numpy().tolist()
    else:
        voltage_angle = theta.tolist()
    
    # Print first few angles for verification
    print(f"Voltage angles (first 10): {voltage_angle[:10]}")
    
    # === STUDENT IMPLEMENTATION ENDS HERE ===

    # Do not change the following line. Any edit below will result in Penalty.
    t1 = time.perf_counter()
    return {"angles": voltage_angle, "time": t1 - t0}

# == STANDING INSTRUCTION ==
if __name__ == "__main__":
    import sys, json
    case_file = sys.argv[1]
    result = test(case_file)
    sys.stdout.write(json.dumps(result))