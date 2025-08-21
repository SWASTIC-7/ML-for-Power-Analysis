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

xp = np
try:
    import cupy as cp
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp
            print("GPU acceleration enabled using CuPy.")
        else:
            print("No CUDA devices available. Running on CPU with NumPy.")
    except cp.cuda.runtime.CUDARuntimeError:
        # Handles cases where driver/runtime mismatch occurs
        print("CUDA not usable. Falling back to CPU with NumPy.")
        xp = np
except ImportError:
    print("CuPy not found. Running on CPU with NumPy.")

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
    
    bus_numbers = sorted(bus_df['bus_i'].unique())
    n = len(bus_numbers)
    #creating mapping for index fixing 
    bus_to_idx = {bus_num: idx for idx, bus_num in enumerate(bus_numbers)}
    idx_to_bus = {idx: bus_num for idx, bus_num in enumerate(bus_numbers)}
    
    
    
    B = xp.zeros((n, n))
    
    for _, branch in branch_df.iterrows():
        fbus_num = int(branch["fbus"])
        tbus_num = int(branch["tbus"])
        
        i = bus_to_idx[fbus_num]
        j = bus_to_idx[tbus_num]
        
        x = branch["x"]
        if x != 0:  
            b = 1.0 / x
            
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b
    
    P = xp.zeros(n)
    
    for _, gen in gen_df.iterrows():
        bus_num = int(gen["bus"])
        if bus_num in bus_to_idx:  
            idx = bus_to_idx[bus_num]
            P[idx] += gen["Pg"] / 100.0  
    
    for _, bus in bus_df.iterrows():
        bus_num = int(bus["bus_i"])
        idx = bus_to_idx[bus_num]
        P[idx] -= bus["Pd"] / 100.0  
    
    if xp is cp:
        B = cp.asarray(B)
        P = cp.asarray(P)
    
    slack_idx = 0
    for _, bus in bus_df.iterrows():
        if bus["type"] == 3:  
            slack_bus_num = int(bus["bus_i"])
            if slack_bus_num in bus_to_idx:
                slack_idx = bus_to_idx[slack_bus_num]
                break
    
    
    B_reduced = xp.delete(xp.delete(B, slack_idx, axis=0), slack_idx, axis=1)
    P_reduced = xp.delete(P, slack_idx)
    
    try:
        theta_reduced = xp.linalg.solve(B_reduced, P_reduced)
        
        theta = xp.zeros(n)
        theta[slack_idx] = 0.0
        
        reduced_indices = [i for i in range(n) if i != slack_idx]
        for i, reduced_idx in enumerate(reduced_indices):
            theta[reduced_idx] = theta_reduced[i]
            
    except xp.linalg.LinAlgError:
        theta_reduced = xp.linalg.pinv(B_reduced) @ P_reduced
        
        theta = xp.zeros(n)
        theta[slack_idx] = 0.0
        
        reduced_indices = [i for i in range(n) if i != slack_idx]
        for i, reduced_idx in enumerate(reduced_indices):
            theta[reduced_idx] = theta_reduced[i]
    
    if xp is cp:
        voltage_angle = theta.get().tolist()
    else:
        voltage_angle = theta.tolist() 
    
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