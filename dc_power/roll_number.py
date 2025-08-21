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


    # … STUDENT IMPLEMENTATION HERE …
    # Solve the DC Power Flow problem here.
    # voltage_angle must be a 1D numpy array or list, length = bus_df.shape[0]
    voltage_angle = [] 
    n = bus_df.shape[0]           
    B = np.zeros((n, n))
    print(bus_df.columns)
    print(bus_df.head())
    for _, branch in branch_df.iterrows():
      i = int(branch["fbus"]) - 1
      j = int(branch["tbus"]) - 1
      x = branch["x"]
      b = 1 / x

      B[i, i] += b
      B[j, j] += b
      B[i, j] -= b
      B[j, i] -= b
    print(gen_df.columns)
    print(gen_df.head())
    P = np.zeros(n)
    i=0
    check = int(gen_df[0]["bus"])
    for _, gen in gen_df.iterrows():
      P[i] += gen["Pg"]
      if(int(bus["bus"])!=check):
        i+=1
      check=int(gen["bus"])
    i=0
    for _, bus in bus_df.iterrows():
      P[i] -= bus["Pd"]
      i+=1

    slack = 0

    B_reduced = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    P_reduced = np.delete(P, slack)

    theta_reduced = np.linalg.solve(B_reduced, P_reduced)

    theta = np.zeros(n)
    theta[slack] = 0
    theta[np.arange(n) != slack] = theta_reduced
    voltage_angle = theta.tolist()
    # Do not change the following line. Any edit below will result in Penalty.
    t1 = time.perf_counter()
    return {"angles": voltage_angle, "time": t1 - t0}

# == STANDING INSTRUCTION ==
if __name__ == "__main__":
    import sys, json
    case_file = sys.argv[1]
    result = test(case_file)
    sys.stdout.write(json.dumps(result))
