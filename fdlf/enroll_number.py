"""
Fast Decoupled Load Flow (FDLF) Boilerplate Code-- Commets are GPT Generated

Standing Instructions:
- Do NOT change the function name or its signature.
- Implement the FDLF method inside run_fdlf().
- Your code must use pandapower.makeYbus() for Y-bus construction.
- Your function should return a dictionary with keys:
    {
      "bus_angles": [...],    # list of bus voltage angles in degrees
      "bus_voltages": [...],  # list of bus magnitudes (p.u.)
      "iterations": int       # number of iterations taken
    }
- Do not hardcode bus numbers or results.
- Your code should converge for case9.m, case14.m, case30.m.
- Tolerance = 1e-6, Maximum iterations = 50.
"""

import time
import pandapower
from pandapower.pypower import loadcase
from pandapower.pypower.makeYbus import makeYbus

def run_fdlf(case_file: str, tol: float = 1e-6, max_iter: int = 1) -> dict:
    """
    Implement the Fast Decoupled Load Flow (FDLF) solver.

    Args:
        case_file (str): Path to MATPOWER case file (.m format).
        tol (float): Convergence tolerance (default: 1e-6).
        max_iter (int): Maximum iterations (default: 1).
    
    Returns:
        dict: {
            "bus_angles": [...],
            "bus_voltages": [...],
            "iterations": int,
            "time_total": float,      # total runtime (seconds)
            "time_solve": float       # time spent in backsolve/linear system solve
        }
    """
    # Start total runtime timer
    t0_total = time.time() 

    # Load MATPOWER case
    ppc = loadcase(case_file)

    # Build Y-bus using pandapower
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    # -----------------------------------------------------------
    # TODO: Implement FDLF iteration loop here
    # You should:
    #   1. Initialize bus angles & voltages
    #   2. Compute P and Q mismatches
    #   3. Use the decoupled B' and B'' matrices
    #   4. Solve the linear system
    #
    # NOTE: Place your matrix inversion / linear solve
    #       between t0_solve and t1_solve timers.
    # -----------------------------------------------------------

    # solve time:
    t0_solve = time.time() # Do not change this line

    # ---------------- Only Your LINEAR SOLVE GOES HERE ----------------
    # If you put anything else here, you will get ZERO points!
    # ---------------------------------------------------------------

    t1_solve = time.time() # Do not change this line
    time_solve = t1_solve - t0_solve # Do not change this line

    # End total runtime timer
    t1_total = time.time()# Do not change this line
    time_total = t1_total - t0_total # Do not change this line

    # -----------------------------------------------------------
    # Dummy return values for structure demonstration only
    # Replace with actual FDLF results
    # -----------------------------------------------------------
    return {
        "bus_angles": [0.0] * len(ppc["bus"]),
        "bus_voltages": [1.0] * len(ppc["bus"]),
        "iterations": 0,
        "time_total": time_total,
        "time_solve": time_solve
    }


if __name__ == "__main__":
    # Minimal self-test
    case_file = "case9.m"
    try:
        res = run_fdlf(case_file)
        print("Self-test run on", case_file)
        print(res)
    except Exception as e:
        print("Something went wrong in self-test:", e)
