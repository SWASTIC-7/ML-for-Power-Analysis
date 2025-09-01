# import pandapower.networks as pn
# from pandapower.converter import to_ppc
# from pypower.makeYbus import makeYbus

# def load_case(case_load: str):
#     cases = {
#         "ieee9": pn.case9,
#         "ieee14": pn.case14,
#         "ieee30": pn.case_ieee30,
#     }
#     if case_load not in cases:
#         raise ValueError(f"Unknown test case: {case_load}")
#     net = cases[case_load]()         # pandapower net
#     ppc= to_ppc(net, init="flat")             # convert to ppc dict
#     return ppc


# # -------------------------------------------------------
# # FDLF runner
# # -------------------------------------------------------
# def run_fdlf(case_load: str, tol=1e-6, max_iter=50):
#     """Run FDLF on a named pandapower test case."""
#     ppc = load_case(case_load)

#     # --- Example setup (your FDLF code goes here) ---
#     Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
#     # Run your solver with Ybus, ppc["gen"], ppc["bus"], etc.
#     # Return results
#     return {"case": case_load, "Ybus_shape": Ybus.shape}

# # -------------------------------------------------------
# # Main test loop
# # -------------------------------------------------------
# if __name__ == "__main__":
#     cases = ["ieee9", "ieee14", "ieee30"]
#     for c in cases:
#         print("=" * 50)
#         print(f"Testing {c}")
#         print("=" * 50)
#         try:
#             res = run_fdlf(c, tol=1e-6, max_iter=50)
#             print(f"{c} finished successfully: {res}\n")
#         except Exception as e:
#             print(f"Error testing {c}: {e}")
#             import traceback; traceback.print_exc()




import os
import time
import pandapower.networks as pn
from pandapower.converter import to_ppc
from pandapower.pypower.makeYbus import makeYbus
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def load_case(case_load: str):
    """Load a pandapower network case and convert to ppc format."""
    cases = {
        "ieee9": pn.case9,
        "ieee14": pn.case14,
        "ieee30": pn.case_ieee30,
    }
    if case_load not in cases:
        raise ValueError(f"Unknown test case: {case_load}. Available cases: {list(cases.keys())}")
    
    net = cases[case_load]()        # pandapower net
    ppc = to_ppc(net, init="flat")  # convert to ppc dict
    return ppc


def ensure_extended_branch(ppc):
    """Ensure branch matrix has enough columns for all required indices."""
    from pandapower.pypower.idx_brch import BR_R_ASYM, BR_X_ASYM
    
    br = ppc["branch"]
    needed_cols = max(br.shape[1], BR_R_ASYM + 1, BR_X_ASYM + 1)
    if br.shape[1] < needed_cols:
        padded = np.zeros((br.shape[0], needed_cols), dtype=br.dtype)
        padded[:, :br.shape[1]] = br
        ppc = ppc.copy()
        ppc["branch"] = padded
    return ppc


def run_fdlf(case_load: str, tol: float = 1e-6, max_iter: int = 50) -> dict:
    """Run Fast Decoupled Load Flow on a named pandapower test case."""
    t0_total = time.time()
    
    # Load pandapower case and convert to ppc
    ppc = load_case(case_load)
    ppc = ensure_extended_branch(ppc)
    
    # Note: pandapower's to_ppc already uses 0-based indexing, so no adjustment needed
    
    # Create admittance matrix
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    nbus = len(ppc["bus"])
    bus_data = ppc["bus"]
    gen_data = ppc["gen"]
    bus_types = bus_data[:, 1].astype(int)

    # Initialize voltage magnitudes and angles
    V = bus_data[:, 7].copy()
    theta = np.deg2rad(bus_data[:, 8])

    # Set slack bus conditions
    slack_idx = np.where(bus_types == 3)[0][0]
    V[slack_idx] = 1.0
    theta[slack_idx] = 0.0

    # Load data
    Pd = bus_data[:, 2] / ppc["baseMVA"]
    Qd = bus_data[:, 3] / ppc["baseMVA"]

    # Generation data
    Pg = np.zeros(nbus)
    Qg = np.zeros(nbus)
    for i in range(len(gen_data)):
        bus_idx = int(gen_data[i, 0])  # Already 0-based from pandapower conversion
        Pg[bus_idx] += gen_data[i, 1] / ppc["baseMVA"]
        if bus_types[bus_idx] != 2:  # Not PV bus
            Qg[bus_idx] += gen_data[i, 2] / ppc["baseMVA"]

    # Net injections
    P_spec = Pg - Pd
    Q_spec = Qg - Qd

    # Convert admittance matrix
    Y = Ybus.toarray() if hasattr(Ybus, "toarray") else np.array(Ybus)
    G = Y.real
    B = Y.imag

    # Create B' and B'' matrices for FDLF
    B_prime = -B.copy()
    B_double_prime = -B.copy()

    # Define bus sets
    pv_pq_buses = np.where((bus_types == 1) | (bus_types == 2))[0]  # PQ + PV buses
    pq_buses = np.where(bus_types == 1)[0]  # PQ buses only

    # Create reduced matrices
    B_prime_reduced = B_prime[np.ix_(pv_pq_buses, pv_pq_buses)]
    B_double_prime_reduced = B_double_prime[np.ix_(pq_buses, pq_buses)]

    # Convert to sparse for efficient solving
    B_prime_sparse = csc_matrix(B_prime_reduced)
    B_double_prime_sparse = csc_matrix(B_double_prime_reduced)

    iterations = 0
    time_solve = 0.0

    # FDLF iterations
    for iter_count in range(max_iter):
        iterations = iter_count + 1
        
        # Calculate power injections
        P_calc = np.zeros(nbus)
        Q_calc = np.zeros(nbus)

        for i in range(nbus):
            for k in range(nbus):
                ang = theta[i] - theta[k]
                P_calc[i] += V[i] * V[k] * (G[i, k] * np.cos(ang) + B[i, k] * np.sin(ang))
                Q_calc[i] += V[i] * V[k] * (G[i, k] * np.sin(ang) - B[i, k] * np.cos(ang))

        # Calculate mismatches
        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc

        # Normalize mismatches
        dP_norm = dP / V
        dQ_norm = dQ / V

        # Extract relevant mismatches
        dP_pv_pq = dP[pv_pq_buses]
        dQ_pq = dQ[pq_buses]

        # Check convergence
        max_mismatch = max(np.max(np.abs(dP_pv_pq)), np.max(np.abs(dQ_pq)) if len(dQ_pq) else 0)
        if max_mismatch < tol:
            break

        # Solve linear systems
        t0_solve = time.time()

        # Solve for angle corrections
        if len(pv_pq_buses) > 0:
            dtheta_reduced = spsolve(B_prime_sparse, dP_norm[pv_pq_buses])
        else:
            dtheta_reduced = np.array([])

        # Solve for voltage magnitude corrections
        if len(pq_buses) > 0:
            dV_reduced = spsolve(B_double_prime_sparse, dQ_norm[pq_buses])
        else:
            dV_reduced = np.array([])

        t1_solve = time.time()
        time_solve += (t1_solve - t0_solve)

        # Update variables
        for i, bus_idx in enumerate(pv_pq_buses):
            theta[bus_idx] += dtheta_reduced[i]

        for i, bus_idx in enumerate(pq_buses):
            V[bus_idx] += dV_reduced[i]
            V[bus_idx] = max(V[bus_idx], 0.1)  # Prevent negative voltages

    time_total = time.time() - t0_total

    return {
        "case": case_load,
        "bus_angles": np.rad2deg(theta).tolist(),
        "bus_voltages": V.tolist(),
        "iterations": iterations,
        "converged": max_mismatch < tol,
        "final_mismatch": max_mismatch,
        "time_total": time_total,
        "time_solve": time_solve,
        "Ybus_shape": Ybus.shape,
    }


if __name__ == "__main__":
    # Test cases using pandapower networks
    test_cases = ["ieee9", "ieee14", "ieee30"]

    for case_load in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing {case_load}")
        print('='*50)

        try:
            res = run_fdlf(case_load, tol=1e-6, max_iter=50)

            print(f"Converged: {res['converged']}")
            print(f"Iterations: {res['iterations']}")
            print(f"Final mismatch: {res['final_mismatch']:.2e}")
            print(f"Total time: {res['time_total']:.6f} seconds")
            print(f"Solve time: {res['time_solve']:.6f} seconds")
            print(f"Overhead: {((res['time_total'] - res['time_solve']) / res['time_total'] * 100):.2f}%")
            print(f"Ybus shape: {res['Ybus_shape']}")

            print(f"\nBus Angles (degrees):")
            for i, angle in enumerate(res['bus_angles']):
                print(f"  Bus {i+1}: {angle:.4f}°")

            print(f"\nBus Voltages (p.u.):")
            for i, voltage in enumerate(res['bus_voltages']):
                print(f"  Bus {i+1}: {voltage:.4f} p.u.")

        except Exception as e:
            print(f"Error testing {case_load}: {e}")
            import traceback
            traceback.print_exc()
