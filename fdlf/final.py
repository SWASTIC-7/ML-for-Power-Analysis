import os
import time
import pandapower
from pandapower.pypower.makeYbus import makeYbus
from pypower.loadcase import loadcase
from pypower.api import case9, case14, case30 
from pandapower.pypower.idx_brch import BR_R_ASYM, BR_X_ASYM  
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def ensure_extended_branch(ppc):
    br = ppc["branch"]
    needed_cols = max(br.shape[1], BR_R_ASYM + 1, BR_X_ASYM + 1)
    if br.shape[1] < needed_cols:
        padded = np.zeros((br.shape[0], needed_cols), dtype=br.dtype)
        padded[:, :br.shape[1]] = br
        ppc = ppc.copy()
        ppc["branch"] = padded
    return ppc


def adjust_bus_numbers(ppc):
    br = ppc["branch"].copy()
    br[:, 0] = br[:, 0] - 1  
    br[:, 1] = br[:, 1] - 1  
    ppc = ppc.copy()
    ppc["branch"] = br
    return ppc


def load_case(case_file_or_ppc):
    if isinstance(case_file_or_ppc, dict):
        return case_file_or_ppc

    if isinstance(case_file_or_ppc, str) and case_file_or_ppc.endswith(".m"):
        name = os.path.splitext(os.path.basename(case_file_or_ppc))[0]
        if name == "case9":
            return case9()
        elif name == "case14":
            return case14()
        elif name == "case30":
            return case30()
        else:
            raise ValueError(f"No Python equivalent available for {case_file_or_ppc}")

    return loadcase(case_file_or_ppc)


def run_fdlf(case_file: str, tol: float = 1e-6, max_iter: int = 50) -> dict:
    t0_total = time.time()
    ppc = load_case(case_file)
    ppc = ensure_extended_branch(ppc)
    ppc = adjust_bus_numbers(ppc)  

    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    nbus = len(ppc["bus"])
    bus_data = ppc["bus"]
    gen_data = ppc["gen"]
    bus_types = bus_data[:, 1].astype(int)

    V = bus_data[:, 7].copy()
    theta = np.deg2rad(bus_data[:, 8])

    slack_idx = np.where(bus_types == 3)[0][0]
    V[slack_idx] = 1.0
    theta[slack_idx] = 0.0

    Pd = bus_data[:, 2] / ppc["baseMVA"]
    Qd = bus_data[:, 3] / ppc["baseMVA"]

    Pg = np.zeros(nbus)
    Qg = np.zeros(nbus)
    for i in range(len(gen_data)):
        bus_idx = int(gen_data[i, 0] - 1)
        Pg[bus_idx] += gen_data[i, 1] / ppc["baseMVA"]
        if bus_types[bus_idx] != 2:
            Qg[bus_idx] += gen_data[i, 2] / ppc["baseMVA"]

    P_spec = Pg - Pd
    Q_spec = Qg - Qd

    Y = Ybus.toarray() if hasattr(Ybus, "toarray") else np.array(Ybus)
    G = Y.real
    B = Y.imag

    B_prime = -B.copy()
    B_double_prime = -B.copy()

    pv_pq_buses = np.where((bus_types == 1) | (bus_types == 2))[0]
    pq_buses = np.where(bus_types == 1)[0]

    B_prime_reduced = B_prime[np.ix_(pv_pq_buses, pv_pq_buses)]
    B_double_prime_reduced = B_double_prime[np.ix_(pq_buses, pq_buses)]

    B_prime_sparse = csc_matrix(B_prime_reduced)
    B_double_prime_sparse = csc_matrix(B_double_prime_reduced)

    iterations = 0
    time_solve = 0.0

    for iter_count in range(max_iter):
        iterations = iter_count + 1
        P_calc = np.zeros(nbus)
        Q_calc = np.zeros(nbus)

        for i in range(nbus):
            for k in range(nbus):
                ang = theta[i] - theta[k]
                P_calc[i] += V[i] * V[k] * (G[i, k] * np.cos(ang) + B[i, k] * np.sin(ang))
                Q_calc[i] += V[i] * V[k] * (G[i, k] * np.sin(ang) - B[i, k] * np.cos(ang))

        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc

        dP_norm = dP / V
        dQ_norm = dQ / V

        dP_pv_pq = dP[pv_pq_buses]
        dQ_pq = dQ[pq_buses]

        max_mismatch = max(np.max(np.abs(dP_pv_pq)), np.max(np.abs(dQ_pq)) if len(dQ_pq) else 0)
        if max_mismatch < tol:
            break

        t0_solve = time.time()

        if len(pv_pq_buses) > 0:
            dtheta_reduced = spsolve(B_prime_sparse, dP_norm[pv_pq_buses])
        else:
            dtheta_reduced = np.array([])

        if len(pq_buses) > 0:
            dV_reduced = spsolve(B_double_prime_sparse, dQ_norm[pq_buses])
        else:
            dV_reduced = np.array([])

        t1_solve = time.time()
        time_solve += (t1_solve - t0_solve)

        for i, bus_idx in enumerate(pv_pq_buses):
            theta[bus_idx] += dtheta_reduced[i]

        for i, bus_idx in enumerate(pq_buses):
            V[bus_idx] += dV_reduced[i]
            V[bus_idx] = max(V[bus_idx], 0.1)

    time_total = time.time() - t0_total

    return {
        "bus_angles": np.rad2deg(theta).tolist(),
        "bus_voltages": V.tolist(),
        "iterations": iterations,
        "time_total": time_total,
        "time_solve": time_solve,
    }


if __name__ == "__main__":
    data_dir = os.path.join(".", "matpower", "data")
    test_cases = ["case9.m", "case14.m", "case30.m"]

    for case_file in test_cases:
        full_path = os.path.join(data_dir, case_file)

        print(f"\n{'='*50}")
        print(f"Testing {case_file}")
        print('='*50)

        try:
            res = run_fdlf(full_path, tol=1e-6, max_iter=50)

            print(f"Converged in {res['iterations']} iterations")
            print(f"Total time: {res['time_total']:.6f} seconds")
            print(f"Solve time: {res['time_solve']:.6f} seconds")
            print(f"Overhead: {((res['time_total'] - res['time_solve']) / res['time_total'] * 100):.2f}%")

            print(f"\nBus Angles (degrees):")
            for i, angle in enumerate(res['bus_angles']):
                print(f"  Bus {i+1}: {angle:.4f}°")

            print(f"\nBus Voltages (p.u.):")
            for i, voltage in enumerate(res['bus_voltages']):
                print(f"  Bus {i+1}: {voltage:.4f} p.u.")

        except Exception as e:
            print(f"Error testing {case_file}: {e}")
            import traceback
            traceback.print_exc()
