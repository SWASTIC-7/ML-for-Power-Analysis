import time
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from pandapower.converter import to_ppc
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def load_pandapower_case(case_file: str):
    """Load a pandapower network case dynamically and convert to ppc format."""
    # Extract case name from file path (remove .m extension if present)
    if case_file.endswith('.m'):
        case_name = case_file.replace('.m', '').split('/')[-1].split('\\')[-1]
    else:
        case_name = case_file.split('/')[-1].split('\\')[-1]
    
    # Clean the case name and make it lowercase for consistent matching
    case_name = case_name.lower().strip()
    
    # Try to get the function from pandapower.networks dynamically
    net_function = None
    
    # First, try direct attribute access
    if hasattr(pn, case_name):
        net_function = getattr(pn, case_name)
    
    # If not found, try common variations
    if net_function is None:
        # Try with 'case_' prefix
        if hasattr(pn, f"case_{case_name}"):
            net_function = getattr(pn, f"case_{case_name}")
        # Try with 'case_ieee' prefix
        elif hasattr(pn, f"case_ieee{case_name.replace('case', '').replace('ieee', '')}"):
            net_function = getattr(pn, f"case_ieee{case_name.replace('case', '').replace('ieee', '')}")
        # Try removing 'case' prefix if present
        elif case_name.startswith('case') and hasattr(pn, case_name[4:]):
            net_function = getattr(pn, case_name[4:])
        # Try adding 'case' prefix if not present
        elif not case_name.startswith('case') and hasattr(pn, f"case{case_name}"):
            net_function = getattr(pn, f"case{case_name}")
    
    # If still not found, try a comprehensive search through all available functions
    if net_function is None:
        available_cases = []
        for attr_name in dir(pn):
            attr = getattr(pn, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                available_cases.append(attr_name)
                # Check if this matches our case (flexible matching)
                if (case_name in attr_name.lower() or 
                    attr_name.lower() in case_name or
                    case_name.replace('case', '').replace('ieee', '') in attr_name.lower()):
                    net_function = attr
                    break
        
        if net_function is None:
            raise ValueError(f"Unknown test case: '{case_file}'. Available cases in pandapower.networks: {available_cases}")
    
    # Load pandapower network and convert to ppc
    try:
        net = net_function()
        ppc = to_ppc(net, init="flat")
        return ppc
    except Exception as e:
        raise ValueError(f"Error loading case '{case_file}': {e}")


def run_fdlf(case_file: str, tol: float = 1e-6, max_iter: int = 1) -> dict:
    """Run Fast Decoupled Load Flow on a pandapower case using makeYbus."""
    t0_total = time.time()
    
    ppc = load_pandapower_case(case_file)
    
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    
    nbus = len(ppc["bus"])
    bus_data = ppc["bus"]
    gen_data = ppc["gen"]
    bus_types = bus_data[:, 1].astype(int)
    
    V = bus_data[:, 7].copy()
    theta = np.deg2rad(bus_data[:, 8])
    
    slack_idx = np.where(bus_types == 3)[0]
    if len(slack_idx) > 0:
        slack_idx = slack_idx[0]
        V[slack_idx] = 1.0
        theta[slack_idx] = 0.0
    
    Pd = bus_data[:, 2] / ppc["baseMVA"]
    Qd = bus_data[:, 3] / ppc["baseMVA"]
    
    Pg = np.zeros(nbus)
    Qg = np.zeros(nbus)
    for i in range(len(gen_data)):
        bus_idx = int(gen_data[i, 0]) 
        Pg[bus_idx] += gen_data[i, 1] / ppc["baseMVA"]
        if bus_types[bus_idx] != 2:  # Not PV bus
            Qg[bus_idx] += gen_data[i, 2] / ppc["baseMVA"]
    
    P_spec = Pg - Pd
    Q_spec = Qg - Qd
    
    Y = Ybus.toarray() if hasattr(Ybus, "toarray") else np.array(Ybus)
    G = Y.real
    B = Y.imag
    
    B_prime = -B.copy()
    B_double_prime = -B.copy()
    
    pv_pq_buses = np.where((bus_types == 1) | (bus_types == 2))[0]  # PQ + PV buses
    pq_buses = np.where(bus_types == 1)[0]  # PQ buses only
    
    if len(pv_pq_buses) > 0:
        B_prime_reduced = B_prime[np.ix_(pv_pq_buses, pv_pq_buses)]
        B_prime_sparse = csc_matrix(B_prime_reduced)
    else:
        B_prime_sparse = None
    
    if len(pq_buses) > 0:
        B_double_prime_reduced = B_double_prime[np.ix_(pq_buses, pq_buses)]
        B_double_prime_sparse = csc_matrix(B_double_prime_reduced)
    else:
        B_double_prime_sparse = None
    
    iterations = 0
    time_solve = 0.0
    max_mismatch = float('inf')
    
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
        
        dP_norm = np.divide(dP, V, out=np.zeros_like(dP), where=V!=0)
        dQ_norm = np.divide(dQ, V, out=np.zeros_like(dQ), where=V!=0)
        
        dP_pv_pq = dP[pv_pq_buses] if len(pv_pq_buses) > 0 else np.array([])
        dQ_pq = dQ[pq_buses] if len(pq_buses) > 0 else np.array([])
        
        max_mismatch_p = np.max(np.abs(dP_pv_pq)) if len(dP_pv_pq) > 0 else 0.0
        max_mismatch_q = np.max(np.abs(dQ_pq)) if len(dQ_pq) > 0 else 0.0
        max_mismatch = max(max_mismatch_p, max_mismatch_q)
        
        if max_mismatch < tol:
            break
        
        t0_solve = time.time()
        
        if len(pv_pq_buses) > 0 and B_prime_sparse is not None:
            try:
                dtheta_reduced = spsolve(B_prime_sparse, dP_norm[pv_pq_buses])
            except:
                dtheta_reduced = np.zeros(len(pv_pq_buses))
        else:
            dtheta_reduced = np.array([])
        
        if len(pq_buses) > 0 and B_double_prime_sparse is not None:
            try:
                dV_reduced = spsolve(B_double_prime_sparse, dQ_norm[pq_buses])
            except:
                dV_reduced = np.zeros(len(pq_buses))
        else:
            dV_reduced = np.array([])
        
        t1_solve = time.time()
        time_solve += (t1_solve - t0_solve)
        
        for i, bus_idx in enumerate(pv_pq_buses):
            if i < len(dtheta_reduced):
                theta[bus_idx] += dtheta_reduced[i]
        
        for i, bus_idx in enumerate(pq_buses):
            if i < len(dV_reduced):
                V[bus_idx] += dV_reduced[i]
    
    time_total = time.time() - t0_total
    
    return {
        "bus_angles": np.rad2deg(theta).tolist(),
        "bus_voltages": V.tolist(),
        "iterations": iterations,
        "time_total": time_total,
        "time_solve": time_solve
    }


if __name__ == "__main__":
    test_cases = ["case9", "case14", "case30"]
    
    for case_file in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing {case_file}")
        print('='*50)
        
        try:
            
            res = run_fdlf(case_file, tol=1e-6, max_iter=50)
                
            print(f"Iterations: {res['iterations']}")
            print(f"Total time: {res['time_total']:.6f}")
            print(f"Solve time: {res['time_solve']:.6f}")
            print(f"Overhead: {((res['time_total'] - res['time_solve']) / res['time_total'] * 100):.2f}%" if res['time_total'] > 0 else "0.00%")
                
            print(f"Bus Angles (first 5): {[f'{angle:.4f}°' for angle in res['bus_angles'][:5]]}")
            print(f"Bus Voltages (first 5): {[f'{voltage:.4f} p.u.' for voltage in res['bus_voltages'][:5]]}")
                
        except Exception as e:
            print(f"Error testing {case_file}: {e}")
            import traceback
            traceback.print_exc()