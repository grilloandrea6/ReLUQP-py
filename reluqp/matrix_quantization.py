import numpy as np
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB

EXPS = np.arange(-32, 32)


def matrix_quantization(A, bins=10000, metric='absolute', time_limit=600):
    # 1. SETUP AND SOLVE MIP
    centers, counts, rep_values, zeros = prepare_bins(A, B=bins)
    E = compute_error_bins(rep_values, metric=metric)

    # Solve the MIP to find the best K exponents and their assignment to the log-bins
    res = solve_mip_on_bins(rep_values, counts, E, K=16, time_limit=time_limit, mip_gap=1e-4, verbose=True)
    assigned_exponents_per_bin = np.asarray(res['assigned']) # Array: [exponent_bin_0, exponent_bin_1, ...]

    # 2. MAP ORIGINAL VALUES TO BINS AND GET ASSIGNED EXPONENTS
    A_flat = A.ravel()
    nz_mask = np.abs(A_flat) > 0
    a_flat_nz = A_flat[nz_mask]
    
    # Rebuild histogram edges (identical to prepare_bins logic)
    if centers.size == 1:
        edges = np.array([centers[0] - 0.5, centers[0] + 0.5])
    else:
        gaps = np.diff(centers)
        left = centers[0] - gaps[0] / 2.0
        right = centers[-1] + gaps[-1] / 2.0
        mid = centers[:-1] + gaps / 2.0
        edges = np.empty(centers.size + 1, dtype=mid.dtype)
        edges[0] = left
        edges[-1] = right
        edges[1:-1] = mid

    # Map each nonzero element's log-magnitude to the bin index
    logv = np.log2(np.abs(a_flat_nz) + 1e-300)
    bin_idx = np.digitize(logv, edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(centers) - 1)

    # Get the exponent assigned to each *original* nonzero element
    exps = assigned_exponents_per_bin[bin_idx] # This is correct

    # 3. RECONSTRUCT QUANTIZED VALUE (MANTISSA * 2^EXPONENT)
    scales = 2.0 ** exps
    m = np.round(a_flat_nz / scales)
    m = np.clip(m, -128, 127) # Quantize mantissa to 8-bit signed

    # Compute final approximated values
    approx_flat = np.zeros_like(A_flat)
    approx_flat[nz_mask] = m * scales
    
    return approx_flat.reshape(A.shape)

def prepare_bins(A, B=10000, ignore_zeros=True):
    a = A.ravel()
    mags = np.abs(a)
    if ignore_zeros:
        nz_mask = mags > 0
        mags_nz = mags[nz_mask]
        zero_count = (~nz_mask).sum()
    else:
        mags_nz = mags
        zero_count = 0
    logv = np.log2(mags_nz + 1e-300)
    hist, edges = np.histogram(logv, bins=B)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nonzero = hist > 0
    centers = centers[nonzero]
    counts = hist[nonzero].astype(np.int64)
    rep_values = 2.0 ** centers
    return centers, counts, rep_values, zero_count

def compute_error_bins(rep_values, metric='absolute'):
    V = rep_values.reshape(-1,1)
    E = np.empty((V.shape[0], len(EXPS)), dtype=np.float64)
    for j,e in enumerate(EXPS):
        scale = 2.0 ** e
        m = np.round(V / scale)
        m = np.clip(m, -128, 127)
        approx = m * scale
        if metric == 'absolute':
            E[:,j] = np.abs(V - approx).ravel()
        elif metric == 'squared':
            E[:,j] = (V - approx).ravel()**2
        elif metric == 'relative':
            E[:,j] = np.abs(V - approx).ravel() / (V.ravel() + 1e-30)
        else:
            raise ValueError("metric")
    return E

def solve_mip_on_bins(rep_values, counts, E, K=16, time_limit=600, mip_gap=1e-4, verbose=True):
    if gp is None:
        raise RuntimeError("gurobipy not available")

    B = rep_values.shape[0]
    m = gp.Model("quantize_bins")
    m.setParam('OutputFlag', 1 if verbose else 0)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', mip_gap)

    # binary y_e
    y = m.addVars(len(EXPS), vtype=GRB.BINARY, name="y")
    # binary x_b_e
    x = m.addVars(B, len(EXPS), vtype=GRB.BINARY, name="x")
    # objective: sum_b counts[b] * sum_e E[b,e] * x[b,e]
    obj = gp.quicksum(counts[b] * E[b,j] * x[b,j] for b in range(B) for j in range(len(EXPS)))
    m.setObjective(obj, GRB.MINIMIZE)
    # constraints: sum_e x_b_e == 1 for all b
    for b in range(B):
        m.addConstr(gp.quicksum(x[b,j] for j in range(len(EXPS))) == 1)
    # x_b_e <= y_e
    for b in range(B):
        for j in range(len(EXPS)):
            m.addConstr(x[b,j] <= y[j])
    # sum_e y_e == K
    m.addConstr(gp.quicksum(y[j] for j in range(len(EXPS))) == K)

    m.optimize()
    # extract solution
    y_sol = [int(y[j].X) for j in range(len(EXPS))]
    chosen = [int(EXPS[j]) for j,val in enumerate(y_sol) if val==1]
    # compute assigned exponent per bin
    assigned = np.empty(B, dtype=int)
    for b in range(B):
        for j in range(len(EXPS)):
            if x[b,j].X > 0.5:
                assigned[b] = int(EXPS[j])
                break
    return dict(chosen=chosen, assigned=assigned, model=m)
