"""Graph-constrained neural ODE for network-spreading models of neurodegeneration.

Each of 82 regions carries a disease burden x in [0, 1]. Disease spreads along
white-matter edges and grows locally within a region. The pipeline fits a mean
baseline, NDM (spreading only), FKPP (spreading + logistic growth), and then FKPP
plus a learned edge-gated correction, all scored on the same held-out patients.

Two caveats attach to the results. First, only the LEARNED CORRECTION is
conservative. The full model does not conserve total disease burden and should not,
because the growth term creates burden -- that is the biology being modelled.
check_correction_conservation verifies the narrow claim, never the broad one.
Second, synthetic mode runs on data generated from a known FKPP equation, so
recovering FKPP-like parameters there is evidence the pipeline is correct, not a
biological finding.

    python pipeline.py                        (synthetic, all 7 steps)
    python pipeline.py --data-source real     (real OASIS data, Steps 2-4 only)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import huber_loss, relu
from scipy.linalg import eigvalsh
from torchdiffeq import odeint

# Imported, not duplicated, so one definition of the burden transform and one value
# of Z_CAP govern the project; a local copy could drift from prep_oasis.py's.
from prep_oasis import Z_CAP, atrophy_score_normalize, report_atrophy_score_transform

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SYNTH_DIR = BASE_DIR / "data" / "synthetic"
REAL_DIR = BASE_DIR / "data" / "real"

# One seed for both generators: fixes the patient split and the network's starting
# weights, so a re-run reproduces the reported numbers.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"[SEED] torch={SEED}, numpy={SEED}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {DEVICE}")

# Huber changeover point, roughly 1.5x the measurement noise (0.02). At its former
# value of 1.0 every error in this data sat below the threshold, so the loss never
# left its squared-error branch and the robust behaviour was inactive.
HUBER_DELTA = 0.03

# Tolerance band for the monotonicity penalty: dips up to 0.04 between timepoints
# are treated as measurement wobble and go unpenalised.
TAU_MONO = 0.04


# --- Step 1: load and validate the inputs, build the Laplacian ---

def load_atrophy(path, disease_prefix, region_order):
    """Read a patient-scan CSV into (raw table, list of per-patient dicts).

    Each dict is {"patient_id", "times", "values"} with that patient's own scan
    times and a (timepoints x 82) burden array. Per-patient time arrays rather than
    one shared grid, because real scan schedules are irregular and the solver has to
    integrate to each patient's actual dates. Patients with a single scan are
    dropped: with one timepoint there is no change over time to fit.
    """
    # --- Load the table and check its column order ---
    df = pd.read_csv(path)
    assert list(df.columns[2:]) == region_order, f"{path} columns != ground truth region order"

    # --- Sort chronologically within each patient ---
    df = df.sort_values(["patient_id", "time_years"]).reset_index(drop=True)

    # --- Split into per-patient records, dropping single-scan patients ---
    patients = []
    dropped = 0
    for pid, g in df.groupby("patient_id", sort=True):
        g = g.sort_values("time_years")
        times = g["time_years"].values.astype(np.float64)
        if len(times) < 2:
            dropped += 1
            continue
        values = g[region_order].values.astype(np.float64)
        patients.append({"patient_id": pid, "times": times, "values": values})

    assert len(patients) > 0, f"{disease_prefix}: no patients with >=2 timepoints"
    print(f"  [{disease_prefix}] loaded {len(patients)} patients ({dropped} dropped for <2 timepoints)")
    return df, patients


def group_by_time_signature(patients):
    """Bundle patients with identical scan times into batched solver groups.

    Performance only, no effect on results. odeint can integrate many patients at
    once but only on a shared time array, so patients are keyed by their rounded
    time vector. Synthetic data (everyone at [0,1,2,3]) collapses to one group;
    irregular real schedules degrade to more, smaller groups. Times are rounded to 8
    decimals so identical schedules are not split apart by floating-point dust, and
    first-seen order is preserved so grouping is deterministic.
    """
    # --- Accumulate patients under their rounded time signature ---
    groups_map = {}
    order = []

    for pos, p in enumerate(patients):
        key = tuple(np.round(p["times"], 8))
        if key not in groups_map:
            groups_map[key] = {"times": np.array(key, dtype=np.float64), "positions": [], "values_list": []}
            order.append(key)
        groups_map[key]["positions"].append(pos)
        groups_map[key]["values_list"].append(p["values"])

    # --- Stack each group into one array for batched solving ---
    groups = []
    for key in order:
        g = groups_map[key]
        groups.append({
            "times": g["times"],
            "positions": np.array(g["positions"], dtype=np.int64),
            "values": np.stack(g["values_list"], axis=0),
        })
    return groups


def load_and_validate(data_source="synthetic"):
    """Load and verify every input file, build L_norm, print diagnostics.

    Returns (gt, W, L_norm, ad_patients, pd_patients, ad_pids, pd_pids). The mode is
    an explicit argument and the two modes read from entirely separate folders, so
    synthetic and real data cannot be mixed within one run.

    Ordering is asserted rather than assumed: a region-order mismatch between files
    would still compute cleanly while comparing the hippocampus to the insula. Every
    eigenvalue of a correctly built normalized Laplacian lies in [0, 2], so that
    check is a real test that W is well formed, not a preference.

    data/real/tractography.csv is never read here -- it uses a different naming and
    ordering. build_connectome_82region.py translates it, and only that verified
    output is loaded.
    """
    assert data_source in ("synthetic", "real"), f"Unknown data_source: {data_source!r}"
    print(f"[DATA SOURCE] {data_source}")

    # --- Load the canonical region order ---
    with open(CONFIG_DIR / "ground_truth_params.json") as f:
        gt = json.load(f)
    region_order = gt["region_order"]
    n_regions = len(region_order)
    assert n_regions == 82, f"Expected 82 regions, got {n_regions}"

    # --- Pick the input paths for this mode ---
    if data_source == "synthetic":
        conn_path = SYNTH_DIR / "synthetic_connectome.csv"
        ad_path = SYNTH_DIR / "Ad_atrophy_longitudinal.csv"
        pd_path = SYNTH_DIR / "Pd_atrophy_longitudinal.csv"
    else:
        conn_path = REAL_DIR / "connectome_82region.csv"
        ad_path = REAL_DIR / "Ad_atrophy_real.csv"
        pd_path = REAL_DIR / "Pd_atrophy_real.csv"
        assert conn_path.exists(), (
            f"{conn_path} not found -- run code/build_connectome_82region.py first. "
            f"Never fall back to data/real/tractography.csv directly."
        )
        assert ad_path.exists(), f"{ad_path} not found -- run code/prep_oasis.py first."

    # --- Load the connectome and check its structure ---
    conn_df = pd.read_csv(conn_path, index_col=0)
    assert list(conn_df.columns) == region_order, f"{conn_path.name} columns != ground truth region order"
    assert list(conn_df.index) == region_order, f"{conn_path.name} index != ground truth region order"
    W = conn_df.values.astype(np.float64)
    assert W.shape == (82, 82), f"Connectome shape {W.shape}"
    assert np.allclose(W, W.T), "Connectome not symmetric"
    assert np.allclose(np.diag(W), 0), "Connectome diagonal nonzero"
    assert (W.sum(axis=1) > 0).all(), "Isolated node in connectome"

    # --- Build the normalized Laplacian and verify its eigenvalue range ---
    deg = W.sum(axis=1)
    dinv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L_norm = np.eye(82) - dinv_sqrt @ W @ dinv_sqrt
    assert np.allclose(L_norm, L_norm.T, atol=1e-12), "L_norm not symmetric"
    eigs = eigvalsh(L_norm)
    assert eigs.min() >= -1e-10, f"Negative eigenvalue: {eigs.min()}"
    assert eigs.max() <= 2.0 + 1e-10, f"Eigenvalue > 2: {eigs.max()}"
    print(f"[L_norm] eigenvalue range: [{eigs.min():.6f}, {eigs.max():.6f}]")

    # --- Load the AD patient scans ---
    ad_df, ad_patients = load_atrophy(ad_path, "AD", region_order)
    ad_pids = np.array([p["patient_id"] for p in ad_patients])

    # --- Load the PD scans, which are allowed to be absent on the real path only ---
    if pd_path.exists():
        pd_df, pd_patients = load_atrophy(pd_path, "PD", region_order)
        pd_pids = np.array([p["patient_id"] for p in pd_patients])
    else:
        assert data_source == "real", f"{pd_path} missing in synthetic mode -- STOP"
        print(f"  [PD] {pd_path.name} not found -- no real PD data yet, PD side skipped")
        pd_patients, pd_pids = None, None

    # --- Print the loaded shapes ---
    print(f"\n[SHAPES]")
    print(f"  AD patients: {len(ad_patients)}  (variable timepoints/regions=82 each)")
    print(f"  PD patients: {len(pd_patients) if pd_patients is not None else 'N/A'}")
    print(f"  Connectome W: {W.shape}")
    print(f"  L_norm: {L_norm.shape}")

    # --- Print the burden value ranges ---
    ad_all_vals = np.concatenate([p["values"] for p in ad_patients], axis=0)
    print(f"\n[ATROPHY RANGES]")
    print(f"  AD: min={ad_all_vals.min():.6f}, max={ad_all_vals.max():.6f}")
    if pd_patients is not None:
        pd_all_vals = np.concatenate([p["values"] for p in pd_patients], axis=0)
        print(f"  PD: min={pd_all_vals.min():.6f}, max={pd_all_vals.max():.6f}")

    return gt, W, L_norm, ad_patients, pd_patients, ad_pids, pd_pids


# --- Step 2: train/val/test split and the mean baseline ---

def patient_split(n_patients, seed=SEED):
    """Split patients 70/15/15 into train, validation and test index arrays.

    The split is patient-level, not scan-level. If one person's year-0 scan were in
    training and their year-3 scan in test, the model would already have seen that
    individual's disease pattern and the test number would be optimistic, so all of
    a person's scans go to the same split.

    Validation chooses between model settings (here, lambda_reg); test is touched
    once, to report. The seed fixes the split so every model is compared on exactly
    the same held-out people.
    """
    # --- Shuffle patient positions under the fixed seed ---
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_patients)

    # --- Cut at 70% and 85%, giving the remainder to test ---
    n_train = int(0.70 * n_patients)
    n_val = int(0.15 * n_patients)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def atrophy_score_normalize_patients(patients):
    """Convert real-data w-scores to the model's burden scale, x = clip(-w, 0, Z_CAP) / Z_CAP.

    Takes patient dicts holding raw w-scores, returns (converted patients, a record
    of the transform). The transform is defined in prep_oasis.py; see its docstring
    for why it replaced global min-max and why Z_CAP is fixed in advance.
    """
    # --- Print the before and after distribution, including clipping counts ---
    all_raw = np.concatenate([p["values"] for p in patients], axis=0)
    report_atrophy_score_transform(all_raw, label="real AD, all patients/timepoints")

    # --- Apply the transform to every patient ---
    norm_patients = [
        {"patient_id": p["patient_id"], "times": p["times"],
         "values": atrophy_score_normalize(p["values"])}
        for p in patients
    ]
    return norm_patients, {"transform": "atrophy_score", "z_cap": Z_CAP}


# Dead code, kept for the record. Never called. See docstring for why it was rejected.
def minmax_normalize_patients(patients):
    """Rejected alternative to atrophy_score_normalize_patients, replaced 2026-07-29.

    Stretching the global min to 0 and max to 1 kept the raw w-score sign, so more
    diseased regions (more negative w) landed nearer x = 0 -- backwards for a growth
    term that needs x = 1 to mean fully diseased. It also put zero in the wrong
    place: on a raw range of about -6.7 to +5.5 a healthy region (w = 0) landed at
    x = 0.55, compressing ~99% of the data into x in [0.20, 0.76] where x*(1-x) is
    0.24 +/- 0.02, pinned near its 0.25 maximum. And L_norm applied to a constant is
    zero only on a perfectly regular graph; measured on this connectome the worst
    case reached 0.685, so a constant 0.55 offset added up to 0.377 of
    disease-irrelevant drift. Do not reinstate without re-deriving both points.

    Note also that min-max is a shift AND a scale, so parameters fitted after it are
    not on the same footing as parameters fitted on data already in [0,1].
    """
    # --- Rescale every patient by the global min and max ---
    all_vals = np.concatenate([p["values"] for p in patients], axis=0)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    scale = vmax - vmin
    assert scale > 0, "Degenerate data: all values identical, cannot min-max normalize"
    norm_patients = [
        {"patient_id": p["patient_id"], "times": p["times"], "values": (p["values"] - vmin) / scale}
        for p in patients
    ]
    print(f"  [NORMALIZE] global min-max: min={vmin:.4f}, max={vmax:.4f} -> mapped to [0, 1]")
    return norm_patients, {"min": vmin, "max": vmax}


def mean_baseline(patients, train_idx, test_idx, disease_name, patient_ids):
    """Predict the mean training trajectory for every test patient; return (Huber, MSE).

    Averaging only works when the test patient's scan schedule also appeared in
    training. When it did not, the fallback is a per-region least-squares line
    fitted on all pooled training scans and evaluated at that patient's own times.
    On synthetic data every patient shares [0,1,2,3], so the fallback never fires.

    Errors are divided by element count, not patient count, so patients with more
    scans do not pick up extra weight.
    """
    train_patients = [patients[i] for i in train_idx]
    test_patients = [patients[i] for i in test_idx]

    # --- Average the training patients within each scan schedule ---
    train_groups = group_by_time_signature(train_patients)
    mean_by_signature = {}
    for g in train_groups:
        key = tuple(np.round(g["times"], 8))
        mean_by_signature[key] = g["values"].mean(axis=0)

    # --- Fit the fallback: one straight line per region over all training scans ---
    all_times = np.concatenate([p["times"] for p in train_patients])
    all_values = np.concatenate([p["values"] for p in train_patients], axis=0)
    A = np.stack([all_times, np.ones_like(all_times)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, all_values, rcond=None)

    # --- Score each test patient against whichever prediction applies ---
    total_huber = 0.0
    total_sq = 0.0
    total_n = 0
    for p in test_patients:
        key = tuple(np.round(p["times"], 8))
        if key in mean_by_signature:
            pred_traj = mean_by_signature[key]
        else:
            pred_traj = coeffs[0] * p["times"][:, None] + coeffs[1]
        actual = p["values"]
        pred_t = torch.tensor(pred_traj, dtype=torch.float64)
        act_t = torch.tensor(actual, dtype=torch.float64)
        total_huber += huber_loss(pred_t, act_t, delta=HUBER_DELTA, reduction="sum").item()
        total_sq += ((pred_traj - actual) ** 2).sum()
        total_n += actual.size

    # --- Average by element count and report ---
    h_loss = total_huber / total_n
    mse = total_sq / total_n

    print(f"\n[STEP 2 -- MEAN BASELINE: {disease_name}]")
    print(f"  Test patients ({len(test_idx)}): {sorted(patient_ids[test_idx])}")
    print(f"  Test Huber: {h_loss:.6f}")
    print(f"  Test MSE:   {mse:.6f}")
    return h_loss, mse


def step2(ad_patients, pd_patients, ad_pids, pd_pids):
    """Fix the split reused by Steps 3-5 and run the mean baseline for both diseases.

    Steps 3, 4 and 5 reuse these indices, which is what makes the final comparison
    table honest. The count assertion reflects how the synthetic data was generated:
    matched AD and PD cohorts of equal size, so one index set addresses both.
    """
    # --- Split patients once, for every later step ---
    n_ad, n_pd = len(ad_patients), len(pd_patients)
    assert n_ad == n_pd, f"AD/PD patient count mismatch: {n_ad} vs {n_pd}"
    train_idx, val_idx, test_idx = patient_split(n_ad)
    print(f"\n[SPLIT] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Test indices: {sorted(test_idx)}")

    # --- Run the baseline for both diseases ---
    ad_base = mean_baseline(ad_patients, train_idx, test_idx, "AD", ad_pids)
    pd_base = mean_baseline(pd_patients, train_idx, test_idx, "PD", pd_pids)

    return train_idx, val_idx, test_idx, ad_base, pd_base


# --- Step 3: NDM, spreading only ---

def fit_ndm(patients, L_norm_np, train_idx, test_idx, disease_name,
            lr=1e-2, n_epochs=500):
    """Fit the diffusivity k in dx/dt = -k * L_norm @ x; return (k, test Huber, test MSE).

    Each patient's first scan is the initial condition, odeint integrates to that
    patient's own later scan times, and the Huber error is backpropagated through
    the solver to k. dopri5 is the adaptive-step default, safe with irregular times.

    k is clamped each step: below zero would mean disease flowing up the gradient,
    and too large makes the equation stiff and wrecks the solver. A NaN loss aborts
    rather than propagating silently into every downstream number.
    """
    L_norm_t = torch.tensor(L_norm_np, dtype=torch.float64)

    # --- Set up the single trainable parameter and its optimizer ---
    k = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([k], lr=lr)

    train_groups = group_by_time_signature([patients[i] for i in train_idx])

    def ode_fn(t, x):
        """Return dx/dt for the current state; t is unused, the equation is autonomous."""
        return -k * (x @ L_norm_t.T)

    # --- Fit k by gradient descent ---
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float64)
        total_n = 0

        # --- Integrate each batched group and accumulate the error ---
        for g in train_groups:
            times_t = torch.tensor(g["times"], dtype=torch.float64)
            data_t = torch.tensor(g["values"], dtype=torch.float64)
            x0 = data_t[:, 0, :]
            pred = odeint(ode_fn, x0, times_t, method="dopri5").permute(1, 0, 2)
            total_loss = total_loss + huber_loss(pred, data_t, delta=HUBER_DELTA, reduction="sum")
            total_n += data_t.numel()

        # --- Average by measurement count, not by group ---
        loss = total_loss / total_n

        # --- Abort on a NaN or infinite loss ---
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [{disease_name} NDM] ABORT epoch {epoch+1}: loss={loss.item()}, k={k.item():.4f}")
            raise RuntimeError(f"NaN/Inf loss at epoch {epoch+1}")

        # --- Step the optimizer and clamp k to its physical range ---
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            k.clamp_(1e-3, 5.0)

        # --- Log every epoch for the first 50, then every 20th ---
        if (epoch + 1) <= 50 or (epoch + 1) % 20 == 0:
            print(f"  [{disease_name} NDM] epoch {epoch+1}/{n_epochs}, train Huber={loss.item():.6f}, k={k.item():.4f}")

    # --- Evaluate on held-out patients ---
    test_groups = group_by_time_signature([patients[i] for i in test_idx])
    with torch.no_grad():
        total_huber = 0.0
        total_sq = 0.0
        total_n = 0
        for g in test_groups:
            times_t = torch.tensor(g["times"], dtype=torch.float64)
            data_t = torch.tensor(g["values"], dtype=torch.float64)
            x0 = data_t[:, 0, :]
            pred = odeint(ode_fn, x0, times_t, method="dopri5").permute(1, 0, 2)
            total_huber += huber_loss(pred, data_t, delta=HUBER_DELTA, reduction="sum").item()
            total_sq += ((pred - data_t) ** 2).sum().item()
            total_n += data_t.numel()
        test_loss = total_huber / total_n
        test_mse = total_sq / total_n

    print(f"\n[STEP 3 -- NDM FIT: {disease_name}]")
    print(f"  Fitted k = {k.item():.4f}")
    print(f"  Test Huber: {test_loss:.6f}")
    print(f"  Test MSE:   {test_mse:.6f}")
    return k.item(), test_loss, test_mse


def step3(ad_patients, pd_patients, L_norm, train_idx, test_idx):
    """Fit NDM for both diseases and compare k against the synthetic ground truth.

    Pure diffusion only redistributes burden and can never increase the total, which
    is the limitation Step 4 addresses. The gap from the true k is expected and
    informative: with no growth term, the single k absorbs some of the growth
    behaviour and drifts away from the true spreading rate.
    """
    print("\n" + "="*60)
    print("STEP 3 -- NDM (Network Diffusion Model)")
    print("="*60)

    # --- Fit both diseases ---
    ad_k, ad_huber, ad_mse = fit_ndm(ad_patients, L_norm, train_idx, test_idx, "AD")
    pd_k, pd_huber, pd_mse = fit_ndm(pd_patients, L_norm, train_idx, test_idx, "PD")

    # --- Compare the fitted k against the generator's values ---
    print(f"\n[NDM vs Ground Truth]")
    print(f"  AD: fitted k={ad_k:.4f}, true k=0.80 (gap expected -- NDM lacks growth term)")
    print(f"  PD: fitted k={pd_k:.4f}, true k=0.50")

    return {"AD": (ad_k, ad_huber, ad_mse), "PD": (pd_k, pd_huber, pd_mse)}


# --- Step 4: FKPP, spreading plus logistic growth ---

def fit_fkpp(patients, L_norm_np, train_idx, test_idx, disease_name,
             lr=1e-2, n_epochs=500):
    """Fit k and a in dx/dt = -k * L_norm @ x + a * x * (1 - x); return both plus test scores.

    Growth is proportional to x, so nothing starts from perfectly healthy, and the
    (1 - x) factor saturates at full degeneration.

    Deliberately identical to fit_ndm in batching, solver, loop and guard rails --
    the only differences are the growth term and the second parameter -- so any
    difference in reported error is attributable to the growth term rather than to
    an incidental change in the fitting procedure. Starting values are generic
    rather than the known synthetic truth, so the fit has to actually work.
    """
    L_norm_t = torch.tensor(L_norm_np, dtype=torch.float64)

    # --- Set up the two trainable parameters and their optimizer ---
    k = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([k, a], lr=lr)

    train_groups = group_by_time_signature([patients[i] for i in train_idx])

    def ode_fn(t, x):
        """Return dx/dt: network spreading plus local logistic growth."""
        return -k * (x @ L_norm_t.T) + a * x * (1.0 - x)

    # --- Fit k and a by gradient descent ---
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float64)
        total_n = 0

        # --- Integrate each batched group and accumulate the error ---
        for g in train_groups:
            times_t = torch.tensor(g["times"], dtype=torch.float64)
            data_t = torch.tensor(g["values"], dtype=torch.float64)
            x0 = data_t[:, 0, :]
            pred = odeint(ode_fn, x0, times_t, method="dopri5").permute(1, 0, 2)
            total_loss = total_loss + huber_loss(pred, data_t, delta=HUBER_DELTA, reduction="sum")
            total_n += data_t.numel()
        loss = total_loss / total_n

        # --- Abort on a NaN or infinite loss ---
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [{disease_name} FKPP] ABORT epoch {epoch+1}: loss={loss.item()}, k={k.item():.4f}, a={a.item():.4f}")
            raise RuntimeError(f"NaN/Inf loss at epoch {epoch+1}")

        # --- Step the optimizer and clamp both rates ---
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            k.clamp_(1e-3, 5.0)
            a.clamp_(1e-3, 5.0)

        # --- Log every epoch for the first 50, then every 20th ---
        if (epoch + 1) <= 50 or (epoch + 1) % 20 == 0:
            print(f"  [{disease_name} FKPP] epoch {epoch+1}/{n_epochs}, train Huber={loss.item():.6f}, k={k.item():.4f}, a={a.item():.4f}")

    # --- Evaluate on held-out patients ---
    test_groups = group_by_time_signature([patients[i] for i in test_idx])
    with torch.no_grad():
        total_huber = 0.0
        total_sq = 0.0
        total_n = 0
        for g in test_groups:
            times_t = torch.tensor(g["times"], dtype=torch.float64)
            data_t = torch.tensor(g["values"], dtype=torch.float64)
            x0 = data_t[:, 0, :]
            pred = odeint(ode_fn, x0, times_t, method="dopri5").permute(1, 0, 2)
            total_huber += huber_loss(pred, data_t, delta=HUBER_DELTA, reduction="sum").item()
            total_sq += ((pred - data_t) ** 2).sum().item()
            total_n += data_t.numel()
        test_loss = total_huber / total_n
        test_mse = total_sq / total_n

    print(f"\n[STEP 4 -- FKPP FIT: {disease_name}]")
    print(f"  Fitted k = {k.item():.4f}, a = {a.item():.4f}")
    print(f"  Test Huber: {test_loss:.6f}")
    print(f"  Test MSE:   {test_mse:.6f}")
    return k.item(), a.item(), test_loss, test_mse


def step4(ad_patients, pd_patients, L_norm, train_idx, test_idx, ndm_results):
    """Fit FKPP for both diseases, then assert it beat NDM on held-out patients.

    The assertion is a hard gate: Step 5 is not built on a Step 4 that failed to
    justify itself. real_ad_pipeline warns instead, for the reason given there.

    The fit does not recover the synthetic truth exactly (AD: true k=0.80, a=0.70;
    fitted around k=0.66, a=0.56). k and a are partially interchangeable, the data
    carries measurement noise, and there are only four timepoints per patient. The
    claim rests on FKPP predicting better than NDM, which the gate tests directly.
    """
    print("\n" + "="*60)
    print("STEP 4 -- FKPP (Fisher-KPP)")
    print("="*60)

    # --- Fit both diseases ---
    ad_k, ad_a, ad_huber, ad_mse = fit_fkpp(ad_patients, L_norm, train_idx, test_idx, "AD")
    pd_k, pd_a, pd_huber, pd_mse = fit_fkpp(pd_patients, L_norm, train_idx, test_idx, "PD")

    # --- Compare the fitted parameters against the generator's values ---
    print(f"\n[FKPP vs Ground Truth]")
    print(f"  AD: fitted k={ad_k:.4f} (true 0.80), a={ad_a:.4f} (true 0.70)")
    print(f"  PD: fitted k={pd_k:.4f} (true 0.50), a={pd_a:.4f} (true 1.00)")

    # --- Enforce the hard gate: FKPP must beat NDM or the run stops ---
    ad_ndm_huber = ndm_results["AD"][1]
    pd_ndm_huber = ndm_results["PD"][1]
    print(f"\n[GATE CHECK]")
    print(f"  AD: FKPP Huber={ad_huber:.6f} vs NDM Huber={ad_ndm_huber:.6f} ->{'[PASS]' if ad_huber < ad_ndm_huber else '[FAIL] -- STOP'}")
    print(f"  PD: FKPP Huber={pd_huber:.6f} vs NDM Huber={pd_ndm_huber:.6f} ->{'[PASS]' if pd_huber < pd_ndm_huber else '[FAIL] -- STOP'}")
    assert ad_huber < ad_ndm_huber, f"FKPP did NOT beat NDM for AD: {ad_huber:.6f} >= {ad_ndm_huber:.6f}"
    assert pd_huber < pd_ndm_huber, f"FKPP did NOT beat NDM for PD: {pd_huber:.6f} >= {pd_ndm_huber:.6f}"

    return {"AD": (ad_k, ad_a, ad_huber, ad_mse), "PD": (pd_k, pd_a, pd_huber, pd_mse)}


# --- Real-data mode: Steps 2 to 4 on the OASIS-3 Alzheimer's cohort ---

def real_ad_pipeline(ad_patients_raw, ad_pids, L_norm):
    """Run the mean baseline, NDM and FKPP on the real AD patients.

    Takes patients holding RAW w-scores; returns a dict of the split, the transform
    record and all three results.

    The FKPP-vs-NDM gate is a WARNING here, unlike step4's assertion. On synthetic
    data, generated from FKPP itself, FKPP losing to NDM can only be a pipeline bug,
    so crashing is right. On real data the same outcome is a genuine finding about
    biology, and suppressing an unexpected real-patient result would be the wrong
    call, so the run continues and reports it.

    This is a separate function rather than step2/3/4 with PD set to None: threading
    None through them would add cross-disease branches to code that is currently
    simple and verified. Steps 5-7 are not run here -- they assume the paired
    cohort, and single-disease real data has not been scoped.
    """
    n_ad = len(ad_patients_raw)
    print(f"\n[REAL-DATA MODE] AD patients with >=2 timepoints: {n_ad}")

    # --- Convert raw w-scores to the model's burden scale ---
    ad_patients, norm_info = atrophy_score_normalize_patients(ad_patients_raw)

    # --- Split patients, using the same logic and seed as the synthetic path ---
    train_idx, val_idx, test_idx = patient_split(n_ad)
    print(f"\n[SPLIT] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} (of {n_ad} total)")
    print(f"  Test patient IDs: {sorted(ad_pids[test_idx])}")

    # --- Warn on splits too small for their metrics to mean much ---
    for name, split in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(split) < 5:
            print(f"  WARNING: {name} split has only {len(split)} patients (<5) -- "
                  f"metrics on this split are noisy and should not be over-interpreted")

    # --- Run the mean baseline ---
    print("\n" + "=" * 60)
    print("STEP 2 -- Mean Baseline (real AD)")
    print("=" * 60)
    base_huber, base_mse = mean_baseline(ad_patients, train_idx, test_idx, "AD", ad_pids)

    # --- Fit NDM ---
    print("\n" + "=" * 60)
    print("STEP 3 -- NDM (real AD)")
    print("=" * 60)
    ndm_k, ndm_huber, ndm_mse = fit_ndm(ad_patients, L_norm, train_idx, test_idx, "AD")

    # --- Fit FKPP ---
    print("\n" + "=" * 60)
    print("STEP 4 -- FKPP (real AD)")
    print("=" * 60)
    fkpp_k, fkpp_a, fkpp_huber, fkpp_mse = fit_fkpp(ad_patients, L_norm, train_idx, test_idx, "AD")

    # --- Check the gate, warning rather than stopping ---
    print(f"\n[GATE CHECK -- real AD]")
    print(f"  FKPP Huber={fkpp_huber:.6f} vs NDM Huber={ndm_huber:.6f} -> "
          f"{'[PASS]' if fkpp_huber < ndm_huber else '[WARN] FKPP did not beat NDM'}")
    if fkpp_huber >= ndm_huber:
        print(f"  NOTE: unlike the synthetic gate this is a WARNING, not a hard stop -- "
              f"this is the first real-patient run and an unexpected result here is itself "
              f"informative and should be surfaced, not suppressed.")

    # --- Print the three-model table ---
    print(f"\n[REAL AD -- Test-set performance, normalized-x scale]")
    print(f"  {'Model':<20} {'Huber':<15} {'MSE':<15}")
    print(f"  {'-'*50}")
    print(f"  {'Mean Baseline':<20} {base_huber:<15.6f} {base_mse:<15.6f}")
    print(f"  {'NDM (k)':<20} {ndm_huber:<15.6f} {ndm_mse:<15.6f}  k={ndm_k:.4f}")
    print(f"  {'FKPP (k,a)':<20} {fkpp_huber:<15.6f} {fkpp_mse:<15.6f}  k={fkpp_k:.4f}, a={fkpp_a:.4f}")

    # --- State the units caveat that must accompany these numbers ---
    print(f"\n  CAVEAT: k/a above are fit on atrophy-score-normalized w-scores "
          f"(x = clip(-w, 0, {norm_info['z_cap']}) / {norm_info['z_cap']}), NOT the same "
          f"units as the synthetic ground truth (k=0.80/a=0.70 for AD). "
          f"Do not compare these values directly to the synthetic table.")
    print(f"  Unlike the old min-max map this transform is PURE SCALING of the "
          f"disease-burden axis with a true zero, so healthy regions sit at x=0 and "
          f"inject no constant-offset drift through L_norm -- but the time axis and the "
          f"w-score SD unit still differ from the synthetic generator's units.")

    return {
        "split": (train_idx, val_idx, test_idx),
        "norm_info": norm_info,
        "baseline": (base_huber, base_mse),
        "ndm": (ndm_k, ndm_huber, ndm_mse),
        "fkpp": (fkpp_k, fkpp_a, fkpp_huber, fkpp_mse),
    }


# --- Step 5: the graph-constrained neural ODE ---

class EdgeGateNetwork(nn.Module):
    """Per-edge valve opening in (0, 1): 2 inputs -> 8 hidden -> tanh -> 1 -> sigmoid.

    Both inputs are unsigned -- the absolute burden difference across the edge and
    the edge weight -- so the valve opens identically in both directions. That is
    what makes the flux antisymmetric and therefore conservative; feeding the signed
    difference would let the network open wider one way and manufacture burden.

    The final bias is initialised to -3.0, so sigmoid gives about 0.05 and every
    valve starts nearly shut. Training therefore begins from pure FKPP and any
    correction has to be earned.
    """
    def __init__(self, hidden_dim=8):
        super().__init__()
        # --- Build the two-layer gate ---
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # --- Initialize the gates closed ---
        with torch.no_grad():
            self.net[-2].bias.fill_(-3.0)

    def forward(self, abs_diff, a_ij):
        """Map the two per-edge features to one valve opening per edge."""
        features = torch.stack([abs_diff, a_ij], dim=-1)
        return self.net(features).squeeze(-1)


class GraphConstrainedNeuralODE(nn.Module):
    """FKPP physics plus the constrained learned correction.

    Buffers: edge_i / edge_j, edge_weights, L_norm. Parameters: k (spreading), a
    (growth), w (82 per-region growth multipliers, initialised to ones so the model
    starts as plain FKPP), scale (one multiplier on the correction, initialised to
    0.01), and gate_net.

    W is symmetric, so both (i,j) and (j,i) appear in the edge list. The conservation
    argument in check_correction_conservation relies on that.
    """
    def __init__(self, L_norm_np, W_np, hidden_dim=8):
        super().__init__()

        # --- Register the edge list and matrices as fixed buffers ---
        edge_i, edge_j = np.nonzero(W_np)
        self.register_buffer("edge_i", torch.tensor(edge_i, dtype=torch.long))
        self.register_buffer("edge_j", torch.tensor(edge_j, dtype=torch.long))
        self.register_buffer("edge_weights", torch.tensor(W_np[edge_i, edge_j], dtype=torch.float64))
        self.register_buffer("L_norm", torch.tensor(L_norm_np, dtype=torch.float64))
        self.n_regions = L_norm_np.shape[0]
        self.n_edges = len(edge_i)

        # --- Create the trainable parameters, matching Step 4's starting values ---
        self.k = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.a = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.w = nn.Parameter(torch.ones(self.n_regions, dtype=torch.float64))
        self.gate_net = EdgeGateNetwork(hidden_dim=hidden_dim).double()
        self.scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

    def classical_base(self, x):
        """Return FKPP with a per-region growth rate: -k * L_norm @ x + a * w * x * (1 - x).

        The 82-long w broadcasts across the region dimension, so this works
        unchanged for a single state or a batch.
        """
        return -self.k * (x @ self.L_norm.T) + self.a * self.w * x * (1.0 - x)

    def neural_correction(self, x):
        """Return valve-controlled flux along edges, scaled by `scale`. Sums to zero always.

        The gate sees only unsigned features, so it is identical in both directions,
        while the signed difference flips when the endpoints swap. Hence
        flux(j,i) = -flux(i,j) exactly, and scattering each edge's flux into its
        source region leaves a total of zero across all 82 regions: the correction
        redistributes burden but never creates or destroys it.
        """
        is_batched = x.dim() > 1

        # --- Gather the burden at both ends of every edge ---
        if is_batched:
            x_i = x[:, self.edge_i]
            x_j = x[:, self.edge_j]
        else:
            x_i = x[self.edge_i]
            x_j = x[self.edge_j]

        abs_diff = torch.abs(x_j - x_i)
        signed_diff = x_j - x_i

        # --- Get the valve opening per edge, flattening the batch into one pass ---
        if is_batched:
            N, E = abs_diff.shape
            flat_abs = abs_diff.reshape(N * E)
            flat_w = self.edge_weights.unsqueeze(0).expand(N, -1).reshape(N * E)
            gates = self.gate_net(flat_abs, flat_w).reshape(N, E)
        else:
            gates = self.gate_net(abs_diff, self.edge_weights)

        # --- Form the antisymmetric flux ---
        flux = gates * signed_diff

        # --- Accumulate each edge's flux into its source region ---
        if is_batched:
            correction = torch.zeros(x.shape[0], self.n_regions, dtype=x.dtype, device=x.device)
            idx = self.edge_i.unsqueeze(0).expand(x.shape[0], -1)
            correction.scatter_add_(1, idx, flux)
        else:
            correction = torch.zeros(self.n_regions, dtype=x.dtype, device=x.device)
            correction.scatter_add_(0, self.edge_i, flux)

        return self.scale * correction

    def forward(self, t, x):
        """Return dx/dt for the solver: physics plus correction. t is unused."""
        return self.classical_base(x) + self.neural_correction(x)


def check_w_identity_sanity(model, x_batched):
    """Assert the per-region growth weights are a no-op at initialisation.

    Moving from a*x*(1-x) to a*w*x*(1-x) introduced a broadcast that, if aligned
    against the wrong axis, would still run and still produce plausible numbers
    while invalidating everything downstream. Since w starts as ones, the new
    formula must reproduce the old one to within 1e-12, batched and unbatched.
    """
    with torch.no_grad():
        # --- Compare the new formula against the pre-w formula, both shapes ---
        for label, x in [("batched", x_batched), ("unbatched", x_batched[0])]:
            actual = model.classical_base(x)
            expected = -model.k * (x @ model.L_norm.T) + model.a * x * (1.0 - x)
            max_diff = (actual - expected).abs().max().item()
            status = "PASS" if max_diff <= 1e-12 else "FAIL"
            print(f"  [w_i SANITY CHECK - {label}] max diff vs pre-w_i formula: {max_diff:.3e} [{status}]")
            if status == "FAIL":
                raise RuntimeError(
                    f"w_i sanity check FAILED ({label}): max diff {max_diff:.3e} exceeds 1e-12 tolerance"
                )


def compute_soft_monotonicity_penalty(pred_traj, reduction="mean"):
    """Charge predicted trajectories for regions that spontaneously improve.

    Neurodegeneration does not reverse, but measurements are noisy, so only
    decreases beyond TAU_MONO (about twice the measurement noise) are charged --
    that is what the relu implements. The "sum" mode also returns the element count
    so callers can combine groups into a correctly weighted average.
    """
    # --- Charge only decreases exceeding the tolerance band ---
    diffs = pred_traj[:-1] - pred_traj[1:]
    vals = relu(diffs - TAU_MONO)
    if reduction == "sum":
        return vals.sum(), vals.numel()
    return vals.mean()


def train_neural_ode(patients, L_norm_np, W_np, train_idx, val_idx, test_idx,
                     disease_name, lambda_reg=0.1, lambda_mono=0.01,
                     hidden_dim=8, n_epochs=100):
    """Train one graph-constrained neural ODE at one penalty strength.

    Returns (model, test Huber, test MSE, validation Huber). The objective is
    Huber + lambda_mono * monotonicity + lambda_reg * |correction|^2. The third term
    is the key one: without it the network would use its correction wherever it
    helped even slightly and would gradually take over from the physics.

    Three constraints keep the correction subordinate to the physics. It is
    ANTISYMMETRIC, so whatever it takes out of region i along an edge it puts into
    region j and it sums to zero by construction for any input, not as an outcome of
    training. It carries a LEARNABLE SCALE initialised at 0.01, so the data decides
    how much correction is warranted; on pure-FKPP data it settles near zero (about
    0.005 for AD). And its GATES ARE INITIALISED CLOSED at bias -3.0, so epoch 1 is
    effectively pure FKPP.

    Two optimiser groups reinforce that: the physics parameters (k, a, w) move
    freely at 1e-2 with no weight decay, while the neural parts learn ten times
    slower with weight decay pulling them toward zero.

    The correction-magnitude term is computed on detached states to avoid a second
    gradient path through the whole integration; the correction still receives
    gradient through the Huber term.
    """
    # --- Build the model and batch each split ---
    model = GraphConstrainedNeuralODE(L_norm_np, W_np, hidden_dim=hidden_dim).to(DEVICE)
    train_groups = group_by_time_signature([patients[i] for i in train_idx])
    val_groups = group_by_time_signature([patients[i] for i in val_idx])
    test_groups = group_by_time_signature([patients[i] for i in test_idx])

    # --- Confirm the per-region weights are a no-op before any training ---
    x0_sample = torch.tensor(train_groups[0]["values"][:, 0, :], dtype=torch.float64, device=DEVICE)
    check_w_identity_sanity(model, x0_sample)

    # --- Set up the two optimizer groups ---
    optimizer = torch.optim.Adam([
        {"params": [model.k, model.a, model.w], "lr": 1e-2, "weight_decay": 0.0},
        {"params": list(model.gate_net.parameters()) + [model.scale], "lr": 1e-3, "weight_decay": 1e-3},
    ])

    # --- Use fixed-step rk4; dopri5 is for irregular real-data scan gaps ---
    ode_kwargs = dict(method="rk4", options=dict(step_size=0.25))

    # --- Train the model ---
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # --- Reset the running sums and element counts for this epoch ---
        total_h = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
        total_n = 0
        total_mono_sum = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
        total_mono_n = 0
        corr_mag_sum = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
        corr_mag_n = 0

        for g in train_groups:
            # --- Integrate this group forward from its first scan ---
            times_t = torch.tensor(g["times"], dtype=torch.float64, device=DEVICE)
            data_t = torch.tensor(g["values"], dtype=torch.float64, device=DEVICE)
            x0 = data_t[:, 0, :]
            pred = odeint(model, x0, times_t, **ode_kwargs)
            pred_ntd = pred.permute(1, 0, 2)

            # --- Accumulate the prediction error ---
            total_h = total_h + huber_loss(pred_ntd, data_t, delta=HUBER_DELTA, reduction="sum")
            total_n += data_t.numel()

            # --- Accumulate the monotonicity penalty over consecutive timepoints ---
            mono_s, mono_n = compute_soft_monotonicity_penalty(pred, reduction="sum")
            total_mono_sum = total_mono_sum + mono_s
            total_mono_n += mono_n

            # --- Accumulate the correction magnitude on detached states ---
            pred_detached = pred_ntd.detach()
            Tg = pred_detached.shape[1]
            for t_idx in range(Tg):
                corr = model.neural_correction(pred_detached[:, t_idx, :])
                corr_mag_sum = corr_mag_sum + (corr ** 2).sum()
                corr_mag_n += corr.numel()

        # --- Combine the three terms into the objective ---
        h = total_h / total_n
        mono_pen = total_mono_sum / total_mono_n
        corr_mag = corr_mag_sum / corr_mag_n
        loss = h + lambda_mono * mono_pen + lambda_reg * corr_mag

        # --- Abort on a NaN or infinite loss ---
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [{disease_name} NeuralODE lam_reg={lambda_reg}] ABORT epoch {epoch+1}: "
                  f"loss={loss.item()}, k={model.k.item():.4f}, a={model.a.item():.4f}")
            raise RuntimeError(f"NaN/Inf loss at epoch {epoch+1}")

        # --- Clip the gradient so one bad batch cannot throw the parameters ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- Step the optimizer and clamp both physics rates ---
        optimizer.step()
        with torch.no_grad():
            model.k.clamp_(1e-3, 5.0)
            model.a.clamp_(1e-3, 5.0)

        # --- Log progress, watching `scale` stay near zero on pure-FKPP data ---
        if (epoch + 1) % 20 == 0 or (epoch + 1) == 1:
            print(f"  [{disease_name} NeuralODE lam_reg={lambda_reg}] epoch {epoch+1}/{n_epochs}, "
                  f"Huber={h.item():.6f}, k={model.k.item():.4f}, a={model.a.item():.4f}, "
                  f"scale={model.scale.item():.6f}", flush=True)

    model.eval()

    def eval_groups(groups):
        """Run the trained model over one split and return (Huber, MSE)."""
        with torch.no_grad():
            total_huber = 0.0
            total_sq = 0.0
            total_n = 0
            for g in groups:
                times_t = torch.tensor(g["times"], dtype=torch.float64, device=DEVICE)
                data_t = torch.tensor(g["values"], dtype=torch.float64, device=DEVICE)
                x0 = data_t[:, 0, :]
                pred = odeint(model, x0, times_t, **ode_kwargs).permute(1, 0, 2)
                total_huber += huber_loss(pred, data_t, delta=HUBER_DELTA, reduction="sum").item()
                total_sq += ((pred - data_t) ** 2).sum().item()
                total_n += data_t.numel()
            return total_huber / total_n, total_sq / total_n

    # --- Score validation, which selects lambda_reg, and test, which only reports ---
    val_huber, _ = eval_groups(val_groups)
    test_huber, test_mse = eval_groups(test_groups)

    # --- Report the spread of the learned per-region growth weights ---
    w = model.w.detach().cpu().numpy()
    print(f"  [{disease_name} NeuralODE lam_reg={lambda_reg}] VAL Huber={val_huber:.6f}, "
          f"TEST Huber={test_huber:.6f}, MSE={test_mse:.6f}, scale={model.scale.item():.6f}", flush=True)
    print(f"  [{disease_name} NeuralODE lam_reg={lambda_reg}] w stats: "
          f"mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}", flush=True)
    return model, test_huber, test_mse, val_huber


def check_correction_conservation(model, x_samples, disease_name):
    """Verify numerically that the learned correction only redistributes burden.

    Scope: the FULL model does not conserve total burden and is not meant to. The
    growth term a*w*x*(1-x) is non-negative on [0,1], so total burden rises whenever
    any region is partly diseased -- that is the biology. A PASS here means the
    correction stayed a pure redistribution, never that total burden is constant.

    The property holds for every state, not only the sampled ones: both (i,j) and
    (j,i) are in the edge list and the gate depends only on symmetric inputs, so the
    two directions share a valve opening while the signed difference flips, making
    flux(j,i) = -flux(i,j) identically. The samples check that the code matches that
    argument to finite precision.

    The per-edge pass is a second, finer check: a zero total could in principle hide
    individual edges misbehaving and cancelling out.
    """
    model.eval()
    with torch.no_grad():
        # --- Check the correction sums to zero across all 82 regions ---
        print(f"\n[CORRECTION CONSERVATION -- {disease_name}]")
        for s_idx, x_sample in enumerate(x_samples):
            x = torch.tensor(x_sample, dtype=torch.float64, device=DEVICE)
            correction = model.neural_correction(x)
            total_sum = correction.sum().item()
            max_abs = correction.abs().max().item()
            ok = abs(total_sum) < 1e-5
            print(f"  state {s_idx}: sum={total_sum:.2e}, max|corr_i|={max_abs:.6f} "
                  f"{'[PASS]' if ok else '[FAIL]'}")
            assert ok, (f"Neural-correction conservation violated ({disease_name}, "
                        f"state {s_idx}): sum={total_sum}")

        # --- Recompute the per-edge flux on the last sampled state ---
        print(f"  [Flux Antisymmetry Check -- {disease_name}]")
        edge_i = model.edge_i
        edge_j = model.edge_j
        x_i = x[edge_i]
        x_j = x[edge_j]
        abs_diff = torch.abs(x_j - x_i)
        signed_diff = x_j - x_i
        gates = model.gate_net(abs_diff, model.edge_weights)
        flux = gates * signed_diff

        # --- Check sampled edge pairs have exactly opposite fluxes ---
        rng = np.random.RandomState(42)
        checked = 0
        pairs_seen = set()
        for _ in range(100):
            idx = rng.randint(0, len(edge_i))
            i_node = edge_i[idx].item()
            j_node = edge_j[idx].item()
            pair = (min(i_node, j_node), max(i_node, j_node))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)

            # --- Locate both directions of this edge in the flat edge list ---
            mask_ij = (edge_i == i_node) & (edge_j == j_node)
            mask_ji = (edge_i == j_node) & (edge_j == i_node)
            if mask_ij.any() and mask_ji.any():
                f_ij = flux[mask_ij][0].item()
                f_ji = flux[mask_ji][0].item()
                ok = abs(f_ij + f_ji) < 1e-6
                if checked < 5:
                    print(f"    edge ({i_node},{j_node}): flux_ij={f_ij:+.6f}, flux_ji={f_ji:+.6f}, "
                          f"sum={f_ij+f_ji:.2e} {'[PASS]' if ok else '[FAIL]'}")
                assert ok, f"Antisymmetry violated: ({i_node},{j_node}) flux_ij={f_ij}, flux_ji={f_ji}"
                checked += 1
        print(f"    ... verified {checked} edge pairs total, all antisymmetric [PASS]")


def step5(ad_patients, pd_patients, L_norm, W, train_idx, val_idx, test_idx, fkpp_results):
    """Train the neural ODE at two penalty strengths, run the dial test, check conservation.

    The dial test trains at lambda_reg = 0.1 and at 1.0. Under the stronger penalty
    the model should slide back toward plain FKPP -- scale shrinks, k and a move
    toward the Step 4 values -- showing the neural part is a controllable knob rather
    than an opaque black box.

    Which penalty wins is decided purely on validation; test numbers are printed but
    never used to choose, since selecting on test would make the reported test number
    optimistic.
    """
    print("\n" + "="*60)
    print("STEP 5 -- Graph-Constrained Neural ODE")
    print("="*60)

    # --- Set the two ends of the dial ---
    lambda_reg_small = 0.1
    lambda_reg_large = 1.0

    results = {}
    models = {}
    for disease, patients in [("AD", ad_patients), ("PD", pd_patients)]:
        print(f"\n--- {disease} ---")
        fkpp_test_huber = fkpp_results[disease][2]

        # --- Train at the weak penalty, reseeding so both runs start identically ---
        print(f"\n  Training with lam_reg={lambda_reg_small} (small):")
        torch.manual_seed(SEED)
        model_small, huber_small, mse_small, val_small = train_neural_ode(
            patients, L_norm, W, train_idx, val_idx, test_idx,
            disease, lambda_reg=lambda_reg_small)

        # --- Train at the strong penalty from the same starting weights ---
        print(f"\n  Training with lam_reg={lambda_reg_large} (large):")
        torch.manual_seed(SEED)
        model_large, huber_large, mse_large, val_large = train_neural_ode(
            patients, L_norm, W, train_idx, val_idx, test_idx,
            disease, lambda_reg=lambda_reg_large)

        # --- Measure how far each run sits from pure FKPP ---
        delta_small = abs(huber_small - fkpp_test_huber)
        delta_large = abs(huber_large - fkpp_test_huber)
        scale_small = model_small.scale.item()
        scale_large = model_large.scale.item()
        k_small = model_small.k.item()
        k_large = model_large.k.item()
        a_small = model_small.a.item()
        a_large = model_large.a.item()

        # --- Select on validation only, ties going to the smaller penalty ---
        if val_small <= val_large:
            sel_lambda, sel_model = lambda_reg_small, model_small
            sel_huber, sel_mse, sel_scale = huber_small, mse_small, scale_small
        else:
            sel_lambda, sel_model = lambda_reg_large, model_large
            sel_huber, sel_mse, sel_scale = huber_large, mse_large, scale_large

        # --- Print the dial table ---
        print(f"\n  [DIAL CHECK -- {disease}]")
        print(f"  {'lam_reg':<10} {'Val Huber':<15} {'Test Huber':<15} {'|D vs FKPP|':<15} {'k':<10} {'a':<10} {'scale':<12}")
        print(f"  {lambda_reg_small:<10} {val_small:<15.6f} {huber_small:<15.6f} {delta_small:<15.6f} {k_small:<10.4f} {a_small:<10.4f} {scale_small:<12.6f}")
        print(f"  {lambda_reg_large:<10} {val_large:<15.6f} {huber_large:<15.6f} {delta_large:<15.6f} {k_large:<10.4f} {a_large:<10.4f} {scale_large:<12.6f}")
        print(f"  Pure FKPP: k={fkpp_results[disease][0]:.4f}, a={fkpp_results[disease][1]:.4f}, test Huber={fkpp_test_huber:.6f}")
        print(f"  SELECTED by val: lam_reg={sel_lambda} (val Huber {min(val_small, val_large):.6f})")

        # --- Pass the dial check if the strong penalty lands within 1% of FKPP ---
        if delta_large > 0.01 * fkpp_test_huber + 0.001:
            print(f"  WARNING: large lam_reg not converging to FKPP -- delta={delta_large:.6f}")
        else:
            print(f"  Dial check [PASS] -- large lam_reg approximates pure FKPP")

        # --- Store both runs and the selected one ---
        results[disease] = {
            "small": (huber_small, mse_small, scale_small),
            "large": (huber_large, mse_large, scale_large),
            "selected": (sel_huber, sel_mse, sel_scale),
            "selected_lambda": sel_lambda,
        }
        models[disease] = {"small": model_small, "large": model_large, "selected": sel_model}

    print("\n" + "="*60)
    print("PHYSICS CHECK -- NEURAL-CORRECTION CONSERVATION (numerical)")
    print("  (correction term only -- the FKPP growth term adds burden by design)")
    print("="*60)
    for disease, patients in [("AD", ad_patients), ("PD", pd_patients)]:
        def sample(pos, t_idx):
            """Take patient `pos` of the test set at timepoint `t_idx`, index clamped."""
            vals = patients[test_idx[pos]]["values"]
            return vals[min(t_idx, vals.shape[0] - 1), :]

        # --- Check the correction sums to zero on three different states ---
        x_samples = [sample(0, 1), sample(1, 2), sample(2, 3)]
        for lam_label in ["small", "large"]:
            check_correction_conservation(models[disease][lam_label], x_samples,
                                          f"{disease} (lam_reg {lam_label})")

    return results, models


# --- Step 6: the comparison table ---

def step6(ad_base, pd_base, ndm_results, fkpp_results, node_results):
    """Print the four-model comparison table for each disease.

    All four models are scored on the same held-out patients, lower is better. This
    is evidence of pipeline correctness, not a biological result: the synthetic data
    came from an FKPP equation, so FKPP-shaped answers are expected by construction,
    and the correction's internal weights do not map onto k and a.
    """
    print("\n" + "="*60)
    print("STEP 6 -- Pipeline-Correctness Comparison Table")
    print("  (validates pipeline correctness, NOT biology;")
    print("   neural correction weights do NOT map to k/a)")
    print("="*60)

    for disease in ["AD", "PD"]:
        # --- Unpack each step's stored numbers ---
        base_huber, base_mse = (ad_base if disease == "AD" else pd_base)
        ndm_huber, ndm_mse = ndm_results[disease][1], ndm_results[disease][2]
        fkpp_huber, fkpp_mse = fkpp_results[disease][2], fkpp_results[disease][3]
        node_huber, node_mse, node_scale = node_results[disease]["selected"]
        node_lam = node_results[disease]["selected_lambda"]

        # --- Print the four rows ---
        print(f"\n  [{disease}] Test-set performance (same held-out patients):")
        print(f"  {'Model':<30} {'Huber':<15} {'MSE':<15}")
        print(f"  {'-'*60}")
        print(f"  {'Mean Baseline':<30} {base_huber:<15.6f} {base_mse:<15.6f}")
        print(f"  {'NDM':<30} {ndm_huber:<15.6f} {ndm_mse:<15.6f}")
        print(f"  {'FKPP':<30} {fkpp_huber:<15.6f} {fkpp_mse:<15.6f}")
        print(f"  {f'Graph-Constr. NODE (lam={node_lam})':<30} {node_huber:<15.6f} {node_mse:<15.6f}")

        # --- Pass if the neural ODE matches or beats FKPP, which is the ceiling here ---
        if node_huber <= fkpp_huber:
            print(f"  -> Neural ODE <= FKPP Huber [PASS] (pipeline correctness evidence)")
        else:
            print(f"  -> Neural ODE > FKPP Huber (delta={node_huber - fkpp_huber:.6f}) -- may indicate capacity/reg tradeoff")


# --- Step 7: written summary ---

def step7(node_results):
    """Print the final summary and the explicit out-of-scope list.

    Both selected scales should sit near zero on pure-FKPP data: the model reporting
    that the physics was already sufficient.
    """
    print("\n" + "="*60)
    print("STEP 7 -- SUMMARY")
    print("="*60)

    # --- Pull the selected correction scale and penalty for each disease ---
    ad_scale = node_results["AD"]["selected"][2]
    pd_scale = node_results["PD"]["selected"][2]
    ad_lam = node_results["AD"]["selected_lambda"]
    pd_lam = node_results["PD"]["selected_lambda"]
    print(f"""
BUILT:
  - Data loading + validation pipeline (3 CSVs + ground truth JSON)
  - Irregular-timepoint patient loader (list of per-patient {{times, values}} dicts;
      minimum 2 timepoints, no fixed grid required -- ready for real OASIS/ADNI/PPMI
      scan intervals; synthetic CSVs still resolve to a single time-signature group)
  - Normalized Laplacian construction with eigenvalue verification
  - Mean-stage baseline (the floor)
  - NDM fit (linear diffusion only: dx/dt = -k*L_norm@x)
  - FKPP fit (dx/dt = -k*L_norm@x + a*x*(1-x))
  - Graph-Constrained Neural ODE with:
      Per-region growth-rate weights w_i (init 1.0, verified via exact sanity check
        against the pre-w_i formula before any training)
      Edge-gated antisymmetric flux CORRECTION that is conservative by construction
        (it only redistributes burden along edges). The full model is NOT
        mass-conserving -- the FKPP growth term adds burden, which is the biology.
      Learnable scalar `scale` on the correction (init 0.01); on pure-FKPP data it
        stays near zero -- AD selected scale={ad_scale:.6f} (lam_reg={ad_lam}),
        PD selected scale={pd_scale:.6f} (lam_reg={pd_lam})
      Soft monotonicity penalty (tau=0.04, lam_mono=0.01)
      hidden_dim=8, weight_decay=1e-3
      rk4 fixed-step solver (step_size=0.25); dopri5 reserved for irregular
        real-data scan intervals
  - Neural-correction conservation check (correction sum ~= 0, flux antisymmetry
      verified). Scoped to the correction term only -- total burden is NOT conserved.
  - Regularizer dial check (large lam_reg -> pure FKPP behavior)
  - lam_reg selected on validation set; test numbers reported, never selected on
  - Pipeline-correctness comparison table

EXPLICITLY OUT OF SCOPE (do NOT begin):
  - Cross-disease transfer
  - Full-freeze control
  - Disease-specific refit ceiling
  - Null tournament
  - Real ADNI/PPMI/OASIS data loading
  - neuroCombat harmonization
  - Distance matrix
""")


# --- Main ---

if __name__ == "__main__":
    # --- Read the data-source flag ---
    parser = argparse.ArgumentParser(description="CrossSpread-ODE pipeline")
    parser.add_argument("--data-source", choices=["synthetic", "real"], default="synthetic",
                         help="synthetic: data/synthetic/*.csv | real: data/real/*.csv "
                              "(never mixed within one run)")
    args = parser.parse_args()

    # --- Load and validate the inputs ---
    gt, W, L_norm, ad_patients, pd_patients, ad_pids, pd_pids = load_and_validate(args.data_source)

    # --- Branch to the AD-only real path, which skips Steps 5-7 ---
    if args.data_source == "real":
        print("\n" + "=" * 60)
        print("REAL-DATA MODE: Steps 2-4 only (mean baseline, NDM, FKPP), AD-only.")
        print("Steps 5-7 (Neural ODE, pipeline-correctness table, summary) assume a")
        print("paired AD/PD cohort and are NOT run in real-data mode -- real data")
        print("currently has AD only, and cross-disease/Neural-ODE steps on")
        print("single-disease real data have not been scoped yet.")
        print("=" * 60)
        real_ad_pipeline(ad_patients, ad_pids, L_norm)
        raise SystemExit(0)

    # --- Step 2: fix the split and get the baseline floor ---
    train_idx, val_idx, test_idx, ad_base, pd_base = step2(ad_patients, pd_patients, ad_pids, pd_pids)

    # --- Step 3: fit NDM ---
    ndm_results = step3(ad_patients, pd_patients, L_norm, train_idx, test_idx)

    # --- Step 4: fit FKPP, gated against NDM ---
    fkpp_results = step4(ad_patients, pd_patients, L_norm, train_idx, test_idx, ndm_results)

    # --- Step 5: train the neural ODE and run the dial test ---
    node_results, node_models = step5(ad_patients, pd_patients, L_norm, W, train_idx, val_idx, test_idx, fkpp_results)

    # --- Step 6: print the comparison table ---
    step6(ad_base, pd_base, ndm_results, fkpp_results, node_results)

    # --- Step 7: print the summary ---
    step7(node_results)
