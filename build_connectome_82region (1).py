"""Remap the downloaded 84-region tractography matrix into the project's canonical
82-region order, then re-run pipeline.py's structural validity checks on the result.

The source file and this project disagree on region count (84 vs 82), naming
("Bankssts_L" vs "L_bankssts") and ordering, so row 5 of theirs is not row 5 of
ours. data/real/tractography.csv is read only here; pipeline.py loads only the
output, data/real/connectome_82region.csv, and will not fall back to the raw file.

    python build_connectome_82region.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
REAL_DIR = BASE_DIR / "data" / "real"

SUBCORTICAL_NAME_OVERRIDE = {
    "thalamus_proper": "thalamus",
    "accumbens_area": "accumbens",
}

EXPECTED_DROPPED = {"Cerebellum_Cortex_L", "Cerebellum_Cortex_R"}


def rename_region(raw_name):
    """Convert a RegionList.csv name to our L_/R_<name> convention, or None.

    Two deep structures do not lowercase into our canonical names
    ("Thalamus_Proper_L" -> "thalamus", "Accumbens_Area_L" -> "accumbens"), so they
    are substituted explicitly. Returning None for a name with no hemisphere suffix
    rather than guessing routes it into the drop assertion in main(), instead of
    letting an unrecognised region slip through as if it had mapped.
    """
    if raw_name.endswith("_L"):
        hemi, base = "L", raw_name[:-2]
    elif raw_name.endswith("_R"):
        hemi, base = "R", raw_name[:-2]
    else:
        return None
    base_lower = base.lower()
    base_lower = SUBCORTICAL_NAME_OVERRIDE.get(base_lower, base_lower)
    return f"{hemi}_{base_lower}"


def main():
    """Remap the raw 84-region tractography matrix to verified 82-region output.

    The two cerebellum regions are dropped because the 82-region atlas covers
    cerebral cortex and deep grey structures only. EXPECTED_DROPPED is asserted BY
    NAME rather than by count: a count check would still pass if a cortical region
    failed to map on a naming quirk while a cerebellum region mapped by accident, so
    we would silently discard real cortex and keep a phantom region in its place.

    The validation block repeats the checks pipeline.py runs, at the point of
    creation, so a broken file never leaves this script.
    """
    print("=" * 60)
    print("build_connectome_82region.py")
    print("=" * 60)

    # --- Load the canonical region order ---
    with open(CONFIG_DIR / "ground_truth_params.json") as f:
        gt = json.load(f)
    region_order = gt["region_order"]
    assert len(region_order) == 82, f"Expected 82 canonical regions, got {len(region_order)}"

    # --- Load the region list, which defines the matrix row and column order ---
    region_list = pd.read_csv(REAL_DIR / "RegionList.csv")["region_name"].tolist()
    print(f"\n[LOAD] RegionList.csv: {len(region_list)} regions")

    # --- Load the tractography matrix and check it is square against that list ---
    tract = pd.read_csv(REAL_DIR / "tractography.csv", header=None)
    assert tract.shape == (len(region_list), len(region_list)), (
        f"tractography.csv shape {tract.shape} != RegionList length {len(region_list)}"
    )
    print(f"[LOAD] tractography.csv: {tract.shape}")

    # --- Rename every region and collect those outside the canonical atlas ---
    renamed = [rename_region(r) for r in region_list]
    canonical_set = set(region_order)
    dropped = [raw for raw, new in zip(region_list, renamed) if new not in canonical_set]

    print(f"\n[RENAME] {len(region_list) - len(dropped)}/{len(region_list)} regions map into the canonical atlas")

    # --- Drop the 2 cerebellum regions, asserted by name ---
    assert set(dropped) == EXPECTED_DROPPED, (
        f"Expected to drop exactly {EXPECTED_DROPPED}, but got {set(dropped)} "
        f"-- STOP, naming assumptions are wrong, do not proceed"
    )
    print(f"[DROP] confirmed exactly 2 extra regions: {sorted(dropped)}")

    # --- Build the reverse lookup and confirm all 82 regions are covered ---
    mapped = {new: raw for raw, new in zip(region_list, renamed) if new in canonical_set}
    assert len(mapped) == 82, f"Expected 82 mapped regions, got {len(mapped)}"
    assert set(mapped.keys()) == canonical_set, "Mapped region names != ground truth region_order set"

    # --- Label the raw matrix by name so the reorder addresses cells by name ---
    tract.index = region_list
    tract.columns = region_list

    # --- Reorder matrix to canonical region order, both axes together ---
    ordered_raw_names = [mapped[name] for name in region_order]
    W = tract.loc[ordered_raw_names, ordered_raw_names].values.astype(np.float64)
    assert W.shape == (82, 82)

    # --- Write canonical names onto the output and save ---
    out_df = pd.DataFrame(W, index=region_order, columns=region_order)
    out_path = REAL_DIR / "connectome_82region.csv"
    out_df.to_csv(out_path)
    print(f"\n[SAVED] {out_path} -- {out_df.shape[0]}x{out_df.shape[1]}, region order verified vs ground_truth_params.json")

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # --- Collect all check results so one run reports every failure ---
    checks = []

    # --- Check the matrix equals its own transpose ---
    is_symmetric = np.allclose(W, W.T)
    checks.append(("Symmetric", is_symmetric))
    print(f"  Symmetric: {'[PASS]' if is_symmetric else '[FAIL]'} (max |W - W.T| = {np.abs(W - W.T).max():.3e})")

    # --- Check no region wires to itself ---
    diag_zero = np.allclose(np.diag(W), 0)
    checks.append(("Zero diagonal", diag_zero))
    print(f"  Zero diagonal: {'[PASS]' if diag_zero else '[FAIL]'} (max |diag| = {np.abs(np.diag(W)).max():.3e})")

    # --- Check every region has nonzero degree ---
    deg = W.sum(axis=1)
    no_isolated = (deg > 0).all()
    checks.append(("No isolated nodes", no_isolated))
    n_isolated = int((deg <= 0).sum())
    print(f"  No isolated nodes: {'[PASS]' if no_isolated else '[FAIL]'} ({n_isolated} isolated nodes, min degree = {deg.min():.6f})")

    # --- Build the normalized Laplacian and check its eigenvalues lie in [0, 2] ---
    if no_isolated:
        dinv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L_norm = np.eye(len(region_order)) - dinv_sqrt @ W @ dinv_sqrt
        L_sym = np.allclose(L_norm, L_norm.T, atol=1e-12)
        checks.append(("L_norm symmetric", L_sym))
        eigs = eigvalsh(L_norm)
        eig_ok = eigs.min() >= -1e-10 and eigs.max() <= 2.0 + 1e-10
        checks.append(("L_norm eigenvalues in [0,2]", eig_ok))
        print(f"  L_norm symmetric: {'[PASS]' if L_sym else '[FAIL]'}")
        print(f"  L_norm eigenvalue range: [{eigs.min():.6f}, {eigs.max():.6f}] -> {'[PASS]' if eig_ok else '[FAIL]'}")
    else:
        # --- Record the skipped Laplacian check as a failure ---
        print("  L_norm eigenvalues: SKIPPED (isolated nodes present, degree normalization undefined)")
        checks.append(("L_norm eigenvalues in [0,2]", False))

    # --- Raise if any check failed, leaving the file on disk for debugging ---
    all_pass = all(ok for _, ok in checks)
    print(f"\n[OVERALL] {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED -- DO NOT USE THIS FILE'}")
    if not all_pass:
        raise RuntimeError("connectome_82region.csv failed validation -- see checks above")

    return out_df


if __name__ == "__main__":
    main()
