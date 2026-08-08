"""Convert raw OASIS-3 FreeSurfer measurements into the model-ready CSV that
pipeline.py consumes.

Cortical thickness (mm) and subcortical volume (mm^3) are incomparable units, so
every measurement becomes a w-score against the cognitively normal controls, per
region. The FreeSurfer export carries no diagnosis, so the UDSd1 clinical file is
joined in.

The join matches each scan to the nearest clinical visit within a window rather
than taking each subject's first record. A diagnosis is not fixed -- normal in
2005, MCI in 2008, probable AD in 2011 -- so applying the first record to all of a
subject's scans both mislabels later scans and discards the longitudinal AD scans
this project needs most. WINDOW_DAYS_COMPARISON reports counts at both 180 and 365
days so the sensitivity of the patient count to that arbitrary window is on the
record rather than silently depended upon; WINDOW_DAYS is the value actually used.

Output is data/real/Ad_atrophy_real.csv, holding RAW (unflipped) w-scores.
pipeline.py applies atrophy_score_normalize on load, so the transform is applied
exactly once and the saved file stays inspectable.

    python prep_oasis.py

Requires data/raw_oasis/OASIS3_Freesurfer_output.csv,
data/raw_oasis/OASIS3_UDSd1_diagnoses.csv and config/ground_truth_params.json.
Runs entirely locally; no patient row is transmitted anywhere.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
RAW_OASIS_DIR = BASE_DIR / "data" / "raw_oasis"
REAL_DATA_DIR = BASE_DIR / "data" / "real"

SUBJECT_COL = "Subject"          # e.g. "OAS30001"
SESSION_COL = "MR_session"       # e.g. "OAS30001_MR_d0757", "_d0757" = days since entry

CLINICAL_CSV_PATH = RAW_OASIS_DIR / "OASIS3_UDSd1_diagnoses.csv"
CLINICAL_SUBJECT_COL = "OASISID"
CLINICAL_DAY_COL = "days_to_visit"
DIAGNOSIS_COL = "NORMCOG"              # cognitively normal -> control group
AD_DIAGNOSIS_COL = "PROBAD"            # probable Alzheimer's -> patient group
CONTROL_LABEL = 1
AD_LABEL = 1

WINDOW_DAYS = 365
WINDOW_DAYS_COMPARISON = [180, 365]

CORTICAL_THICKNESS_PREFIX_LH = "lh_"
CORTICAL_THICKNESS_PREFIX_RH = "rh_"
CORTICAL_THICKNESS_SUFFIX = "_thickness"
SUBCORTICAL_PREFIX_L = "Left-"
SUBCORTICAL_PREFIX_R = "Right-"
SUBCORTICAL_SUFFIX = "_volume"

DK_CORTICAL = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]

SUBCORTICAL = [
    "thalamus", "caudate", "putamen", "pallidum",
    "hippocampus", "amygdala", "accumbens",
]

REGION_ORDER = (
    [f"L_{r}" for r in DK_CORTICAL] +
    [f"R_{r}" for r in DK_CORTICAL] +
    [f"L_{r}" for r in SUBCORTICAL] +
    [f"R_{r}" for r in SUBCORTICAL]
)
assert len(REGION_ORDER) == 82, f"Region count is {len(REGION_ORDER)}, expected 82"


SUBCORTICAL_NAME_OVERRIDE = {
    "accumbens": "Accumbens-area",
    "thalamus": "Thalamus-Proper",
}

def build_column_map(df_columns):
    """Map each of our 82 region names to its raw FreeSurfer column name.

    Returns (col_map, unmapped). Several candidate spellings are tried per region
    so the script survives minor naming differences between FreeSurfer versions.
    Two subcortical structures need explicit overrides because no capitalisation
    rule produces their FreeSurfer names ("Left-Accumbens-area_volume",
    "Left-Thalamus-Proper_volume"). Anything still unmatched is returned rather
    than skipped, because a missing region is a hole in the network, and main()
    stops on it.
    """
    col_set = set(df_columns)
    col_map = {}
    unmapped = []

    for region in REGION_ORDER:
        hemi, base = region.split("_", 1)

        # --- Build candidate column names for a subcortical volume ---
        if base in SUBCORTICAL:
            fs_name = SUBCORTICAL_NAME_OVERRIDE.get(base, base.capitalize())
            if hemi == "L":
                candidates = [
                    f"{SUBCORTICAL_PREFIX_L}{fs_name}{SUBCORTICAL_SUFFIX}",
                    f"{SUBCORTICAL_PREFIX_L}{base.capitalize()}{SUBCORTICAL_SUFFIX}",
                    f"{SUBCORTICAL_PREFIX_L}{base.title()}{SUBCORTICAL_SUFFIX}",
                ]
            else:
                candidates = [
                    f"{SUBCORTICAL_PREFIX_R}{fs_name}{SUBCORTICAL_SUFFIX}",
                    f"{SUBCORTICAL_PREFIX_R}{base.capitalize()}{SUBCORTICAL_SUFFIX}",
                    f"{SUBCORTICAL_PREFIX_R}{base.title()}{SUBCORTICAL_SUFFIX}",
                ]
        # --- Build candidate column names for a cortical thickness ---
        else:
            if hemi == "L":
                candidates = [
                    f"{CORTICAL_THICKNESS_PREFIX_LH}{base}{CORTICAL_THICKNESS_SUFFIX}",
                    f"lh_{base}_thickness",
                    f"lh_{base}",
                ]
            else:
                candidates = [
                    f"{CORTICAL_THICKNESS_PREFIX_RH}{base}{CORTICAL_THICKNESS_SUFFIX}",
                    f"rh_{base}_thickness",
                    f"rh_{base}",
                ]

        # --- Take the first candidate present in the file ---
        matched = None
        for c in candidates:
            if c in col_set:
                matched = c
                break

        # --- Record the match, or the attempted names for the error message ---
        if matched:
            col_map[region] = matched
        else:
            unmapped.append((region, candidates))

    return col_map, unmapped


def match_diagnosis_to_scans(df, clinical_df, window_days):
    """Label every scan as control, Alzheimer's, or neither.

    Returns four boolean masks aligned to df: control_mask, ad_mask (either route),
    ad_direct_mask (a real nearby clinical visit) and ad_forward_mask (forward-fill).

    Once a subject has a confirmed PROBAD=1 visit, the AD label is forward-filled to
    their LATER scans only. AD does not reverse, so propagation is strictly forward;
    an earlier scan is never relabelled by a later diagnosis, because at that date
    the subject may genuinely not have met the criteria. The two routes are returned
    separately because a direct in-window match is stronger evidence than a
    forward-filled one.
    """
    # --- Sort both sides by day for the as-of merge ---
    left = df[["_join_id", "_mri_day"]].reset_index().sort_values("_mri_day")
    right = (
        clinical_df.dropna(subset=[CLINICAL_DAY_COL])
        .sort_values(CLINICAL_DAY_COL)
    )

    # --- Join diagnosis to scans by nearest visit within the window ---
    merged = pd.merge_asof(
        left,
        right[["_join_id", CLINICAL_DAY_COL, DIAGNOSIS_COL, AD_DIAGNOSIS_COL]],
        left_on="_mri_day", right_on=CLINICAL_DAY_COL,
        by="_join_id", direction="nearest", tolerance=float(window_days),
    ).set_index("index").sort_index()

    # --- Read the two diagnosis flags, treating no match as False ---
    control_mask = (merged[DIAGNOSIS_COL] == CONTROL_LABEL).reindex(df.index, fill_value=False)
    ad_direct_mask = (merged[AD_DIAGNOSIS_COL] == AD_LABEL).reindex(df.index, fill_value=False)

    # --- Find each subject's earliest confirmed probable-AD day ---
    first_ad_day = (
        clinical_df.loc[clinical_df[AD_DIAGNOSIS_COL] == AD_LABEL]
        .groupby("_join_id")[CLINICAL_DAY_COL].min()
    )
    subj_first_ad_day = df["_join_id"].map(first_ad_day)

    # --- Forward-fill the AD label to later unmatched scans only ---
    ad_forward_mask = (
        (~ad_direct_mask)
        & subj_first_ad_day.notna()
        & (df["_mri_day"] >= subj_first_ad_day)
    )

    ad_mask = ad_direct_mask | ad_forward_mask
    return control_mask, ad_mask, ad_direct_mask, ad_forward_mask


def report_ad_longitudinal_counts(df, window_days, ad_mask, ad_direct_mask, ad_forward_mask):
    """Print patient and scan counts for one choice of matching window.

    "Patients with 2+ visits" is the number that matters: a single scan carries no
    change over time for the model to fit. A window whose labels come mostly from
    forward-fill rather than real nearby visits is weaker evidence, so that case is
    flagged instead of being reported as a flat total.
    """
    # --- Count AD scans, distinct patients, and patients with repeat visits ---
    ad_subjects = df.loc[ad_mask, "_join_id"]
    n_ad_rows = ad_mask.sum()
    n_ad_patients = ad_subjects.nunique()
    visits_per_patient = ad_subjects.value_counts()
    n_multi_visit = (visits_per_patient >= 2).sum()

    print(f"\n  --- window = +/-{window_days} days ---")
    print(f"    AD-labeled scans: {n_ad_rows}  (direct match: {ad_direct_mask.sum()}, "
          f"forward-filled: {ad_forward_mask.sum()})")
    print(f"    AD patients (>=1 visit): {n_ad_patients}")
    print(f"    AD patients (>=2 visits): {n_multi_visit}")

    # --- Flag windows whose labels are mostly forward-filled ---
    if n_ad_rows > 0:
        fwd_share = 100.0 * ad_forward_mask.sum() / n_ad_rows
        if fwd_share > 50.0:
            print(f"    [FLAG] {fwd_share:.0f}% of AD-labeled scans came from forward-fill, "
                  f"not a direct in-window UDS match — treat this window's count with caution.")


def compute_wscores(df, col_map, control_mask):
    """Convert every raw measurement to a w-score against the healthy controls.

    w = (value - control mean) / control SD, computed per region so regions with
    different natural scales become comparable.

    Sign convention: both cortical thickness and subcortical volume DECREASE with
    atrophy, so more atrophy is a more NEGATIVE w-score. This function returns those
    raw, unflipped scores; the flip to the model's "larger = sicker" convention
    happens downstream in atrophy_score_normalize. Never feed this output into the
    model directly.

    Two guard rails warn rather than fail silently: fewer than 5 controls for a
    region leaves the mean and SD too unstable to trust, and a near-zero control SD
    would make the division explode, so zeros are filled and the region is flagged.
    """
    controls = df[control_mask]
    result = pd.DataFrame(index=df.index, columns=REGION_ORDER, dtype=np.float64)

    for target_region, source_col in col_map.items():
        # --- Take this region's control distribution ---
        ctrl_vals = controls[source_col].dropna().values.astype(np.float64)
        if len(ctrl_vals) < 5:
            print(f"  WARNING: only {len(ctrl_vals)} controls for {target_region} — w-score unreliable")
        ctrl_mean = ctrl_vals.mean()
        ctrl_std = ctrl_vals.std()

        # --- Fill zeros where the control SD is degenerate ---
        if ctrl_std < 1e-6:
            print(f"  WARNING: near-zero std for {target_region} — w-score undefined, filling 0")
            result[target_region] = 0.0
        # --- Compute w-scores against controls ---
        else:
            result[target_region] = (df[source_col].values.astype(np.float64) - ctrl_mean) / ctrl_std

    return result


Z_CAP = 3.0


def atrophy_score_normalize(z):
    """Convert raw w-scores to the model's disease burden x = clip(-z, 0, Z_CAP) / Z_CAP.

    x = 0 means the region is at or above the healthy-control average; x = 1 means
    it has atrophied by Z_CAP standard deviations or more. pipeline.py imports this
    function and Z_CAP rather than keeping its own copy, so one definition governs
    the whole project.

    This replaced global min-max normalization, rejected on two measured grounds.
    Min-max kept the raw sign, sending more-diseased regions toward x = 0, backwards
    for a growth term needing x = 1 to mean fully diseased. And it put zero in the
    wrong place: on a raw range of about -6.73 to +5.52 a healthy region (w = 0)
    landed at x = 0.55, compressing ~99% of the data into x in [0.20, 0.76] where
    x*(1-x) sits at 0.24 +/- 0.02, pinned near its 0.25 maximum. It also injected
    drift: L_norm applied to a constant is zero only on a perfectly regular graph,
    and measured on this connectome the worst case reached 0.685, so a constant 0.55
    offset contributed up to 0.377 of disease-irrelevant drift.

    Z_CAP = 3.0 is the standard 3-SD outlier convention, fixed on that basis BEFORE
    any model parameters were fitted and deliberately not tuned against results;
    tuning it would make the burden scale a free parameter chosen to flatter our own
    numbers. Unlike min-max this is a fixed transform using no statistic of the data
    it transforms, so it is identical across train, validation, test and any future
    cohort, and carries no data-leakage concern.

    The trailing "+ 0.0" only normalises IEEE negative zero, so an input of exactly
    w = 0 prints as 0.0000 rather than -0.0000 in the diagnostics.
    """
    # --- Flip the sign, clip to [0, Z_CAP], and rescale to [0, 1] ---
    return np.clip(-z, 0.0, Z_CAP) / Z_CAP + 0.0


def report_atrophy_score_transform(z, label="", indent="  "):
    """Apply the transform and print its distribution, including clipping counts.

    Clipping beyond 3 SD is expected, but how much data hits each limit is something
    a reader needs to judge: a large share pinned at x = 1 would mean the scale is
    compressing away real differences.
    """
    # --- Apply the transform and count values hitting each limit ---
    z_arr = np.asarray(z, dtype=np.float64)
    x = atrophy_score_normalize(z_arr)
    n_tot = x.size
    n_lo = int((-z_arr <= 0.0).sum())
    n_hi = int((-z_arr >= Z_CAP).sum())

    # --- Print the before and after distributions ---
    tag = f" [{label}]" if label else ""
    print(f"{indent}[ATROPHY-SCORE TRANSFORM]{tag} x = clip(-w, 0, {Z_CAP}) / {Z_CAP}")
    print(f"{indent}  raw w-score range: [{z_arr.min():.4f}, {z_arr.max():.4f}]")
    print(f"{indent}  x: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, "
          f"median={np.median(x):.4f}")
    print(f"{indent}  clipped at x=0 (w >= 0, no atrophy vs controls): "
          f"{n_lo} / {n_tot} ({100.0 * n_lo / n_tot:.2f}%)")
    print(f"{indent}  clipped at x=1 (w <= -{Z_CAP}, atrophy beyond {Z_CAP} SD): "
          f"{n_hi} / {n_tot} ({100.0 * n_hi / n_tot:.2f}%)")
    return x


def main():
    """Run the conversion: raw OASIS files in, one model-ready CSV out.

    Every stage that can go wrong prints [STOP] and returns early rather than
    continuing on bad data. The region-order check runs first because nothing after
    it is valid if the orders disagree.
    """
    print("=" * 60)
    print("prep_oasis.py — OASIS-3 FreeSurfer preprocessor")
    print("=" * 60)

    # --- Check our region order matches the project's canonical one ---
    with open(CONFIG_DIR / "ground_truth_params.json") as f:
        gt = json.load(f)
    assert gt["region_order"] == REGION_ORDER, \
        "REGION_ORDER in this script does not match ground_truth_params.json — STOP"
    print(f"[OK] Region order matches ground_truth_params.json (82 regions)")

    # --- Load the raw FreeSurfer measurements ---
    print(f"\nLoading OASIS3_Freesurfer_output.csv ...")
    df = pd.read_csv(RAW_OASIS_DIR / "OASIS3_Freesurfer_output.csv")
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns (first 10): {list(df.columns[:10])}")
    print(f"  Total columns: {len(df.columns)}")

    # --- Map our 82 region names onto raw column names ---
    print(f"\nBuilding column map ...")
    col_map, unmapped = build_column_map(df.columns)
    print(f"  Mapped: {len(col_map)} / 82 regions")

    # --- Stop on any unmapped region, printing what was tried ---
    if unmapped:
        print(f"\n  [STOP] Could not map {len(unmapped)} regions:")
        for region, candidates in unmapped[:10]:
            print(f"    {region} — tried: {candidates[:3]}")
        print(f"\n  Fix: open this script, adjust CORTICAL_THICKNESS_PREFIX_LH,")
        print(f"  CORTICAL_THICKNESS_SUFFIX, SUBCORTICAL_PREFIX_L etc. to match")
        print(f"  the actual column names printed above, then re-run.")
        return
    print(f"  [OK] All 82 regions mapped")

    # --- Load the clinical diagnosis file ---
    print(f"\n  '{DIAGNOSIS_COL}' is not in the FreeSurfer file — this is expected,")
    print(f"  OASIS3_Freesurfer_output.csv is structural-only.")
    print(f"  Loading clinical data from '{CLINICAL_CSV_PATH}' ...")

    if not Path(CLINICAL_CSV_PATH).exists():
        print(f"\n  [STOP] Clinical CSV not found: '{CLINICAL_CSV_PATH}'")
        return

    clinical_df = pd.read_csv(CLINICAL_CSV_PATH)
    print(f"  Loaded clinical CSV: {clinical_df.shape}")

    # --- Confirm the four clinical columns we depend on exist ---
    for col in [CLINICAL_SUBJECT_COL, CLINICAL_DAY_COL, DIAGNOSIS_COL, AD_DIAGNOSIS_COL]:
        if col not in clinical_df.columns:
            print(f"\n  [STOP] '{col}' not found in clinical CSV.")
            print(f"  Clinical CSV columns: {clinical_df.columns.tolist()}")
            return

    # --- Build a shared join key from the subject ID on both sides ---
    df["_join_id"] = df[SUBJECT_COL].astype(str).str.extract(r"(OAS3\d+)")[0].fillna(df[SUBJECT_COL].astype(str))
    clinical_df["_join_id"] = clinical_df[CLINICAL_SUBJECT_COL].astype(str).str.extract(r"(OAS3\d+)")[0].fillna(clinical_df[CLINICAL_SUBJECT_COL].astype(str))

    # --- Extract days since study entry from the session label ---
    df["_mri_day"] = df[SESSION_COL].str.extract(r"_d(\d+)")[0].astype(float)

    # --- Force the clinical day column numeric, coercing bad entries to NaN ---
    clinical_df[CLINICAL_DAY_COL] = pd.to_numeric(clinical_df[CLINICAL_DAY_COL], errors="coerce").astype("float64")

    # --- Report scans with no extractable date ---
    if df["_mri_day"].isna().any():
        n_bad = df["_mri_day"].isna().sum()
        print(f"  WARNING: {n_bad} rows could not extract a day count from "
              f"'{SESSION_COL}' — these rows cannot be diagnosis-matched.")

    # --- Report counts at both comparison windows before committing to one ---
    print(f"\n  Matching each MRI scan to the closest UDS assessment for that")
    print(f"  subject (proximity-window join), with AD labels forward-filled")
    print(f"  only to LATER scans once a subject has a confirmed PROBAD=1 visit:")
    for w in WINDOW_DAYS_COMPARISON:
        _, ad_mask_w, ad_direct_w, ad_forward_w = match_diagnosis_to_scans(df, clinical_df, w)
        report_ad_longitudinal_counts(df, w, ad_mask_w, ad_direct_w, ad_forward_w)

    # --- Join diagnosis to scans at the committed window ---
    print(f"\n  Using window = +/-{WINDOW_DAYS} days for the actual output "
          f"(change WINDOW_DAYS at the top of this script to compare).")
    control_mask, ad_mask, ad_direct_mask, ad_forward_mask = match_diagnosis_to_scans(
        df, clinical_df, WINDOW_DAYS
    )

    # --- Stop if nothing matched, printing both ID formats ---
    matched = control_mask.sum() + ad_mask.sum()
    if matched == 0:
        print(f"\n  [STOP] Zero rows matched a diagnosis — inspect actual subject ID formats:")
        print(f"  FreeSurfer file sample: {df[SUBJECT_COL].head(3).tolist()}")
        print(f"  Clinical file sample:   {clinical_df[CLINICAL_SUBJECT_COL].head(3).tolist()}")
        return

    print(f"\n  Controls ({CONTROL_LABEL}): {control_mask.sum()} rows")
    print(f"  AD ({AD_LABEL}): {ad_mask.sum()} rows "
          f"(direct match: {ad_direct_mask.sum()}, forward-filled: {ad_forward_mask.sum()})")

    # --- Stop on too few controls or too few AD scans ---
    if control_mask.sum() < 10:
        print(f"  [STOP] Too few controls to compute reliable w-scores.")
        print(f"  Check CONTROL_LABEL and DIAGNOSIS_COL configuration.")
        return
    if ad_mask.sum() < 4:
        print(f"  [STOP] Too few AD rows — check AD_LABEL configuration.")
        return

    # --- Compute w-scores against controls ---
    print(f"\nComputing w-scores (against {control_mask.sum()} controls) ...")
    wscores = compute_wscores(df, col_map, control_mask)

    # --- Count missing w-scores from failed FreeSurfer measurements ---
    nan_count = wscores.isnull().sum().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN w-scores found — subjects with missing FreeSurfer values")
        print(f"  These rows will be dropped in the next step.")

    print(f"\nBuilding longitudinal AD output ...")

    # --- Confirm the subject and session columns exist ---
    if SUBJECT_COL not in df.columns:
        print(f"  [STOP] Subject column '{SUBJECT_COL}' not found.")
        print(f"  Available columns: {[c for c in df.columns if 'oasis' in c.lower() or 'subject' in c.lower() or 'session' in c.lower()]}")
        return
    if SESSION_COL not in df.columns:
        print(f"  [STOP] Session column '{SESSION_COL}' not found.")
        print(f"  Available columns containing 'session': "
              f"{[c for c in df.columns if 'session' in c.lower()]}")
        return

    # --- Keep only the AD scans; controls have served as the w-score reference ---
    ad_df = df[ad_mask].copy()
    ad_wscores = wscores[ad_mask].copy()

    # --- Convert the day count to years and extract patient IDs ---
    ad_df["time_years"] = ad_df["_mri_day"] / 365.25
    ad_df["patient_id"] = ad_df[SUBJECT_COL].astype(str).str.extract(r"(OAS3\d+)")[0].fillna(ad_df[SUBJECT_COL])

    # --- Carry the match route through for the diagnostics ---
    ad_df["_match_type"] = np.where(ad_direct_mask.loc[ad_df.index], "direct", "forward_fill")

    # --- Assemble the output columns in the order pipeline.py expects ---
    out_df = pd.DataFrame()
    out_df["patient_id"] = ad_df["patient_id"].values
    out_df["time_years"] = ad_df["time_years"].values
    out_df["_match_type"] = ad_df["_match_type"].values
    for region in REGION_ORDER:
        out_df[region] = ad_wscores[region].values

    # --- Drop any scan missing a w-score in any region ---
    before = len(out_df)
    out_df = out_df.dropna()
    after = len(out_df)
    if before != after:
        print(f"  Dropped {before - after} rows with missing w-scores")

    # --- Sort by subject, then chronologically within subject ---
    out_df = out_df.sort_values(["patient_id", "time_years"]).reset_index(drop=True)

    # --- Report patient counts, timepoints and value ranges ---
    n_patients = out_df["patient_id"].nunique()
    n_timepoints = out_df.groupby("patient_id").size()
    n_multi_visit = (n_timepoints >= 2).sum()
    n_direct = (out_df["_match_type"] == "direct").sum()
    n_forward = (out_df["_match_type"] == "forward_fill").sum()
    print(f"\n[DIAGNOSTICS] (window = +/-{WINDOW_DAYS} days, after dropping NaN w-score rows)")
    print(f"  AD patients with at least one visit: {n_patients}")
    print(f"  AD patients with 2+ visits: {n_multi_visit}")
    print(f"  AD-labeled scans: direct match = {n_direct}, forward-filled = {n_forward}")
    print(f"  Timepoints per patient: min={n_timepoints.min()}, max={n_timepoints.max()}, mean={n_timepoints.mean():.1f}")
    print(f"  Time range (years): {out_df['time_years'].min():.1f} — {out_df['time_years'].max():.1f}")
    print(f"  W-score range: {out_df[REGION_ORDER].values.min():.3f} — {out_df[REGION_ORDER].values.max():.3f}")

    # --- Check most region-timepoints are negative, as an AD cohort should be ---
    n_neg = int((out_df[REGION_ORDER].values < 0).sum())
    n_all = out_df[REGION_ORDER].values.size
    print(f"  Region-timepoints with negative w-score (= MORE atrophy than controls): "
          f"{n_neg} / {n_all} ({100.0 * n_neg / n_all:.1f}%)")

    # --- Preview the burden scale pipeline.py will apply on load ---
    print(f"\n[BURDEN SCALE PREVIEW] pipeline.py --data-source real will apply:")
    report_atrophy_score_transform(out_df[REGION_ORDER].values, label="AD output rows")

    # --- Drop the provenance column before saving ---
    out_df = out_df.drop(columns=["_match_type"])

    # --- Save and confirm the column order survived ---
    out_path = REAL_DATA_DIR / "Ad_atrophy_real.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path} — {out_df.shape[0]} rows x {out_df.shape[1]} columns")
    print(f"  Column order matches REGION_ORDER: "
          f"{list(out_df.columns[2:]) == REGION_ORDER}")
    print(f"\nNext step: run pipeline.py --data-source real (loads this file + "
          f"data/real/connectome_82region.csv; irregular timepoints already supported).")


if __name__ == "__main__":
    main()
