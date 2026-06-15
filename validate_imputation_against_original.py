#!/usr/bin/env python3
# =============================================================================
# PRIVATE VALIDATION ONLY - DO NOT SUBMIT OR USE FOR MODEL TRAINING
# =============================================================================
"""
validate_imputation_against_original.py

Compares imputed values from the preprocessing pipeline against the original
public dataset (online_shoppers_intention.csv) as a private diagnostic tool.

THIS FILE IS NOT FOR SUBMISSION.
Outputs are written to private_validation/ and are excluded from git.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import io
import textwrap

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PRIVATE_DIR = Path("private_validation")
PRIVATE_DIR.mkdir(exist_ok=True)

BANNER = (
    "\n" + "=" * 70 + "\n"
    " PRIVATE VALIDATION ONLY - DO NOT SUBMIT OR USE FOR MODEL TRAINING \n"
    + "=" * 70
)

# Column mapping: assignment names → original UCI names
FEATURE_MAP = {
    "num_admin_pages"  : "Administrative",
    "admin_duration"   : "Administrative_Duration",
    "num_info_pages"   : "Informational",
    "info_duration"    : "Informational_Duration",
    "num_product_pages": "ProductRelated",
    "product_duration" : "ProductRelated_Duration",
    "bounce_rate"      : "BounceRates",
    "exit_rate"        : "ExitRates",
    "page_value"       : "PageValues",
    "special_day_score": "SpecialDay",
    "month"            : "Month",
    "operating_system" : "OperatingSystems",
    "browser"          : "Browser",
    "region"           : "Region",
    "traffic_type"     : "TrafficType",
    "visitor_type"     : "VisitorType",
    "is_weekend"       : "Weekend",
    "high_intent"      : "Revenue",
}
REVERSE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# Features that were imputed in the pipeline
IMPUTED_FEATURES = ["month", "visitor_type", "product_duration", "page_value"]

# Primary matching columns (no missing values in train, present in both datasets)
MATCH_BASE = [
    "num_admin_pages", "admin_duration",
    "num_info_pages",  "info_duration",
    "num_product_pages",
    "exit_rate", "special_day_score",
    "operating_system", "browser", "region", "traffic_type",
    "is_weekend",
]
FLOAT_MATCH = ["admin_duration", "info_duration", "exit_rate", "special_day_score"]
FLOAT_ROUND  = 6   # decimal places for float matching keys

# ── Imputation pipeline (mirrors notebook Section 7) ─────────────────────────

def run_imputation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run the notebook imputation pipeline on raw data to obtain the
    imputed-but-not-yet-transformed dataset.
    NOTE: high_intent is NOT used anywhere here (no leakage).
    """
    df = df_raw.copy()

    # 7.1  Drop bounce_rate (0.98 corr with abandonment_score, 11% missing)
    if "bounce_rate" in df.columns:
        df = df.drop(columns=["bounce_rate"])

    # 7.2  Clip engagement_score at 95th pct
    clip_thresh = df["engagement_score"].quantile(0.95)
    df["engagement_score"] = df["engagement_score"].clip(upper=clip_thresh)

    # 7.3a  product_duration: zero-fill where num_product_pages == 0
    cond = (df["num_product_pages"] == 0) & df["product_duration"].isna()
    df.loc[cond, "product_duration"] = 0.0

    # 7.3b  month: P(month | special_day_group) from observed rows
    df["_sdg"] = np.where(df["special_day_score"] > 0, "special_day", "regular_day")
    dists = {}
    for g in ["regular_day", "special_day"]:
        d = (df[df["month"].notna() & (df["_sdg"] == g)]["month"]
             .value_counts(normalize=True))
        dists[g] = d
    np.random.seed(RANDOM_SEED)
    for idx in df[df["month"].isna()].index:
        d = dists[df.loc[idx, "_sdg"]]
        df.loc[idx, "month"] = np.random.choice(d.index, p=d.values)
    df.drop(columns=["_sdg"], inplace=True)

    # 7.3c  visitor_type: KNN classifier
    knn_feats = [
        "num_product_pages", "exit_rate", "admin_duration", "info_duration",
        "traffic_type", "special_day_score", "is_weekend",
        "operating_system", "browser", "region",
        "engagement_score", "abandonment_score",
    ]
    tr = df[df["visitor_type"].notna()].dropna(subset=knn_feats)
    pr = df[df["visitor_type"].isna()].dropna(subset=knn_feats)
    if len(pr) > 0:
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(tr[knn_feats].values)
        X_pr_sc = sc.transform(pr[knn_feats].values)
        y_tr    = tr["visitor_type"].values
        best_k, best_acc = 1, 0.0
        for k in range(1, 21):
            s = cross_val_score(
                KNeighborsClassifier(n_neighbors=k, weights="distance"),
                X_tr_sc, y_tr, cv=5, scoring="accuracy",
            )
            if s.mean() > best_acc:
                best_acc, best_k = s.mean(), k
        knn_m = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
        knn_m.fit(X_tr_sc, y_tr)
        df.loc[pr.index, "visitor_type"] = knn_m.predict(X_pr_sc)
    remain_vt = df["visitor_type"].isna()
    if remain_vt.sum() > 0:
        df.loc[remain_vt, "visitor_type"] = df["visitor_type"].mode()[0]

    # 7.3d  product_duration: linear regression on engagement_score
    tr_pd = df[df["product_duration"].notna()]
    pr_pd = df[df["product_duration"].isna()]
    if len(pr_pd) > 0:
        lr = LinearRegression()
        lr.fit(tr_pd[["engagement_score"]], tr_pd["product_duration"])
        df.loc[pr_pd.index, "product_duration"] = np.clip(
            lr.predict(pr_pd[["engagement_score"]]), 0, None
        )

    # 7.3e  page_value: Random Forest — NO high_intent (leakage prevention)
    rf_feats = [
        "product_duration", "engagement_score", "admin_duration", "info_duration",
        "exit_rate", "abandonment_score", "special_day_score", "is_weekend",
        "num_product_pages",
    ]
    tr_pv = df[df["page_value"].notna()]
    pr_pv = df[df["page_value"].isna()]
    if len(pr_pv) > 0:
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        rf.fit(tr_pv[rf_feats], np.log1p(tr_pv["page_value"]))
        df.loc[pr_pv.index, "page_value"] = np.clip(
            np.expm1(rf.predict(pr_pv[rf_feats])), 0, None
        )

    return df


# ── Row matching ──────────────────────────────────────────────────────────────

def _build_key(df: pd.DataFrame, cols: list, extra_float: str | None = None) -> pd.Series:
    """Build a string composite key from `cols`, rounding floats."""
    parts = []
    for c in cols:
        s = df[c].copy()
        if s.dtype in [float, np.float64, np.float32]:
            s = s.round(FLOAT_ROUND)
        parts.append(s.astype(str))
    if extra_float is not None and extra_float in df.columns:
        ef = df[extra_float].round(FLOAT_ROUND).astype(str)
        parts.append(ef)
    return parts[0].str.cat(parts[1:], sep="|")


def match_to_original(
    df_assign: pd.DataFrame,
    df_orig_aligned: pd.DataFrame,
    imputed_feats: list,
) -> pd.DataFrame:
    """
    Match each assignment row to its original-dataset counterpart.

    Strategy:
      Tier 1 — rows where bounce_rate is known:  match on MATCH_BASE + bounce_rate
      Tier 2 — rows where bounce_rate is NaN:     match on MATCH_BASE only

    Returns a DataFrame with columns:
      assign_idx, orig_idx, <imputed_feats from assignment>, <imputed_feats from original>
      match_tier, ambiguous (bool)
    """
    assign = df_assign.copy().reset_index(drop=False).rename(columns={"index": "assign_idx"})
    orig   = df_orig_aligned.copy().reset_index(drop=False).rename(columns={"index": "orig_idx"})

    # Round floats in both
    for c in FLOAT_MATCH + ["bounce_rate"]:
        for frame in [assign, orig]:
            if c in frame.columns:
                frame[c] = frame[c].round(FLOAT_ROUND)

    records = []

    # Tier 1: bounce_rate is not NaN
    tier1_assign = assign[assign["bounce_rate"].notna()].copy()
    tier1_orig   = orig.copy()
    key1_cols    = MATCH_BASE + ["bounce_rate"]
    tier1_assign["_key"] = _build_key(tier1_assign, key1_cols)
    tier1_orig["_key"]   = _build_key(tier1_orig,   key1_cols)

    merged1 = tier1_assign[["assign_idx", "_key"]].merge(
        tier1_orig[["orig_idx", "_key"] + imputed_feats],
        on="_key", how="left",
    )
    merged1["match_tier"] = 1

    # Tier 2: bounce_rate is NaN — match without it
    tier2_assign = assign[assign["bounce_rate"].isna()].copy()
    tier2_assign["_key"] = _build_key(tier2_assign, MATCH_BASE)
    tier2_orig   = orig.copy()
    tier2_orig["_key"]   = _build_key(tier2_orig,   MATCH_BASE)

    merged2 = tier2_assign[["assign_idx", "_key"]].merge(
        tier2_orig[["orig_idx", "_key"] + imputed_feats],
        on="_key", how="left",
    )
    merged2["match_tier"] = 2

    combined = pd.concat([merged1, merged2], ignore_index=True)

    # Count how many original rows match each assignment row
    n_matches = combined.groupby("assign_idx")["orig_idx"].transform("count")
    combined["n_orig_matches"] = n_matches
    combined["ambiguous"] = n_matches > 1

    # For ambiguous rows: keep only if all matches agree on imputed feature values
    def _dedup(group):
        if len(group) == 1:
            return group
        for feat in imputed_feats:
            if group[feat].nunique(dropna=False) > 1:
                # Disagreement — drop all
                return pd.DataFrame()
        # Agreement — keep first
        return group.head(1)

    combined = combined.groupby("assign_idx", group_keys=False).apply(_dedup)
    combined = combined.reset_index(drop=True)

    return combined


# ── Metric computation ────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _pearson(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(y_true, y_pred)
    return float(r), float(p)


def _spearman(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan, np.nan
    r, p = stats.spearmanr(y_true, y_pred)
    return float(r), float(p)


def compare_numeric(
    imputed: pd.Series,
    original: pd.Series,
    feature_name: str,
) -> dict:
    valid = (imputed.notna() & original.notna())
    imp = imputed[valid].values.astype(float)
    ori = original[valid].values.astype(float)
    n   = len(imp)
    if n == 0:
        return {"n": 0, "feature": feature_name, "type": "numeric"}

    mae   = float(mean_absolute_error(ori, imp))
    rmse  = _rmse(ori, imp)
    medae = float(np.median(np.abs(ori - imp)))
    mape  = _mape(pd.Series(ori), pd.Series(imp))
    pr, pp = _pearson(ori, imp)
    sr, sp = _spearman(ori, imp)
    exact  = float((np.abs(ori - imp) < 1e-6).mean() * 100)

    return {
        "feature"        : feature_name,
        "type"           : "numeric",
        "n"              : n,
        "mae"            : round(mae, 4),
        "rmse"           : round(rmse, 4),
        "median_ae"      : round(medae, 4),
        "mape_%"         : round(mape, 2) if not np.isnan(mape) else None,
        "pearson_r"      : round(pr, 4),
        "pearson_p"      : round(pp, 6),
        "spearman_r"     : round(sr, 4),
        "spearman_p"     : round(sp, 6),
        "exact_match_%"  : round(exact, 2),
    }


def compare_categorical(
    imputed: pd.Series,
    original: pd.Series,
    feature_name: str,
) -> dict:
    valid = (imputed.notna() & original.notna())
    imp = imputed[valid].astype(str)
    ori = original[valid].astype(str)
    n   = len(imp)
    if n == 0:
        return {"n": 0, "feature": feature_name, "type": "categorical"}

    acc = float(accuracy_score(ori, imp))
    classes = sorted(ori.unique())
    per_class = {}
    for cls in classes:
        mask = ori == cls
        if mask.sum() > 0:
            per_class[cls] = round(float((imp[mask] == ori[mask]).mean()), 4)

    cm = pd.crosstab(ori, imp, rownames=["original"], colnames=["imputed"])

    # Most common errors
    errors = pd.DataFrame({"original": ori, "imputed": imp})
    errors = errors[errors["original"] != errors["imputed"]]
    common_errors = (
        errors.groupby(["original", "imputed"])
        .size()
        .sort_values(ascending=False)
        .head(5)
        .reset_index(name="count")
        .to_dict("records")
    )

    return {
        "feature"        : feature_name,
        "type"           : "categorical",
        "n"              : n,
        "accuracy"       : round(acc, 4),
        "per_class_acc"  : per_class,
        "confusion_matrix": cm,
        "top_errors"     : common_errors,
    }


# ── Quality label ─────────────────────────────────────────────────────────────

def quality_label(result: dict) -> tuple[str, str]:
    """Return (label, interpretation) based on the main metric."""
    if result["type"] == "numeric":
        r = result.get("pearson_r", 0) or 0
        if   r >= 0.85: return "excellent", "Imputed values closely follow true distribution (r ≥ 0.85)"
        elif r >= 0.65: return "good",      "Reasonable linear alignment with ground truth (r ≥ 0.65)"
        elif r >= 0.40: return "moderate",  "Weak correlation; distribution shape partially preserved"
        else:           return "poor",      "Low correlation with ground truth; consider a better model"
    else:
        acc = result.get("accuracy", 0) or 0
        if   acc >= 0.80: return "excellent", "High accuracy on held-out imputed rows (≥ 80%)"
        elif acc >= 0.65: return "good",      "Reasonable accuracy on imputed rows (≥ 65%)"
        elif acc >= 0.50: return "moderate",  "Accuracy only marginally above chance"
        else:             return "poor",      "Accuracy below 50%; imputation strategy may need revision"


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_numeric(imputed: pd.Series, original: pd.Series, feature: str, out_dir: Path):
    valid = imputed.notna() & original.notna()
    imp = imputed[valid].values.astype(float)
    ori = original[valid].values.astype(float)
    if len(imp) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(ori, imp, alpha=0.4, s=10, color="steelblue")
    lo, hi = min(ori.min(), imp.min()), max(ori.max(), imp.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect")
    axes[0].set_xlabel("Original (ground truth)")
    axes[0].set_ylabel("Imputed")
    axes[0].set_title(f"{feature}: Imputed vs Original")
    axes[0].legend()

    axes[1].hist(ori, bins=40, alpha=0.6, label="Original", color="steelblue")
    axes[1].hist(imp, bins=40, alpha=0.6, label="Imputed",  color="salmon")
    axes[1].set_title("Distribution Comparison")
    axes[1].legend()

    residuals = imp - ori
    axes[2].hist(residuals, bins=40, color="green", alpha=0.7)
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_title("Residuals (imputed − original)")
    axes[2].set_xlabel("Error")

    plt.suptitle(f"Imputation Validation: {feature}", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / f"plot_{feature}.png", dpi=100)
    plt.close(fig)


def plot_categorical(result: dict, out_dir: Path):
    feature = result["feature"]
    cm = result.get("confusion_matrix")
    if cm is None or cm.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix: {feature}\n(rows=original, cols=imputed)")
    plt.tight_layout()
    fig.savefig(out_dir / f"plot_{feature}_cm.png", dpi=100)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(BANNER)
    print()

    # ── 1. Load raw datasets ──────────────────────────────────────────────────
    print("1. Loading datasets...")
    df_train = pd.read_csv("shopper_train.csv")
    df_test  = pd.read_csv("shopper_test.csv")
    df_orig  = pd.read_csv("online_shoppers_intention.csv")

    print(f"   shopper_train.csv             : {df_train.shape}")
    print(f"   shopper_test.csv              : {df_test.shape}")
    print(f"   online_shoppers_intention.csv : {df_orig.shape}")
    total_assign = len(df_train) + len(df_test)
    print(f"   train + test = {total_assign}  |  original = {len(df_orig)}")

    # ── 2. Scan for existing imputed/output files ─────────────────────────────
    print("\n2. Scanning for existing imputed output files...")
    csv_candidates = {
        p.name: p for p in sorted(Path(".").glob("*.csv"))
        if p.name not in ("shopper_train.csv", "shopper_test.csv",
                          "online_shoppers_intention.csv")
    }
    if csv_candidates:
        for fname, fpath in csv_candidates.items():
            size_kb = fpath.stat().st_size / 1024
            print(f"   Found: {fname}  ({size_kb:.0f} KB)")
    else:
        print("   No additional CSV files found.")

    # Decide imputed data source
    df_imputed_train = None
    imputed_source   = ""

    if Path("df_filled.csv").exists():
        try:
            df_imputed_train = pd.read_csv("df_filled.csv")
            # Verify it has the expected imputed columns
            if all(c in df_imputed_train.columns for c in IMPUTED_FEATURES):
                imputed_source = "df_filled.csv  (old notebook intermediate file)"
            else:
                df_imputed_train = None
        except Exception:
            df_imputed_train = None

    if df_imputed_train is None:
        print("\n   No suitable intermediate imputed file found.")
        print("   Re-running imputation pipeline from shopper_train.csv ...")
        print("   (This may take ~1 minute for the KNN and RandomForest steps)")
        df_imputed_train = run_imputation(df_train)
        imputed_source = "Re-run of notebook imputation pipeline (same logic, same seed)"

    print(f"\n   Imputed dataset source : {imputed_source}")
    print(f"   Shape                  : {df_imputed_train.shape}")

    # ── 3. Record which rows were originally missing ───────────────────────────
    missing_masks_train = {
        feat: df_train[feat].isna() for feat in IMPUTED_FEATURES
    }
    for feat, mask in missing_masks_train.items():
        print(f"   Originally missing in train — {feat}: {mask.sum()} rows "
              f"({mask.mean()*100:.1f}%)")

    # ── 4. Align original dataset ─────────────────────────────────────────────
    print("\n3. Aligning original dataset column names...")
    orig_aligned = pd.DataFrame(index=df_orig.index)
    for orig_col, assign_col in REVERSE_MAP.items():
        if orig_col in df_orig.columns:
            orig_aligned[assign_col] = df_orig[orig_col]

    # Convert bool → int
    for col in ["is_weekend", "high_intent"]:
        if col in orig_aligned.columns:
            orig_aligned[col] = orig_aligned[col].astype(int)

    # Add bounce_rate for tier-1 matching
    if "BounceRates" in df_orig.columns:
        orig_aligned["bounce_rate"] = df_orig["BounceRates"]

    print(f"   Aligned columns: {orig_aligned.columns.tolist()}")

    # Prepare assignment train copy for matching (add bounce_rate column)
    train_for_match = df_train.copy()

    # ── 5. Match rows ─────────────────────────────────────────────────────────
    print("\n4. Matching assignment train rows to original dataset...")

    orig_for_match = orig_aligned.copy()

    matched_df = match_to_original(
        train_for_match,
        orig_for_match,
        IMPUTED_FEATURES,
    )

    n_total     = len(df_train)
    n_matched   = matched_df["assign_idx"].nunique()
    n_ambiguous = matched_df[matched_df["ambiguous"]]["assign_idx"].nunique()
    n_unmatched = n_total - n_matched

    print(f"   Total assignment rows : {n_total}")
    print(f"   Successfully matched  : {n_matched}  ({100*n_matched/n_total:.1f}%)")
    print(f"   Ambiguous (agreed)    : {n_ambiguous}")
    print(f"   Unmatched             : {n_unmatched}")

    if n_matched < n_total * 0.5:
        print("   WARNING: Match rate below 50%. Comparison results may not be reliable.")

    # Attach imputed values from df_imputed_train
    imputed_vals = df_imputed_train[IMPUTED_FEATURES].copy()
    imputed_vals.index.name = "assign_idx"
    imputed_vals = imputed_vals.reset_index()

    matched_full = matched_df.merge(
        imputed_vals.rename(columns={f: f"{f}_imputed" for f in IMPUTED_FEATURES}),
        on="assign_idx",
        how="left",
    )

    # Rename original feature columns
    matched_full = matched_full.rename(
        columns={f: f"{f}_original" for f in IMPUTED_FEATURES}
    )

    # ── 6. Compare imputed features ───────────────────────────────────────────
    print("\n5. Comparing imputed features against ground truth...")

    results = {}
    summary_rows = []

    for feat in IMPUTED_FEATURES:
        print(f"\n   -- {feat} --")

        # Filter to rows that were originally missing
        missing_idx = missing_masks_train[feat]
        missing_in_matched = matched_full[
            matched_full["assign_idx"].isin(df_train.index[missing_idx])
        ].copy()

        n_imputed_matched = len(missing_in_matched)
        print(f"   Originally missing rows matched to original: {n_imputed_matched}")

        if n_imputed_matched == 0:
            print("   No matched rows to compare. Skipping.")
            results[feat] = {"feature": feat, "n": 0}
            continue

        col_imp = f"{feat}_imputed"
        col_ori = f"{feat}_original"

        if col_imp not in missing_in_matched.columns or col_ori not in missing_in_matched.columns:
            print(f"   Missing columns {col_imp} or {col_ori}. Skipping.")
            continue

        imp_series = missing_in_matched[col_imp]
        ori_series = missing_in_matched[col_ori]

        # Determine type
        feat_type = "categorical" if feat in ("month", "visitor_type") else "numeric"

        if feat_type == "numeric":
            result = compare_numeric(imp_series, ori_series, feat)
            label, interp = quality_label(result)
            print(f"   n={result['n']}  MAE={result['mae']}  RMSE={result['rmse']}  "
                  f"MedianAE={result['median_ae']}  Pearson r={result['pearson_r']}  "
                  f"Spearman r={result['spearman_r']}")
            print(f"   MAPE={result['mape_%']}%  Exact match={result['exact_match_%']}%")
            print(f"   Quality: {label.upper()} — {interp}")
            plot_numeric(imp_series, ori_series, feat, PRIVATE_DIR)

        else:
            result = compare_categorical(imp_series, ori_series, feat)
            label, interp = quality_label(result)
            print(f"   n={result['n']}  Accuracy={result['accuracy']:.4f}")
            print(f"   Per-class accuracy: {result['per_class_acc']}")
            print(f"   Top errors: {result['top_errors']}")
            print(f"   Quality: {label.upper()} — {interp}")
            plot_categorical(result, PRIVATE_DIR)

        result["quality_label"]   = label
        result["interpretation"]  = interp
        results[feat] = result

        main_metric = (
            result.get("pearson_r")
            if feat_type == "numeric"
            else result.get("accuracy")
        )
        summary_rows.append({
            "feature"          : feat,
            "type"             : feat_type,
            "n_imputed_eval"   : result["n"],
            "main_metric"      : round(float(main_metric), 4) if main_metric is not None else None,
            "metric_name"      : "Pearson r" if feat_type == "numeric" else "Accuracy",
            "quality_label"    : label,
            "interpretation"   : interp,
        })

    # ── 7. Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" IMPUTATION QUALITY SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    print(f"\n6. Saving outputs to {PRIVATE_DIR}/...")

    # CSV report
    csv_path = PRIVATE_DIR / "imputation_validation_report.csv"
    report_rows = []
    for feat, res in results.items():
        row = {"feature": feat}
        if res.get("type") == "numeric":
            for k in ["n", "mae", "rmse", "median_ae", "mape_%", "pearson_r",
                      "pearson_p", "spearman_r", "spearman_p", "exact_match_%",
                      "quality_label", "interpretation"]:
                row[k] = res.get(k)
        elif res.get("type") == "categorical":
            row["n"]             = res.get("n")
            row["accuracy"]      = res.get("accuracy")
            row["per_class_acc"] = str(res.get("per_class_acc", {}))
            row["top_errors"]    = str(res.get("top_errors", []))
            row["quality_label"] = res.get("quality_label")
            row["interpretation"]= res.get("interpretation")
        report_rows.append(row)

    pd.DataFrame(report_rows).to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Text summary
    txt_path = PRIVATE_DIR / "imputation_validation_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(BANNER + "\n\n")
        f.write(f"Imputed dataset source : {imputed_source}\n")
        f.write(f"Matching rate          : {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)\n\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 70 + "\n")
        if not summary_df.empty:
            f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        for feat, res in results.items():
            f.write(f"\n{'─'*60}\n")
            f.write(f"Feature: {feat}\n")
            if res.get("type") == "numeric":
                for k, v in res.items():
                    if k not in ("feature", "type"):
                        f.write(f"  {k}: {v}\n")
            elif res.get("type") == "categorical":
                for k, v in res.items():
                    if k not in ("feature", "type", "confusion_matrix"):
                        f.write(f"  {k}: {v}\n")
                if "confusion_matrix" in res and res["confusion_matrix"] is not None:
                    f.write("\nConfusion matrix:\n")
                    f.write(res["confusion_matrix"].to_string())
                    f.write("\n")
        f.write("\n" + BANNER + "\n")

    print(f"   Saved: {txt_path}")

    plots = list(PRIVATE_DIR.glob("plot_*.png"))
    for p in plots:
        print(f"   Saved: {p}")

    print("\n" + BANNER)
    print()


if __name__ == "__main__":
    main()
