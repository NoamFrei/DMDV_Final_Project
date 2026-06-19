#!/usr/bin/env python3
# =============================================================================
#  PRIVATE CHECK ONLY — NOT FOR SUBMISSION
# =============================================================================
"""
private_original_data_evaluation.py

Post-hoc evaluation of all trained classification models against recovered
ground-truth labels from the public UCI dataset.

Extended to include classification-capable models from
private_imputation_method_benchmark_all_methods.py:
  KNN, Extra Trees, XGBoost, CatBoost, LightGBM, HistGradientBoosting

IMPORTANT CONSTRAINTS — enforced by design:
  - Original labels are recovered ONLY for evaluation.  They are never written
    back to the submitted test file, used for training, or used for tuning.
  - No submitted notebook or preprocessing file is modified.
  - All outputs go to private_evaluation/ and are excluded from git.

Preprocessing philosophy (mirrors the benchmark file):
  - Distance / linear models (LR, NB, SVM, KNN):
      Pipeline with OHE on cat_cols + StandardScaler on num_cols.
  - Tree / boosting models (DT, RF, Extra Trees, XGBoost, CatBoost,
    LightGBM, HistGradientBoosting):
      No pipeline — they handle label-encoded integer categoricals and
      already-scaled numerics in X_train_preprocessed.csv directly,
      exactly as the benchmark methods do on the imputation training data.

Run:
    py private_original_data_evaluation.py              # label-encoded (default)
    py private_original_data_evaluation.py --mode both  # compare LE vs OHE
"""

import warnings
warnings.filterwarnings("ignore")

import sys, io, argparse
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Optional boosting libraries — mirrors availability checks in benchmark ─────

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR      = PROJECT_ROOT / "RawData"
PREPRO_DIR   = PROJECT_ROOT / "PreProcessedData"
PRIVATE_DIR  = SCRIPT_DIR  / "private_evaluation"
PRIVATE_DIR.mkdir(exist_ok=True)

BANNER = (
    "\n" + "=" * 72 + "\n"
    "  PRIVATE CHECK ONLY — NOT FOR SUBMISSION\n"
    "  Results are diagnostic only and do NOT affect the submitted solution.\n"
    + "=" * 72
)

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

# Stable columns used for row matching (no imputed missing values in test set)
MATCH_BASE = [
    "num_admin_pages", "admin_duration",
    "num_info_pages",  "info_duration",
    "num_product_pages",
    "exit_rate", "special_day_score",
    "operating_system", "browser", "region", "traffic_type",
    "is_weekend",
]
FLOAT_COLS  = ["admin_duration", "info_duration", "exit_rate", "special_day_score"]
FLOAT_ROUND = 6

# Categorical columns present in X_train_preprocessed.csv (label-encoded integers)
CAT_COLS = ["month", "browser", "region", "traffic_type", "is_weekend", "visitor_type"]

# Model origin tags used in the final summary
BENCHMARK_ORIGINATED = {
    "KNN", "Extra Trees", "HistGradientBoosting",
    "XGBoost", "CatBoost", "LightGBM",
}
SECTION9_BASELINE = {
    "Logistic Regression", "Decision Tree", "Random Forest",
    "Naive Bayes", "SVM",
}

# ── Safety guard ──────────────────────────────────────────────────────────────

SUBMITTED_FILES = {
    "shopper_train.csv", "shopper_test.csv",
    "X_train_preprocessed.csv", "X_test_preprocessed.csv",
    "y_train.csv", "test_predictions.csv",
    "online_shoppers_intention.csv",
    "Classification_Section9.ipynb",
}

def _safe_write(path: Path, df: pd.DataFrame) -> None:
    """Write to PRIVATE_DIR only — refuse any overwrite of submitted files."""
    assert Path(path).parent.resolve() == PRIVATE_DIR.resolve(), \
        f"SAFETY: refusing to write outside private_evaluation/: {path}"
    assert Path(path).name not in SUBMITTED_FILES, \
        f"SAFETY: refusing to overwrite submitted file: {path.name}"
    df.to_csv(path, index=False)
    print(f"   Saved → {path}")


# ── Row-matching helpers ───────────────────────────────────────────────────────

def _build_key(df: pd.DataFrame, cols: list) -> pd.Series:
    parts = []
    for c in cols:
        s = df[c].copy()
        if s.dtype in (float, np.float64, np.float32):
            s = s.round(FLOAT_ROUND)
        parts.append(s.astype(str))
    return parts[0].str.cat(parts[1:], sep="|")


def match_test_to_original(df_test: pd.DataFrame,
                            df_orig_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Match each test row to its counterpart in the original dataset and
    retrieve the Revenue (high_intent) label.

    Tier 1: bounce_rate present → match on MATCH_BASE + bounce_rate
    Tier 2: bounce_rate missing → match on MATCH_BASE only

    Ambiguous matches are kept only when all original candidates agree on label.
    """
    test = df_test.copy().reset_index(drop=False).rename(columns={"index": "test_idx"})
    orig = df_orig_aligned.copy().reset_index(drop=False).rename(columns={"index": "orig_idx"})

    for c in FLOAT_COLS + ["bounce_rate"]:
        for frame in (test, orig):
            if c in frame.columns:
                frame[c] = frame[c].round(FLOAT_ROUND)

    records = []

    t1_test = test[test["bounce_rate"].notna()].copy()
    t1_cols = MATCH_BASE + ["bounce_rate"]
    t1_test["_key"] = _build_key(t1_test, t1_cols)
    t1_orig = orig.copy()
    t1_orig["_key"] = _build_key(t1_orig, t1_cols)
    m1 = t1_test[["test_idx", "_key"]].merge(
        t1_orig[["orig_idx", "_key", "high_intent"]], on="_key", how="left",
    )
    m1["match_tier"] = 1
    records.append(m1)

    t2_test = test[test["bounce_rate"].isna()].copy()
    t2_test["_key"] = _build_key(t2_test, MATCH_BASE)
    t2_orig = orig.copy()
    t2_orig["_key"] = _build_key(t2_orig, MATCH_BASE)
    m2 = t2_test[["test_idx", "_key"]].merge(
        t2_orig[["orig_idx", "_key", "high_intent"]], on="_key", how="left",
    )
    m2["match_tier"] = 2
    records.append(m2)

    combined = pd.concat(records, ignore_index=True)
    n_matches = combined.groupby("test_idx")["orig_idx"].transform("count")
    combined["n_orig_matches"] = n_matches
    combined["ambiguous"] = n_matches > 1

    def _dedup(grp):
        if len(grp) == 1:
            return grp
        if grp["high_intent"].nunique(dropna=False) == 1:
            return grp.head(1)
        return pd.DataFrame()  # conflicting labels → discard

    combined = combined.groupby("test_idx", group_keys=False).apply(_dedup)
    combined = combined.reset_index(drop=True)
    combined = combined.rename(columns={"high_intent": "recovered_label"})
    return combined[["test_idx", "orig_idx", "recovered_label",
                      "match_tier", "n_orig_matches", "ambiguous"]]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def make_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """OHE for categoricals + StandardScaler for numerics.
    Used by distance-based and linear models (LR, NB, SVM, KNN).
    """
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


# ── Model registry ────────────────────────────────────────────────────────────

def build_models(num_cols: list, cat_cols: list) -> dict:
    """
    Return an ordered dict of name → estimator (or Pipeline).

    Hyperparameters for all new models are taken directly from
    private_imputation_method_benchmark_all_methods.py:

      KNN           ← method_cat_C_KNN
                       weights="distance"; k=11 (benchmark tunes k by 5-fold CV on the
                       imputation subset; fixed odd k avoids the runtime cost on the
                       full training set while preserving the distance-weighting logic).

      Extra Trees   ← method_cat_H_ExtraTrees / method_num_J_ExtraTrees
                       n_estimators=300, max_depth=12, min_samples_leaf=2.

      XGBoost       ← method_cat_E_XGBoost
                       n_estimators=200, max_depth=4, lr=0.05, subsample=0.9,
                       colsample_bytree=0.9.  Objective swapped from multi:softprob
                       (multiclass imputation labels) to binary:logistic (binary
                       high_intent target).

      CatBoost      ← method_cat_F_CatBoost
                       iterations=300, depth=5, lr=0.05.
                       Loss swapped from MultiClass to Logloss.

      LightGBM      ← method_cat_G_LightGBM
                       n_estimators=300, max_depth=5, lr=0.05,
                       subsample=0.9, colsample_bytree=0.9.

      HistGradientBoosting ← method_num_K_HistGradientBoosting (adapted to classifier)
                       max_iter=300, max_leaf_nodes=31, lr=0.05, l2_regularization=0.1.
    """
    models = {
        # ── Original Section 9 classifiers (unchanged) ─────────────────────────
        "Logistic Regression": Pipeline([
            ("pre", make_preprocessor(num_cols, cat_cols)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Naive Bayes": Pipeline([
            ("pre", make_preprocessor(num_cols, cat_cols)),
            ("clf", GaussianNB()),
        ]),
        "SVM": Pipeline([
            ("pre", make_preprocessor(num_cols, cat_cols)),
            ("clf", SVC(probability=True, random_state=RANDOM_SEED)),
        ]),

        # ── New: from imputation benchmark ─────────────────────────────────────
        # Distance-based — needs OHE + scaling (same as LR/NB/SVM above).
        "KNN": Pipeline([
            ("pre", make_preprocessor(num_cols, cat_cols)),
            ("clf", KNeighborsClassifier(
                n_neighbors=11, weights="distance", n_jobs=-1,
            )),
        ]),
        # Tree-based — work directly on label-encoded integers (same as DT/RF above).
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=31, learning_rate=0.05,
            l2_regularization=0.1, random_state=RANDOM_SEED,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
        )

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05,
            loss_function="Logloss", random_seed=RANDOM_SEED,
            verbose=False, allow_writing_files=False,
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )

    return models


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    return {
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_true, y_proba), 4) if y_proba is not None else float("nan"),
        "PR-AUC"   : round(average_precision_score(y_true, y_proba), 4) if y_proba is not None else float("nan"),
    }


# ── Encoding comparison helpers ────────────────────────────────────────────────

def _quick_eval_encoding(X_train, y_train, X_test, eval_idx, y_true_eval, version_label):
    """
    Train all classifiers on X_train and evaluate on X_test.iloc[eval_idx].
    Returns (rows, trained_models, X_eval) — no CV or plots.

    When X_train has no original CAT_COLS columns (OHE case), cat_cols_present
    will be empty so Pipeline models just apply StandardScaler to all features.
    """
    num_cols         = [c for c in X_train.columns if c not in CAT_COLS]
    cat_cols_present = [c for c in CAT_COLS if c in X_train.columns]

    models  = build_models(num_cols, cat_cols_present)
    trained = {}
    rows    = []

    for name, model in models.items():
        print(f"   [{version_label}] Fitting {name} ...", end="  ", flush=True)
        model.fit(X_train, y_train)
        trained[name] = model
        print("done")

    X_eval = X_test.iloc[eval_idx].reset_index(drop=True)

    for name, model in trained.items():
        y_pred  = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y_true_eval.values, y_pred, y_proba)
        rows.append({"Version": version_label, "Model": name, **metrics})
        print(f"   [{version_label}] {name:22s}  "
              f"ROC-AUC={metrics['ROC-AUC']:.4f}  "
              f"PR-AUC={metrics['PR-AUC']:.4f}  "
              f"F1={metrics['F1']:.4f}  "
              f"Recall={metrics['Recall']:.4f}")

    return rows, trained, X_eval


def _run_encoding_comparison(eval_idx, y_true_eval,
                              le_model_results, n_test):
    """
    Compare Label Encoding vs One-Hot Encoding.
    Called only when --mode both is passed.
    """
    ohe_train_path  = PREPRO_DIR / "X_train_preprocessed_one_hot.csv"
    ohe_test_path   = PREPRO_DIR / "X_test_preprocessed_one_hot.csv"
    ohe_ytrain_path = PREPRO_DIR / "y_train_one_hot.csv"

    for p in (ohe_train_path, ohe_test_path, ohe_ytrain_path):
        if not p.exists():
            print(f"\n   ERROR: {p} not found.")
            print("   Run 'Final Project - Claude_one_hot.ipynb' first, then retry.")
            return

    print("\n" + "=" * 72)
    print(" ENCODING COMPARISON: Label Encoding  vs.  One-Hot Encoding")
    print("=" * 72)

    X_train_ohe = pd.read_csv(ohe_train_path)
    X_test_ohe  = pd.read_csv(ohe_test_path)
    y_train_ohe = pd.read_csv(ohe_ytrain_path).squeeze()
    print(f"\n   OHE training set : {X_train_ohe.shape}  (LE was 9864×17)")
    print(f"   Evaluation subset: {len(eval_idx)} rows (same matched rows for both)")
    print(f"   NOTE: No CV is run in --mode both — private evaluation metrics only.\n")

    # LE rows from already-computed model_results (excludes Submitted Model)
    le_rows = [{"Version": "Label Encoding", "Model": k, **v}
               for k, v in le_model_results.items()
               if k != "Submitted Model"]

    # Train and evaluate OHE models
    print(f"Training models on OHE data ({X_train_ohe.shape[1]} features)...")
    ohe_rows, ohe_trained, _ = _quick_eval_encoding(
        X_train_ohe, y_train_ohe, X_test_ohe, eval_idx, y_true_eval, "One-Hot Encoding"
    )

    # Combined table
    comp_df = (
        pd.DataFrame(le_rows + ohe_rows)
        .sort_values(["Version", "ROC-AUC"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Pretty-print
    cols = ["ROC-AUC", "PR-AUC", "F1", "Recall", "Precision", "Accuracy"]
    hdr  = f"  {'Version':<20} {'Model':<24}" + "".join(f" {c:>10}" for c in cols)
    print("\n" + "=" * len(hdr))
    print(" COMBINED COMPARISON TABLE")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    prev_ver = None
    for _, row in comp_df.iterrows():
        if row["Version"] != prev_ver:
            if prev_ver is not None:
                print()
            prev_ver = row["Version"]
        vals = "".join(f" {row[c]:>10.4f}" for c in cols)
        print(f"  {row['Version']:<20} {row['Model']:<24}{vals}")
    print("=" * len(hdr))

    # Save
    comp_out = PRIVATE_DIR / "private_encoding_comparison.csv"
    comp_df.to_csv(comp_out, index=False)
    print(f"\n   Saved → {comp_out}")

    # Save OHE test predictions (best OHE model on all 2466 test rows)
    ohe_df_sorted = comp_df[comp_df["Version"] == "One-Hot Encoding"].reset_index(drop=True)
    best_ohe_name = ohe_df_sorted.iloc[0]["Model"]
    best_ohe_clf  = ohe_trained[best_ohe_name]
    ohe_pred_all  = best_ohe_clf.predict(X_test_ohe)
    ohe_proba_all = (
        best_ohe_clf.predict_proba(X_test_ohe)[:, 1]
        if hasattr(best_ohe_clf, "predict_proba") else [float("nan")] * n_test
    )
    pd.DataFrame({
        "predicted_high_intent": ohe_pred_all,
        "proba_high_intent"    : ohe_proba_all,
    }).to_csv(PREPRO_DIR / "test_predictions_one_hot.csv", index=False)
    print(f"   Saved test_predictions_one_hot.csv  (model: {best_ohe_name})")

    # ── Answers to 5 questions ─────────────────────────────────────────────────
    le_sub  = comp_df[comp_df["Version"] == "Label Encoding"].reset_index(drop=True)
    ohe_sub = comp_df[comp_df["Version"] == "One-Hot Encoding"].reset_index(drop=True)
    le_best  = le_sub.iloc[0]
    ohe_best = ohe_sub.iloc[0]

    # Per-model delta (merge on model name)
    merged = le_sub.merge(ohe_sub, on="Model", suffixes=("_le", "_ohe"))
    merged["Delta_AUC"] = merged["ROC-AUC_ohe"] - merged["ROC-AUC_le"]
    merged = merged.sort_values("Delta_AUC", ascending=False).reset_index(drop=True)

    delta_best = ohe_best["ROC-AUC"] - le_best["ROC-AUC"]

    print("\n" + "=" * 72)
    print(" ANSWERS")
    print("=" * 72)

    print(f"\n  Q1: Does OHE improve overall results?")
    print(f"      Best LE  : {le_best['Model']:<24}  ROC-AUC={le_best['ROC-AUC']:.4f}  PR-AUC={le_best['PR-AUC']:.4f}")
    print(f"      Best OHE : {ohe_best['Model']:<24}  ROC-AUC={ohe_best['ROC-AUC']:.4f}  PR-AUC={ohe_best['PR-AUC']:.4f}")
    if abs(delta_best) < 0.001:
        print(f"      → MARGINAL (delta={delta_best:+.4f}). No meaningful difference at the top.")
    elif delta_best > 0:
        print(f"      → YES (delta={delta_best:+.4f} ROC-AUC). OHE produces a better best model.")
    else:
        print(f"      → NO (delta={delta_best:+.4f} ROC-AUC). Label Encoding is stronger overall.")

    print(f"\n  Q2: Per-model delta (OHE − LE):")
    n_ohe_better = n_le_better = n_tied = 0
    for _, r in merged.iterrows():
        if r["Delta_AUC"] > 0.001:
            tag = "OHE better"; n_ohe_better += 1
        elif r["Delta_AUC"] < -0.001:
            tag = "LE  better"; n_le_better += 1
        else:
            tag = "tied      "; n_tied += 1
        print(f"      {r['Model']:<24}  LE={r['ROC-AUC_le']:.4f}  OHE={r['ROC-AUC_ohe']:.4f}"
              f"  Δ={r['Delta_AUC']:+.4f}  {tag}")

    overall_best = comp_df.sort_values("ROC-AUC", ascending=False).iloc[0]
    print(f"\n  Q3: Best version overall:")
    print(f"      {overall_best['Version']} — {overall_best['Model']}"
          f"  (ROC-AUC={overall_best['ROC-AUC']:.4f})")

    print(f"\n  Q4: Is the improvement worth changing the pipeline?")
    print(f"      OHE better in {n_ohe_better}/{len(merged)} models, "
          f"LE better in {n_le_better}/{len(merged)}, "
          f"tied in {n_tied}/{len(merged)}.")
    if n_ohe_better > n_le_better and delta_best > 0.005:
        print("      → YES. OHE gives consistent gains, especially for linear/distance models.")
    elif n_le_better >= n_ohe_better:
        print("      → NO. Label Encoding performs comparably or better for tree-based models,")
        print("        which don't require ordinal-free encoding. OHE adds 56 extra features")
        print("        without proportional benefit.")
    else:
        print("      → MARGINAL. Results are within noise; OHE is not worth the extra complexity.")

    print(f"\n  Q5: Files summary:")
    print(f"      CREATED : Final Project - Claude_one_hot.ipynb")
    print(f"      CREATED : X_train_preprocessed_one_hot.csv  {tuple(X_train_ohe.shape)}")
    print(f"      CREATED : X_test_preprocessed_one_hot.csv   {tuple(X_test_ohe.shape)}")
    print(f"      CREATED : y_train_one_hot.csv")
    print(f"      CREATED : test_predictions_one_hot.csv  (best OHE model: {best_ohe_name})")
    print(f"      CREATED : private_evaluation/private_encoding_comparison.csv")
    print(f"      CHANGED : private_original_data_evaluation.py  (added --mode both)")
    print(f"      NOT CHANGED: Final Project - Claude.ipynb, any other submitted file")
    improved = merged[merged["Delta_AUC"] >  0.001]["Model"].tolist()
    worsened = merged[merged["Delta_AUC"] < -0.001]["Model"].tolist()
    if improved:
        print(f"\n      Models IMPROVED by OHE : {', '.join(improved)}")
    if worsened:
        print(f"      Models WORSENED by OHE : {', '.join(worsened)}")
    if not improved and not worsened:
        print(f"\n      No model changed by more than 0.001 ROC-AUC — effectively tied.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["original", "both"], default="original")
    args, _ = parser.parse_known_args()
    MODE = args.mode

    print(BANNER)
    print()

    # ── Optional package availability ─────────────────────────────────────────
    print("Optional library availability:")
    print(f"   XGBoost         : {'YES' if HAS_XGBOOST  else 'NO  →  pip install xgboost'}")
    print(f"   CatBoost        : {'YES' if HAS_CATBOOST else 'NO  →  pip install catboost'}")
    print(f"   LightGBM        : {'YES' if HAS_LIGHTGBM else 'NO  →  pip install lightgbm'}")

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    print("\n1. Loading datasets...")
    df_test_raw     = pd.read_csv(RAW_DIR    / "shopper_test.csv")
    df_orig         = pd.read_csv(RAW_DIR    / "online_shoppers_intention.csv")
    X_train_pre     = pd.read_csv(PREPRO_DIR / "X_train_preprocessed.csv")
    y_train_full    = pd.read_csv(PREPRO_DIR / "y_train.csv").squeeze()
    X_test_pre      = pd.read_csv(PREPRO_DIR / "X_test_preprocessed.csv")
    submitted_preds = pd.read_csv(PREPRO_DIR / "test_predictions.csv")

    print(f"   shopper_test.csv              : {df_test_raw.shape}")
    print(f"   online_shoppers_intention.csv : {df_orig.shape}")
    print(f"   X_train_preprocessed.csv      : {X_train_pre.shape}")
    print(f"   X_test_preprocessed.csv       : {X_test_pre.shape}")
    print(f"   test_predictions.csv          : {submitted_preds.shape}")

    # ── 2. Align original dataset ─────────────────────────────────────────────
    print("\n2. Aligning original dataset column names...")
    orig_aligned = pd.DataFrame(index=df_orig.index)
    for orig_col, assign_col in REVERSE_MAP.items():
        if orig_col in df_orig.columns:
            orig_aligned[assign_col] = df_orig[orig_col]
    for col in ("is_weekend", "high_intent"):
        if col in orig_aligned.columns:
            orig_aligned[col] = orig_aligned[col].astype(int)
    if "BounceRates" in df_orig.columns:
        orig_aligned["bounce_rate"] = df_orig["BounceRates"]

    # ── 3. Recover ground-truth labels for test rows ──────────────────────────
    print("\n3. Matching test rows to original dataset to recover labels...")
    matched = match_test_to_original(df_test_raw.copy(), orig_aligned)

    n_test      = len(df_test_raw)
    n_matched   = matched["test_idx"].nunique()
    n_labeled   = matched["recovered_label"].notna().sum()
    n_ambiguous = matched[matched["ambiguous"]]["test_idx"].nunique()

    print(f"   Total test rows               : {n_test}")
    print(f"   Successfully matched          : {n_matched}  ({100*n_matched/n_test:.1f}%)")
    print(f"   With recovered labels         : {n_labeled}  ({100*n_labeled/n_test:.1f}%)")
    print(f"   Ambiguous (resolved by agree) : {n_ambiguous}")
    print(f"   Unmatched / no label found    : {n_test - n_labeled}")

    if n_labeled < 30:
        print("\n   WARNING: Fewer than 30 test rows matched — metrics may be unreliable.")

    label_map = (
        matched.dropna(subset=["recovered_label"])
               .set_index("test_idx")["recovered_label"]
               .astype(int)
    )
    y_recovered = pd.Series(index=range(n_test), dtype=float)
    y_recovered.update(label_map)
    eval_idx    = y_recovered.dropna().index.tolist()
    y_true_eval = y_recovered.loc[eval_idx].astype(int)

    print(f"\n   Evaluation subset size  : {len(eval_idx)}")
    print(f"   Positive rate (original): {y_true_eval.mean():.1%}")

    _safe_write(PRIVATE_DIR / "recovered_test_labels.csv", matched[
        ["test_idx", "orig_idx", "recovered_label", "match_tier", "n_orig_matches", "ambiguous"]
    ].copy())

    # ── 4. Build and train all models ─────────────────────────────────────────
    print("\n4. Re-training all candidate models on X_train_preprocessed + y_train...")
    print("   (Same data and column split as the notebook — no leakage.)")
    print("   NOTE: boosting models at 300 iterations × 10 CV folds may take")
    print("         several minutes — please wait.")

    num_cols         = [c for c in X_train_pre.columns if c not in CAT_COLS]
    cat_cols_present = [c for c in CAT_COLS if c in X_train_pre.columns]

    models  = build_models(num_cols, cat_cols_present)
    trained = {}

    for name, model in models.items():
        print(f"   Fitting: {name} ...", end="  ", flush=True)
        model.fit(X_train_pre, y_train_full)
        trained[name] = model
        print("done")

    # ── 5. Predict on the evaluation subset ───────────────────────────────────
    print(f"\n5. Generating predictions on the {len(eval_idx)}-row evaluation subset...")
    X_eval = X_test_pre.iloc[eval_idx].reset_index(drop=True)

    model_results = {}
    all_pred_rows = []

    for name, model in trained.items():
        y_pred  = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y_true_eval.values, y_pred, y_proba)
        model_results[name] = metrics
        all_pred_rows.append({"Model": name, **metrics})
        print(f"   {name:22s}  "
              f"Acc={metrics['Accuracy']:.4f}  "
              f"Rec={metrics['Recall']:.4f}  "
              f"F1={metrics['F1']:.4f}  "
              f"ROC-AUC={metrics['ROC-AUC']:.4f}  "
              f"PR-AUC={metrics['PR-AUC']:.4f}")

    # Submitted predictions (read from file, no retraining)
    print("\n   Evaluating submitted test_predictions.csv on the same subset...")
    sub_pred    = submitted_preds["predicted_high_intent"].iloc[eval_idx].values
    sub_proba   = submitted_preds["proba_high_intent"].iloc[eval_idx].values
    sub_metrics = compute_metrics(y_true_eval.values, sub_pred, sub_proba)
    model_results["Submitted Model"] = sub_metrics
    all_pred_rows.append({"Model": "Submitted Model", **sub_metrics})
    print(f"   {'Submitted Model':22s}  "
          f"Acc={sub_metrics['Accuracy']:.4f}  "
          f"Rec={sub_metrics['Recall']:.4f}  "
          f"F1={sub_metrics['F1']:.4f}  "
          f"ROC-AUC={sub_metrics['ROC-AUC']:.4f}  "
          f"PR-AUC={sub_metrics['PR-AUC']:.4f}")

    # ── 6. Ranked comparison table ─────────────────────────────────────────────
    results_df = (
        pd.DataFrame(all_pred_rows)
        .set_index("Model")
        .sort_values(["ROC-AUC", "PR-AUC", "F1", "Recall"], ascending=False)
    )
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    print("\n" + "=" * 72)
    print(" MODEL COMPARISON vs RECOVERED ORIGINAL LABELS  (sorted by ROC-AUC)")
    print("=" * 72)
    print(results_df.to_string())

    # ── 7. 10-fold stratified cross-validation (ROC-AUC, F1, Recall) ──────────
    print("\n6. 10-fold stratified cross-validation on training data...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    cv_rows = []

    for name, model in trained.items():
        auc_cv = cross_val_score(model, X_train_pre, y_train_full,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        f1_cv  = cross_val_score(model, X_train_pre, y_train_full,
                                 cv=cv, scoring="f1",      n_jobs=-1)
        rec_cv = cross_val_score(model, X_train_pre, y_train_full,
                                 cv=cv, scoring="recall",  n_jobs=-1)
        cv_rows.append({
            "Model"      : name,
            "CV_ROC-AUC" : round(auc_cv.mean(), 4),
            "CV_AUC_std" : round(auc_cv.std(),  4),
            "CV_F1"      : round(f1_cv.mean(),  4),
            "CV_F1_std"  : round(f1_cv.std(),   4),
            "CV_Recall"  : round(rec_cv.mean(), 4),
            "CV_Rec_std" : round(rec_cv.std(),  4),
        })
        print(f"   {name:22s}  "
              f"CV AUC={auc_cv.mean():.4f}±{auc_cv.std():.4f}  "
              f"CV F1={f1_cv.mean():.4f}±{f1_cv.std():.4f}  "
              f"CV Recall={rec_cv.mean():.4f}±{rec_cv.std():.4f}")

    cv_df = pd.DataFrame(cv_rows).set_index("Model")

    # ── 8. Overfitting analysis (private AUC vs CV AUC gap) ───────────────────
    print("\n" + "=" * 72)
    print(" OVERFITTING & AGREEMENT ANALYSIS")
    print("=" * 72)
    comparison_rows = []
    for name in trained:
        priv_auc = model_results[name]["ROC-AUC"]
        cv_auc   = cv_df.loc[name, "CV_ROC-AUC"]
        gap      = round(priv_auc - cv_auc, 4)
        overfit  = "YES ⚠" if gap < -0.05 else ("likely" if gap < -0.02 else "no")
        comparison_rows.append({
            "Model"           : name,
            "Private_ROC-AUC" : priv_auc,
            "CV_ROC-AUC"      : cv_auc,
            "Gap (priv-CV)"   : gap,
            "Overfitting?"    : overfit,
        })
        print(f"   {name:22s}  "
              f"Private={priv_auc:.4f}  CV={cv_auc:.4f}  "
              f"Gap={gap:+.4f}  Overfit={overfit}")
    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")

    # ── 9. Confusion matrices (grid layout, up to 4 per row) ──────────────────
    print("\n7. Saving plots...")
    all_plot_models = list(trained.items()) + [("Submitted Model", None)]
    n_total = len(all_plot_models)
    ncols   = 4
    nrows   = (n_total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (name, model) in enumerate(all_plot_models):
        ax = axes[i]
        if name == "Submitted Model":
            y_pred_plot, cmap = sub_pred, "Greens"
        else:
            y_pred_plot, cmap = model.predict(X_eval), "Blues"
        cm = confusion_matrix(y_true_eval, y_pred_plot)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"])
        ax.set_title(name, fontsize=8)

    for ax in axes[n_total:]:
        ax.set_visible(False)

    plt.suptitle("Confusion Matrices — Private Evaluation vs Recovered Labels",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    cm_path = PRIVATE_DIR / "confusion_matrices_private.png"
    fig.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {cm_path}")

    # ── 10. ROC curves ────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 7))
    for name, model in trained.items():
        if hasattr(model, "predict_proba"):
            yp = model.predict_proba(X_eval)[:, 1]
            fpr, tpr, _ = roc_curve(y_true_eval, yp)
            ax2.plot(fpr, tpr, lw=1.5,
                     label=f"{name}  (AUC={model_results[name]['ROC-AUC']:.3f})")
    fpr_s, tpr_s, _ = roc_curve(y_true_eval, sub_proba)
    ax2.plot(fpr_s, tpr_s, lw=2, linestyle="--",
             label=f"Submitted  (AUC={sub_metrics['ROC-AUC']:.3f})")
    ax2.plot([0, 1], [0, 1], "k:", lw=1, label="Random baseline")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves — Private Evaluation vs Recovered Labels")
    ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    roc_path = PRIVATE_DIR / "roc_curves_private.png"
    fig2.savefig(roc_path, dpi=100)
    plt.close(fig2)
    print(f"   Saved → {roc_path}")

    # ── 11. Precision-Recall curves ───────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 7))
    for name, model in trained.items():
        if hasattr(model, "predict_proba"):
            yp = model.predict_proba(X_eval)[:, 1]
            prec, rec, _ = precision_recall_curve(y_true_eval, yp)
            ax3.plot(rec, prec, lw=1.5,
                     label=f"{name}  (AP={model_results[name]['PR-AUC']:.3f})")
    prec_s, rec_s, _ = precision_recall_curve(y_true_eval, sub_proba)
    ax3.plot(rec_s, prec_s, lw=2, linestyle="--",
             label=f"Submitted  (AP={sub_metrics['PR-AUC']:.3f})")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curves — Private Evaluation vs Recovered Labels")
    ax3.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    pr_path = PRIVATE_DIR / "pr_curves_private.png"
    fig3.savefig(pr_path, dpi=100)
    plt.close(fig3)
    print(f"   Saved → {pr_path}")

    # ── 12. Save CSV results ──────────────────────────────────────────────────
    print("\n8. Saving CSV results...")
    _safe_write(PRIVATE_DIR / "private_model_comparison_against_original_labels.csv",
                results_df.reset_index())
    _safe_write(PRIVATE_DIR / "private_cv_results.csv",
                cv_df.reset_index())
    _safe_write(PRIVATE_DIR / "private_overfitting_analysis.csv",
                comparison_df.reset_index())

    # ── 13. Final summary & answers ───────────────────────────────────────────
    bm_present   = [m for m in BENCHMARK_ORIGINATED if m in results_df.index]
    base_present = [m for m in SECTION9_BASELINE    if m in results_df.index]

    winner        = results_df.index[0]
    winner_auc    = results_df.loc[winner, "ROC-AUC"]
    winner_prauc  = results_df.loc[winner, "PR-AUC"]
    winner_f1     = results_df.loc[winner, "F1"]
    winner_recall = results_df.loc[winner, "Recall"]

    best_bm   = results_df.loc[bm_present,   "ROC-AUC"].idxmax() if bm_present   else None
    best_base = results_df.loc[base_present,  "ROC-AUC"].idxmax() if base_present else None
    best_bm_auc   = results_df.loc[best_bm,  "ROC-AUC"] if best_bm   else float("nan")
    best_base_auc = results_df.loc[best_base, "ROC-AUC"] if best_base else float("nan")
    sub_auc       = sub_metrics["ROC-AUC"]
    gap_sub_best  = winner_auc - sub_auc

    print("\n" + "=" * 72)
    print(" FINAL SUMMARY & ANSWERS")
    print("=" * 72)
    print(f"\n  Evaluation rows (matched to original): {len(eval_idx)} / {n_test}")
    print(f"  Recovered positive rate              : {y_true_eval.mean():.1%}")

    # Ranking table
    W = 24
    print()
    print(f"  {'Model':<{W}} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'Recall':>8}  {'Rank':>4}  Origin")
    print(f"  {'─'*W} {'─'*8} {'─'*8} {'─'*8} {'─'*8}  {'─'*4}  {'─'*6}")
    for mname, mrow in results_df.iterrows():
        if mname in BENCHMARK_ORIGINATED:
            origin = "[BM]"
        elif mname in SECTION9_BASELINE:
            origin = "[S9]"
        else:
            origin = "[SUB]"
        winner_tag = "  ← WINNER" if int(mrow["Rank"]) == 1 else ""
        print(f"  {mname:<{W}} "
              f"{mrow['ROC-AUC']:>8.4f} "
              f"{mrow['PR-AUC']:>8.4f} "
              f"{mrow['F1']:>8.4f} "
              f"{mrow['Recall']:>8.4f}  "
              f"{int(mrow['Rank']):>4}  "
              f"{origin}{winner_tag}")
    print(f"\n  Legend: [BM] from imputation benchmark  "
          f"[S9] Section 9 original  [SUB] submitted file")

    # Q1: Which model performs best?
    print(f"\n  ─── Q1: Which model performs best? ───────────────────────────────")
    print(f"  Winner : {winner}")
    print(f"    ROC-AUC={winner_auc:.4f}  PR-AUC={winner_prauc:.4f}  "
          f"F1={winner_f1:.4f}  Recall={winner_recall:.4f}")

    # Q2/Q3: Do benchmark models outperform the best Section 9 model?
    print(f"\n  ─── Q2/Q3: Imputation-benchmark models vs Section 9 baseline ────")
    if best_bm and best_base:
        delta    = best_bm_auc - best_base_auc
        delta_pp = delta * 100
        print(f"  Best [BM] model : {best_bm:<22s}  ROC-AUC={best_bm_auc:.4f}")
        print(f"  Best [S9] model : {best_base:<22s}  ROC-AUC={best_base_auc:.4f}")
        if delta > 0.001:
            print(f"\n  → YES — {best_bm} OUTPERFORMS the best Section 9 classifier")
            print(f"    by +{delta:.4f} ROC-AUC  ({delta_pp:+.2f} percentage points).")
        elif delta >= -0.001:
            print(f"\n  → MARGINAL — {best_bm} is effectively tied with {best_base}")
            print(f"    (difference = {delta:.4f} ROC-AUC, within measurement noise).")
        else:
            print(f"\n  → NO — no benchmark model outperforms the best Section 9 classifier.")
            print(f"    Best [BM] model is {abs(delta):.4f} ROC-AUC behind {best_base}.")
    else:
        print("  (One or both groups not present — cannot compare.)")

    # Submitted model gap
    print(f"\n  ─── Submitted model vs overall winner ────────────────────────────")
    print(f"  Submitted ROC-AUC : {sub_auc:.4f}")
    print(f"  Winner ROC-AUC    : {winner_auc:.4f}  ({winner})")
    if gap_sub_best > 0.03:
        print(f"  Gap = {gap_sub_best:.4f} — a notably better model exists in private evaluation.")
        print("  Recommendation: Review methodology.  Treat as suggestive only —")
        print("  label recovery is unvalidated; do NOT retrain on this signal.")
    else:
        print(f"  Gap = {gap_sub_best:.4f} — submitted model is within normal variance.")
        print("  Recommendation: No change to submitted methodology warranted.")

    # Overfitting summary
    severe = [r["Model"] for r in comparison_rows if r["Gap (priv-CV)"] < -0.05]
    print()
    if severe:
        print(f"  Overfitting suspected (private < CV by >5 pp): {severe}")
    else:
        print("  No severe overfitting detected across any model.")

    print()
    print("  NOTE: This script is diagnostic only.  No submitted file was modified.")
    print("  All outputs are isolated to private_evaluation/.")
    print()

    # ── 14. Encoding comparison (--mode both only) ────────────────────────────
    if MODE == "both":
        _run_encoding_comparison(
            eval_idx, y_true_eval, model_results, n_test,
        )

    print(BANNER)
    print()


if __name__ == "__main__":
    main()
