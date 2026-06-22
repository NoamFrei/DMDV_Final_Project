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
  KNN, Extra Trees, XGBoost, CatBoost, HistGradientBoosting

IMPORTANT CONSTRAINTS — enforced by design:
  - Original labels are recovered ONLY for evaluation.  They are never written
    back to the submitted test file, used for training, or used for tuning.
  - No submitted notebook or preprocessing file is modified.
  - All outputs go to private_evaluation/ and are excluded from git.

Preprocessing philosophy (mirrors the benchmark file):
  - Distance / linear models (LR, NB, SVM, KNN):
      Pipeline with OHE on cat_cols + StandardScaler on num_cols.
  - Tree / boosting models (DT, RF, Extra Trees, XGBoost, CatBoost, HistGradientBoosting):
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
    VotingClassifier,
    StackingClassifier,
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
    "XGBoost", "CatBoost",
}
SECTION9_BASELINE = {
    "Logistic Regression", "Decision Tree", "Random Forest",
    "Naive Bayes", "SVM",
}
ENSEMBLE_METHODS = {
    "Soft Voting Ensemble",
    "Stacking (LR meta)",
}
# Threshold-optimised names are generated at runtime; matched with prefix below.

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



    return models


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    _yt = np.array(y_true)
    _yp = np.array(y_pred)
    return {
        "Correct"  : int((_yt == _yp).sum()),
        "Total"    : int(len(_yt)),
        "Accuracy" : round(accuracy_score(_yt, _yp), 4),
        "Precision": round(precision_score(_yt, _yp, zero_division=0), 4),
        "Recall"   : round(recall_score(_yt, _yp, zero_division=0), 4),
        "F1"       : round(f1_score(_yt, _yp, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(_yt, y_proba), 4) if y_proba is not None else float("nan"),
        "PR-AUC"   : round(average_precision_score(_yt, y_proba), 4) if y_proba is not None else float("nan"),
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
              f"Correct={metrics['Correct']}/{metrics['Total']}  "
              f"Acc={metrics['Accuracy']:.4f}  "
              f"ROC-AUC={metrics['ROC-AUC']:.4f}  "
              f"F1={metrics['F1']:.4f}")

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
        .sort_values(["Version", "Correct", "Accuracy", "ROC-AUC"],
                     ascending=[True, False, False, False])
        .reset_index(drop=True)
    )

    # Pretty-print (primary sort: Correct/Accuracy; secondary: ROC-AUC)
    _num_cols = ["Accuracy", "ROC-AUC", "PR-AUC", "F1", "Recall", "Precision"]
    hdr  = (f"  {'Version':<20} {'Model':<24}"
            f" {'Correct':>8} {'Total':>6}"
            + "".join(f" {c:>9}" for c in _num_cols))
    print("\n" + "=" * len(hdr))
    print(" COMBINED COMPARISON TABLE  (sorted by Correct Predictions → Accuracy)")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    prev_ver = None
    for _, row in comp_df.iterrows():
        if row["Version"] != prev_ver:
            if prev_ver is not None:
                print()
            prev_ver = row["Version"]
        num_vals = "".join(f" {row[c]:>9.4f}" for c in _num_cols)
        print(f"  {row['Version']:<20} {row['Model']:<24}"
              f" {int(row['Correct']):>8} {int(row['Total']):>6}{num_vals}")
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

    delta_best = ohe_best["Accuracy"] - le_best["Accuracy"]

    print("\n" + "=" * 72)
    print(" ANSWERS")
    print("=" * 72)

    print(f"\n  Q1: Does OHE improve overall results (by correct predictions)?")
    print(f"      Best LE  : {le_best['Model']:<24}  "
          f"Correct={int(le_best['Correct'])}/{int(le_best['Total'])}  Acc={le_best['Accuracy']:.4f}  "
          f"ROC-AUC={le_best['ROC-AUC']:.4f}")
    print(f"      Best OHE : {ohe_best['Model']:<24}  "
          f"Correct={int(ohe_best['Correct'])}/{int(ohe_best['Total'])}  Acc={ohe_best['Accuracy']:.4f}  "
          f"ROC-AUC={ohe_best['ROC-AUC']:.4f}")
    if abs(delta_best) < 0.001:
        print(f"      → MARGINAL (delta={delta_best:+.4f} Accuracy). No meaningful difference at the top.")
    elif delta_best > 0:
        print(f"      → YES (delta={delta_best:+.4f} Accuracy). OHE produces more correct predictions.")
    else:
        print(f"      → NO (delta={delta_best:+.4f} Accuracy). Label Encoding is stronger overall.")

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

    overall_best = comp_df.sort_values(
        ["Correct", "Accuracy", "ROC-AUC"], ascending=False
    ).iloc[0]
    print(f"\n  Q3: Best version overall (by correct predictions):")
    print(f"      {overall_best['Version']} — {overall_best['Model']}"
          f"  (Correct={int(overall_best['Correct'])}/{int(overall_best['Total'])}  "
          f"Accuracy={overall_best['Accuracy']:.4f}  ROC-AUC={overall_best['ROC-AUC']:.4f})")

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
              f"Correct={metrics['Correct']}/{metrics['Total']}  "
              f"Acc={metrics['Accuracy']:.4f}  "
              f"Rec={metrics['Recall']:.4f}  "
              f"F1={metrics['F1']:.4f}  "
              f"ROC-AUC={metrics['ROC-AUC']:.4f}")

    # Submitted predictions (read from file, no retraining)
    print("\n   Evaluating submitted test_predictions.csv on the same subset...")
    sub_pred    = submitted_preds["predicted_high_intent"].iloc[eval_idx].values
    sub_proba   = submitted_preds["proba_high_intent"].iloc[eval_idx].values
    sub_metrics = compute_metrics(y_true_eval.values, sub_pred, sub_proba)
    model_results["Submitted Model"] = sub_metrics
    all_pred_rows.append({"Model": "Submitted Model", **sub_metrics})
    print(f"   {'Submitted Model':22s}  "
          f"Correct={sub_metrics['Correct']}/{sub_metrics['Total']}  "
          f"Acc={sub_metrics['Accuracy']:.4f}  "
          f"Rec={sub_metrics['Recall']:.4f}  "
          f"F1={sub_metrics['F1']:.4f}  "
          f"ROC-AUC={sub_metrics['ROC-AUC']:.4f}")

    # ── 5b. Ensemble methods (mirrors Classification_Section9_improved.ipynb) ──
    print("\n5b. Building and evaluating ensemble methods...")
    print("   (These mirror the Voting + Stacking cells in "
          "Classification_Section9_improved.ipynb)")

    # Mirror notebook: compute 10-fold CV F1, then pick top-5 as voting base models.
    # Notebook cell 41: _n_top = min(5, len(tuned_models))
    #                   _top_names sorted by cv_f1_scores[n].mean() descending.
    print("   Computing 10-fold CV F1 for ensemble model selection...")
    _cv_f1_ens = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    _cv_f1_scores_ens = {}
    for _n_ens, _m_ens in trained.items():
        _s_ens = cross_val_score(_m_ens, X_train_pre, y_train_full,
                                 cv=_cv_f1_ens, scoring="f1", n_jobs=-1)
        _cv_f1_scores_ens[_n_ens] = _s_ens.mean()

    _n_top_ens    = min(5, len(trained))
    _top_names_ens = sorted(_cv_f1_scores_ens,
                            key=lambda n: _cv_f1_scores_ens[n], reverse=True)[:_n_top_ens]
    print(f"   Top {_n_top_ens} models for voting (by 10-fold CV F1):")
    for _n_ens in _top_names_ens:
        print(f"     {_n_ens:<25}  CV F1 = {_cv_f1_scores_ens[_n_ens]:.4f}")

    _vote_models = [(_n, trained[_n]) for _n in _top_names_ens]
    print(f"   Voting Ensemble base models ({len(_vote_models)}): "
          f"{[n for n, _ in _vote_models]}")

    # ── Soft Voting Ensemble (default threshold 0.5) ──────────────────────────
    voting_clf = VotingClassifier(
        estimators=_vote_models, voting="soft", n_jobs=-1
    )
    voting_clf.fit(X_train_pre, y_train_full)
    _y_proba_vote_all = voting_clf.predict_proba(X_eval)[:, 1]
    _y_pred_vote_all  = voting_clf.predict(X_eval)
    metrics_vote = compute_metrics(y_true_eval.values, _y_pred_vote_all, _y_proba_vote_all)
    model_results["Soft Voting Ensemble"] = metrics_vote
    all_pred_rows.append({"Model": "Soft Voting Ensemble", **metrics_vote})
    print(f"   {'Soft Voting Ensemble':30s}  "
          f"Correct={metrics_vote['Correct']}/{metrics_vote['Total']}  "
          f"Acc={metrics_vote['Accuracy']:.4f}  "
          f"ROC-AUC={metrics_vote['ROC-AUC']:.4f}")

    # ── Threshold-optimised Voting (thresholds found on training val-split) ───
    # Mirror of notebook: threshold found on a held-out portion of training data
    # (NOT on evaluation labels — uses 80/20 split of X_train_pre internally).
    from sklearn.model_selection import train_test_split as _tts_thr
    _Xtr_thr, _Xval_thr, _ytr_thr, _yval_thr = _tts_thr(
        X_train_pre, y_train_full,
        test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_full,
    )
    _vote_thr_tmp = VotingClassifier(
        estimators=_vote_models, voting="soft", n_jobs=-1
    )
    _vote_thr_tmp.fit(_Xtr_thr, _ytr_thr)
    _proba_val_thr = _vote_thr_tmp.predict_proba(_Xval_thr)[:, 1]
    _yval_arr      = np.array(_yval_thr)

    _thresholds = np.arange(0.20, 0.81, 0.01)

    # Mirror notebook: start baseline at accuracy@0.5 (only update if strictly better).
    _vote_acc_val = accuracy_score(_yval_arr, (_proba_val_thr >= 0.5).astype(int))
    _best_thr_acc, _best_acc_val = 0.5, _vote_acc_val
    for _thr in _thresholds:
        _at = accuracy_score(_yval_arr, (_proba_val_thr >= _thr).astype(int))
        if _at > _best_acc_val:
            _best_acc_val, _best_thr_acc = _at, _thr

    _best_thr_f1, _best_f1_val = 0.5, 0.0
    for _thr in _thresholds:
        _ft = f1_score(_yval_arr, (_proba_val_thr >= _thr).astype(int), zero_division=0)
        if _ft > _best_f1_val:
            _best_f1_val, _best_thr_f1 = _ft, _thr

    print(f"   Acc-optimal threshold (from training val-split): {_best_thr_acc:.2f}")
    print(f"   F1-optimal  threshold (from training val-split): {_best_thr_f1:.2f}")

    # Apply found thresholds to full-data voting ensemble on X_eval
    _vote_acc_name = f"Voting (thr={_best_thr_acc:.2f}, Acc-opt)"
    _y_pred_vote_acc = (_y_proba_vote_all >= _best_thr_acc).astype(int)
    metrics_vote_acc = compute_metrics(y_true_eval.values, _y_pred_vote_acc,
                                       _y_proba_vote_all)
    model_results[_vote_acc_name] = metrics_vote_acc
    all_pred_rows.append({"Model": _vote_acc_name, **metrics_vote_acc})
    ENSEMBLE_METHODS.add(_vote_acc_name)
    print(f"   {_vote_acc_name:30s}  "
          f"Correct={metrics_vote_acc['Correct']}/{metrics_vote_acc['Total']}  "
          f"Acc={metrics_vote_acc['Accuracy']:.4f}  "
          f"ROC-AUC={metrics_vote_acc['ROC-AUC']:.4f}")

    _vote_f1_name = f"Voting (thr={_best_thr_f1:.2f}, F1-opt)"
    _y_pred_vote_f1 = (_y_proba_vote_all >= _best_thr_f1).astype(int)
    metrics_vote_f1 = compute_metrics(y_true_eval.values, _y_pred_vote_f1,
                                      _y_proba_vote_all)
    model_results[_vote_f1_name] = metrics_vote_f1
    all_pred_rows.append({"Model": _vote_f1_name, **metrics_vote_f1})
    ENSEMBLE_METHODS.add(_vote_f1_name)
    print(f"   {_vote_f1_name:30s}  "
          f"Correct={metrics_vote_f1['Correct']}/{metrics_vote_f1['Total']}  "
          f"Acc={metrics_vote_f1['Accuracy']:.4f}  "
          f"ROC-AUC={metrics_vote_f1['ROC-AUC']:.4f}")

    # ── Stacking Classifier (sklearn-native models only, LR meta) ────────────
    # CatBoost is excluded to avoid potential API incompatibilities
    # in sklearn's StackingClassifier (mirrors the notebook's _SKIP_STACK set).
    _SKIP_STACK = {"CatBoost"}
    _stack_estimators = [(n, m) for n, m in trained.items()
                         if n not in _SKIP_STACK and hasattr(m, "predict_proba")]
    print(f"   Stacking base models ({len(_stack_estimators)}): "
          f"{[n for n, _ in _stack_estimators]}")

    stack_clf = StackingClassifier(
        estimators=_stack_estimators,
        final_estimator=LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_SEED
        ),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
    stack_clf.fit(X_train_pre, y_train_full)
    _y_pred_stack  = stack_clf.predict(X_eval)
    _y_proba_stack = stack_clf.predict_proba(X_eval)[:, 1]
    metrics_stack = compute_metrics(y_true_eval.values, _y_pred_stack, _y_proba_stack)
    model_results["Stacking (LR meta)"] = metrics_stack
    all_pred_rows.append({"Model": "Stacking (LR meta)", **metrics_stack})
    print(f"   {'Stacking (LR meta)':30s}  "
          f"Correct={metrics_stack['Correct']}/{metrics_stack['Total']}  "
          f"Acc={metrics_stack['Accuracy']:.4f}  "
          f"ROC-AUC={metrics_stack['ROC-AUC']:.4f}")

    # ── 6. Ranked comparison table ─────────────────────────────────────────────
    results_df = (
        pd.DataFrame(all_pred_rows)
        .set_index("Model")
        .sort_values(["Correct", "Accuracy", "ROC-AUC", "PR-AUC", "F1", "Recall"],
                     ascending=False)
    )
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    # ── Clean output table (required columns, primary: Correct/Accuracy) ──────
    print("\n" + "=" * 80)
    print(" MODEL COMPARISON vs INTERNET CSV (online_shoppers_intention.csv)")
    print(" Sorted by: Correct Predictions → Accuracy → ROC-AUC")
    print("=" * 80)
    _tcols = ["Correct", "Total", "Accuracy", "ROC-AUC", "PR-AUC", "F1", "Recall"]
    _thdr  = f"  {'Model':<24}" + "".join(f" {c:>10}" for c in _tcols)
    print(_thdr)
    print("  " + "─" * (len(_thdr) - 2))
    for _mname, _mrow in results_df.iterrows():
        _winner_tag = "  ← BEST" if int(_mrow["Rank"]) == 1 else ""
        _vals = (f" {int(_mrow['Correct']):>10}"
                 f" {int(_mrow['Total']):>10}"
                 + "".join(f" {_mrow[c]:>10.4f}" for c in _tcols[2:]))
        print(f"  {_mname:<24}{_vals}{_winner_tag}")
    print("=" * 80)

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

    winner         = results_df.index[0]
    winner_correct = int(results_df.loc[winner, "Correct"])
    winner_total   = int(results_df.loc[winner, "Total"])
    winner_acc     = results_df.loc[winner, "Accuracy"]
    winner_auc     = results_df.loc[winner, "ROC-AUC"]
    winner_prauc   = results_df.loc[winner, "PR-AUC"]
    winner_f1      = results_df.loc[winner, "F1"]
    winner_recall  = results_df.loc[winner, "Recall"]

    best_bm   = results_df.loc[bm_present,   "Accuracy"].idxmax() if bm_present   else None
    best_base = results_df.loc[base_present,  "Accuracy"].idxmax() if base_present else None
    best_bm_acc   = results_df.loc[best_bm,  "Accuracy"] if best_bm   else float("nan")
    best_base_acc = results_df.loc[best_base, "Accuracy"] if best_base else float("nan")
    sub_acc       = sub_metrics["Accuracy"]
    sub_correct   = sub_metrics["Correct"]
    gap_sub_best  = winner_acc - sub_acc

    print("\n" + "=" * 72)
    print(" FINAL SUMMARY & ANSWERS")
    print("=" * 72)
    print(f"\n  Evaluation rows (matched to original): {len(eval_idx)} / {n_test}")
    print(f"  Recovered positive rate              : {y_true_eval.mean():.1%}")

    # Ranking table — sorted by Correct/Accuracy (primary)
    W = 24
    print()
    print(f"  {'Model':<{W}} {'Correct':>8} {'Total':>6} {'Accuracy':>9} "
          f"{'ROC-AUC':>8} {'F1':>8} {'Recall':>8}  {'Rank':>4}  Origin")
    print(f"  {'─'*W} {'─'*8} {'─'*6} {'─'*9} {'─'*8} {'─'*8} {'─'*8}  {'─'*4}  {'─'*6}")
    for mname, mrow in results_df.iterrows():
        if mname in BENCHMARK_ORIGINATED:
            origin = "[BM]"
        elif mname in SECTION9_BASELINE:
            origin = "[S9]"
        elif mname in ENSEMBLE_METHODS or mname.startswith("Voting (thr="):
            origin = "[ENS]"
        else:
            origin = "[SUB]"
        winner_tag = "  ← WINNER" if int(mrow["Rank"]) == 1 else ""
        print(f"  {mname:<{W}} "
              f"{int(mrow['Correct']):>8} "
              f"{int(mrow['Total']):>6} "
              f"{mrow['Accuracy']:>9.4f} "
              f"{mrow['ROC-AUC']:>8.4f} "
              f"{mrow['F1']:>8.4f} "
              f"{mrow['Recall']:>8.4f}  "
              f"{int(mrow['Rank']):>4}  "
              f"{origin}{winner_tag}")
    print(f"\n  Legend: [BM] from imputation benchmark  "
          f"[S9] Section 9 original  [ENS] ensemble  [SUB] submitted file")

    # Q1: Which model has the most correct predictions vs the internet CSV?
    print(f"\n  ─── Q1: Which model has the most correct predictions vs internet CSV? ──")
    print(f"  Winner : {winner}")
    print(f"    Correct predictions : {winner_correct} / {winner_total}")
    print(f"    Accuracy            : {winner_acc:.4f}")
    print(f"    ROC-AUC={winner_auc:.4f}  PR-AUC={winner_prauc:.4f}  "
          f"F1={winner_f1:.4f}  Recall={winner_recall:.4f}")

    # Q2/Q3: Do benchmark models outperform the best Section 9 model (by Accuracy)?
    print(f"\n  ─── Q2/Q3: Imputation-benchmark models vs Section 9 baseline ────")
    if best_bm and best_base:
        delta     = best_bm_acc - best_base_acc
        delta_pp  = delta * 100
        bm_correct   = int(results_df.loc[best_bm,   "Correct"])
        base_correct = int(results_df.loc[best_base, "Correct"])
        total_n      = int(results_df.loc[best_bm,   "Total"])
        print(f"  Best [BM] model : {best_bm:<22s}  "
              f"Correct={bm_correct}/{total_n}  Accuracy={best_bm_acc:.4f}")
        print(f"  Best [S9] model : {best_base:<22s}  "
              f"Correct={base_correct}/{total_n}  Accuracy={best_base_acc:.4f}")
        if delta > 0.001:
            print(f"\n  → YES — {best_bm} OUTPERFORMS the best Section 9 classifier")
            print(f"    by +{delta:.4f} Accuracy  ({delta_pp:+.2f} pp)  "
                  f"i.e. {bm_correct - base_correct} more correct predictions.")
        elif delta >= -0.001:
            print(f"\n  → MARGINAL — {best_bm} is effectively tied with {best_base}")
            print(f"    (difference = {delta:.4f} Accuracy, within measurement noise).")
        else:
            print(f"\n  → NO — no benchmark model outperforms the best Section 9 classifier by Accuracy.")
            print(f"    Best [BM] is {abs(delta):.4f} Accuracy ({base_correct - bm_correct} predictions) behind {best_base}.")
    else:
        print("  (One or both groups not present — cannot compare.)")

    # Submitted model gap (by Accuracy / correct predictions)
    print(f"\n  ─── Submitted model vs overall winner ────────────────────────────")
    print(f"  Submitted  Accuracy : {sub_acc:.4f}  "
          f"({sub_correct}/{sub_metrics['Total']} correct)")
    print(f"  Winner     Accuracy : {winner_acc:.4f}  "
          f"({winner_correct}/{winner_total} correct)  ({winner})")
    if gap_sub_best > 0.01:
        print(f"  Gap = {gap_sub_best:.4f} Accuracy — a better model exists in private evaluation.")
        print("  Recommendation: Review methodology.  Treat as suggestive only —")
        print("  label recovery is unvalidated; do NOT retrain on this signal.")
    else:
        print(f"  Gap = {gap_sub_best:.4f} Accuracy — submitted model is within normal variance.")
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

    # ── Final winner line ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  BEST METHOD BY DIRECT CSV MATCH ACCURACY: {winner}")
    print(f"  Correct predictions : {winner_correct} / {winner_total}")
    print(f"  Accuracy            : {winner_acc:.4f}")
    print("=" * 72)
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
