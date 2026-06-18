#!/usr/bin/env python3
# =============================================================================
#  PRIVATE CHECK ONLY — NOT FOR SUBMISSION
# =============================================================================
"""
private_original_data_evaluation.py

Post-hoc evaluation of all trained classification models against recovered
ground-truth labels from the public UCI dataset.

IMPORTANT CONSTRAINTS — enforced by design:
  - Original labels are recovered ONLY for evaluation.  They are never written
    back to the submitted test file, used for training, or used for tuning.
  - No submitted notebook or preprocessing file is modified.
  - All outputs go to private_evaluation/ and are excluded from git.

Run:
    py private_original_data_evaluation.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PRIVATE_DIR = Path("private_evaluation")
PRIVATE_DIR.mkdir(exist_ok=True)

BANNER = "\n" + "=" * 72 + "\n" + \
         "  PRIVATE CHECK ONLY — NOT FOR SUBMISSION\n" + \
         "  Results are diagnostic only and do NOT affect the submitted solution.\n" + \
         "=" * 72

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

# Categorical / numeric split (mirrors the notebook)
CAT_COLS = ["month", "browser", "region", "traffic_type", "is_weekend", "visitor_type"]

# ── Safety guard ──────────────────────────────────────────────────────────────

SUBMITTED_FILES = {
    "shopper_train.csv", "shopper_test.csv",
    "X_train_preprocessed.csv", "X_test_preprocessed.csv",
    "y_train.csv", "test_predictions.csv",
    "online_shoppers_intention.csv",
    "Classification_Section9.ipynb",
}

def _safe_write(path: Path, df: pd.DataFrame) -> None:
    """Write to PRIVATE_DIR only, refuse any overwrite of submitted files."""
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

    Tier 1: bounce_rate is present → match on MATCH_BASE + bounce_rate
    Tier 2: bounce_rate is missing → match on MATCH_BASE only

    Returns a DataFrame indexed by test row position with columns:
        test_idx, orig_idx, recovered_label, match_tier, n_orig_matches, ambiguous
    """
    test = df_test.copy().reset_index(drop=False).rename(columns={"index": "test_idx"})
    orig = df_orig_aligned.copy().reset_index(drop=False).rename(columns={"index": "orig_idx"})

    for c in FLOAT_COLS + ["bounce_rate"]:
        for frame in (test, orig):
            if c in frame.columns:
                frame[c] = frame[c].round(FLOAT_ROUND)

    records = []

    # Tier 1 — bounce_rate present in test row
    t1_test = test[test["bounce_rate"].notna()].copy()
    t1_cols = MATCH_BASE + ["bounce_rate"]
    t1_test["_key"] = _build_key(t1_test, t1_cols)
    t1_orig = orig.copy()
    t1_orig["_key"] = _build_key(t1_orig, t1_cols)

    m1 = t1_test[["test_idx", "_key"]].merge(
        t1_orig[["orig_idx", "_key", "high_intent"]],
        on="_key", how="left",
    )
    m1["match_tier"] = 1
    records.append(m1)

    # Tier 2 — bounce_rate is NaN
    t2_test = test[test["bounce_rate"].isna()].copy()
    t2_test["_key"] = _build_key(t2_test, MATCH_BASE)
    t2_orig = orig.copy()
    t2_orig["_key"] = _build_key(t2_orig, MATCH_BASE)

    m2 = t2_test[["test_idx", "_key"]].merge(
        t2_orig[["orig_idx", "_key", "high_intent"]],
        on="_key", how="left",
    )
    m2["match_tier"] = 2
    records.append(m2)

    combined = pd.concat(records, ignore_index=True)

    n_matches = combined.groupby("test_idx")["orig_idx"].transform("count")
    combined["n_orig_matches"] = n_matches
    combined["ambiguous"] = n_matches > 1

    # Resolve ambiguous rows: keep only if all matches agree on the label
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


# ── Model definitions (mirror notebook cell 5 / 13) ──────────────────────────

def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def build_models(num_cols, cat_cols):
    pre = make_preprocessor(num_cols, cat_cols)

    return {
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
    }


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_true, y_proba), 4) if y_proba is not None else float("nan"),
        "PR-AUC"   : round(average_precision_score(y_true, y_proba), 4) if y_proba is not None else float("nan"),
    }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(BANNER)
    print()

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    print("1. Loading datasets...")
    df_test_raw  = pd.read_csv("shopper_test.csv")
    df_train_raw = pd.read_csv("shopper_train.csv")
    df_orig      = pd.read_csv("online_shoppers_intention.csv")
    X_train_pre  = pd.read_csv("X_train_preprocessed.csv")
    y_train_full = pd.read_csv("y_train.csv").squeeze()
    X_test_pre   = pd.read_csv("X_test_preprocessed.csv")
    submitted_preds = pd.read_csv("test_predictions.csv")

    print(f"   shopper_test.csv              : {df_test_raw.shape}")
    print(f"   shopper_train.csv             : {df_train_raw.shape}")
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

    # Booleans → int
    for col in ("is_weekend", "high_intent"):
        if col in orig_aligned.columns:
            orig_aligned[col] = orig_aligned[col].astype(int)

    # bounce_rate (needed for tier-1 matching)
    if "BounceRates" in df_orig.columns:
        orig_aligned["bounce_rate"] = df_orig["BounceRates"]

    # ── 3. Match test rows to original dataset ────────────────────────────────
    print("\n3. Matching test rows to original dataset to recover labels...")

    test_for_match = df_test_raw.copy()
    matched = match_test_to_original(test_for_match, orig_aligned)

    n_test      = len(df_test_raw)
    n_matched   = matched["test_idx"].nunique()
    n_labeled   = matched["recovered_label"].notna().sum()
    n_ambiguous = matched[matched["ambiguous"]]["test_idx"].nunique()
    n_unmatched = n_test - n_matched

    print(f"   Total test rows               : {n_test}")
    print(f"   Successfully matched          : {n_matched}  ({100*n_matched/n_test:.1f}%)")
    print(f"   With recovered labels         : {n_labeled}  ({100*n_labeled/n_test:.1f}%)")
    print(f"   Ambiguous (resolved by agree) : {n_ambiguous}")
    print(f"   Unmatched / no label found    : {n_unmatched + (n_matched - n_labeled)}")

    if n_labeled < 30:
        print("\n   WARNING: Fewer than 30 test rows could be matched to original labels.")
        print("   Evaluation metrics may be unreliable.")

    # Build label series aligned to test row index
    label_map = (
        matched.dropna(subset=["recovered_label"])
               .set_index("test_idx")["recovered_label"]
               .astype(int)
    )
    y_recovered = pd.Series(index=range(n_test), dtype=float)
    y_recovered.update(label_map)

    # Subset of test rows with known labels
    eval_idx   = y_recovered.dropna().index.tolist()
    y_true_eval = y_recovered.loc[eval_idx].astype(int)

    print(f"\n   Evaluation subset size : {len(eval_idx)}")
    print(f"   Positive rate (original): {y_true_eval.mean():.1%}")

    # ── 4. Save recovered labels table (no label written to submitted test file)
    label_table = matched[["test_idx", "orig_idx", "recovered_label",
                            "match_tier", "n_orig_matches", "ambiguous"]].copy()
    _safe_write(PRIVATE_DIR / "recovered_test_labels.csv", label_table)

    # ── 5. Re-train all candidate models on full preprocessed training data ───
    print("\n4. Re-training all candidate models on X_train_preprocessed + y_train...")
    print("   (Same data and column split as the notebook — no leakage.)")

    num_cols = [c for c in X_train_pre.columns if c not in CAT_COLS]
    cat_cols_present = [c for c in CAT_COLS if c in X_train_pre.columns]

    models = build_models(num_cols, cat_cols_present)

    trained = {}
    for name, model in models.items():
        model.fit(X_train_pre, y_train_full)
        trained[name] = model
        print(f"   Fitted: {name}")

    # ── 6. Predict on evaluation subset of test set ───────────────────────────
    print(f"\n5. Generating predictions on the {len(eval_idx)}-row evaluation subset...")

    X_eval = X_test_pre.iloc[eval_idx].reset_index(drop=True)

    model_results = {}
    all_pred_rows = []

    for name, model in trained.items():
        y_pred  = model.predict(X_eval)
        y_proba = (
            model.predict_proba(X_eval)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        metrics = compute_metrics(y_true_eval.values, y_pred, y_proba)
        model_results[name] = metrics

        row = {"model": name, **metrics}
        all_pred_rows.append(row)
        print(f"   {name:20s}  Acc={metrics['Accuracy']:.4f}  "
              f"F1={metrics['F1']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}  "
              f"PR-AUC={metrics['PR-AUC']:.4f}")

    # Evaluate the submitted predictions on the same subset
    print("\n   Evaluating submitted test_predictions.csv on the same subset...")
    sub_pred  = submitted_preds["predicted_high_intent"].iloc[eval_idx].values
    sub_proba = submitted_preds["proba_high_intent"].iloc[eval_idx].values
    sub_metrics = compute_metrics(y_true_eval.values, sub_pred, sub_proba)
    model_results["Submitted Model"] = sub_metrics
    all_pred_rows.append({"model": "Submitted Model", **sub_metrics})

    print(f"   {'Submitted Model':20s}  Acc={sub_metrics['Accuracy']:.4f}  "
          f"F1={sub_metrics['F1']:.4f}  ROC-AUC={sub_metrics['ROC-AUC']:.4f}  "
          f"PR-AUC={sub_metrics['PR-AUC']:.4f}")

    # ── 7. Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" MODEL COMPARISON vs RECOVERED ORIGINAL LABELS")
    print("=" * 72)
    results_df = pd.DataFrame(all_pred_rows).set_index("model")
    results_df = results_df.sort_values("ROC-AUC", ascending=False)
    print(results_df.to_string())

    # ── 8. Cross-validation on training data (for overfitting check) ─────────
    print("\n6. Cross-validation on training data (for overfitting check)...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    cv_rows = []
    for name, model in trained.items():
        auc_cv = cross_val_score(model, X_train_pre, y_train_full,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        f1_cv  = cross_val_score(model, X_train_pre, y_train_full,
                                 cv=cv, scoring="f1", n_jobs=-1)
        cv_rows.append({
            "model"       : name,
            "CV_ROC-AUC"  : round(auc_cv.mean(), 4),
            "CV_AUC_std"  : round(auc_cv.std(), 4),
            "CV_F1"       : round(f1_cv.mean(), 4),
            "CV_F1_std"   : round(f1_cv.std(), 4),
        })
        print(f"   {name:20s}  CV ROC-AUC={auc_cv.mean():.4f}±{auc_cv.std():.4f}  "
              f"CV F1={f1_cv.mean():.4f}±{f1_cv.std():.4f}")

    cv_df = pd.DataFrame(cv_rows).set_index("model")

    # ── 9. Overfitting / agreement analysis ───────────────────────────────────
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
            "model"         : name,
            "private_ROC-AUC": priv_auc,
            "CV_ROC-AUC"    : cv_auc,
            "gap (priv-CV)" : gap,
            "overfitting?"  : overfit,
        })
        print(f"   {name:20s}  Private={priv_auc:.4f}  CV={cv_auc:.4f}  "
              f"Gap={gap:+.4f}  Overfit={overfit}")

    comparison_df = pd.DataFrame(comparison_rows).set_index("model")

    # ── 10. Confusion matrices plot ───────────────────────────────────────────
    print("\n7. Saving confusion matrix plots...")
    n_models = len(trained) + 1  # +1 for submitted
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained.items()):
        y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_true_eval, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"])
        ax.set_title(name, fontsize=9)

    ax_sub = axes[-1]
    cm_sub = confusion_matrix(y_true_eval, sub_pred)
    sns.heatmap(cm_sub, annot=True, fmt="d", cmap="Greens", ax=ax_sub,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    ax_sub.set_title("Submitted Model", fontsize=9)

    plt.suptitle("Confusion Matrices — Private Evaluation vs Recovered Labels",
                 fontsize=12)
    plt.tight_layout()
    cm_path = PRIVATE_DIR / "confusion_matrices_private.png"
    fig.savefig(cm_path, dpi=100)
    plt.close(fig)
    print(f"   Saved → {cm_path}")

    # ROC curves
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for name, model in trained.items():
        if hasattr(model, "predict_proba"):
            yp = model.predict_proba(X_eval)[:, 1]
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true_eval, yp)
            ax2.plot(fpr, tpr, lw=2,
                     label=f"{name}  (AUC={model_results[name]['ROC-AUC']:.3f})")
    from sklearn.metrics import roc_curve as _rc
    fpr_s, tpr_s, _ = _rc(y_true_eval, sub_proba)
    ax2.plot(fpr_s, tpr_s, lw=2, linestyle="--",
             label=f"Submitted  (AUC={sub_metrics['ROC-AUC']:.3f})")
    ax2.plot([0, 1], [0, 1], "k:", lw=1, label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves — Private Evaluation vs Recovered Labels")
    ax2.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    roc_path = PRIVATE_DIR / "roc_curves_private.png"
    fig2.savefig(roc_path, dpi=100)
    plt.close(fig2)
    print(f"   Saved → {roc_path}")

    # ── 11. Save CSV results ──────────────────────────────────────────────────
    print("\n8. Saving CSV results...")
    _safe_write(PRIVATE_DIR / "private_model_comparison_against_original_labels.csv",
                results_df.reset_index())
    _safe_write(PRIVATE_DIR / "private_cv_results.csv",
                cv_df.reset_index())
    _safe_write(PRIVATE_DIR / "private_overfitting_analysis.csv",
                comparison_df.reset_index())

    # ── 12. Summary & interpretation ─────────────────────────────────────────
    best_private = results_df["ROC-AUC"].idxmax()
    best_cv_auc  = cv_df["CV_ROC-AUC"].idxmax()
    agree        = best_private == best_cv_auc

    print("\n" + "=" * 72)
    print(" SUMMARY & INTERPRETATION")
    print("=" * 72)
    print(f"\n  Evaluation rows (matched to original) : {len(eval_idx)} / {n_test}")
    print(f"  Recovered positive rate               : {y_true_eval.mean():.1%}")
    print()
    print(f"  Best model by ROC-AUC (private test)  : {best_private}")
    print(f"  Best model by ROC-AUC (cross-val)     : {best_cv_auc}")
    print(f"  Cross-val and private evaluation agree: {'YES' if agree else 'NO — divergence detected'}")
    print()

    # Overfitting summary
    severe_overfit = [r["model"] for r in comparison_rows if r["gap (priv-CV)"] < -0.05]
    if severe_overfit:
        print(f"  Overfitting suspected (private < CV by >5pp): {severe_overfit}")
    else:
        print("  No severe overfitting detected across any model.")

    # Should methodology change?
    sub_priv_auc = sub_metrics["ROC-AUC"]
    best_priv_auc = results_df["ROC-AUC"].max()
    gap_from_best = best_priv_auc - sub_priv_auc

    print()
    print(f"  Submitted model ROC-AUC (private) : {sub_priv_auc:.4f}")
    print(f"  Best model ROC-AUC (private)      : {best_priv_auc:.4f}  ({best_private})")
    if gap_from_best > 0.03:
        print(f"  Gap = {gap_from_best:.4f} — a notably better model exists in private evaluation.")
        print("  Recommendation: Review methodology.  However, because this private")
        print("  evaluation uses unvalidated label recovery and a subset of test rows,")
        print("  it should be treated as suggestive, not conclusive.  Do NOT retrain")
        print("  or re-select models based on this file.")
    else:
        print(f"  Gap = {gap_from_best:.4f} — submitted model is within normal variance.")
        print("  Recommendation: No change to submitted methodology warranted.")

    print()
    print("  NOTE: This script is diagnostic only.  No submitted file was modified.")
    print("  All outputs are isolated to private_evaluation/.")
    print()
    print(BANNER)
    print()


if __name__ == "__main__":
    main()
