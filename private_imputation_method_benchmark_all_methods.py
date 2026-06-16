#!/usr/bin/env python3
# =============================================================================
# PRIVATE VALIDATION ONLY - DO NOT SUBMIT - DO NOT USE ORIGINAL DATASET FOR TRAINING
# =============================================================================
"""
private_imputation_method_benchmark.py

Extended benchmark comparing multiple imputation methods, including XGBoost, CatBoost, LightGBM, ExtraTrees, and HistGradientBoosting when installed, for each feature
that has missing values in shopper_train.csv (except bounce_rate, which is
intentionally dropped, and high_intent, which is the target).

Scoring uses online_shoppers_intention.csv ONLY as ground-truth labels AFTER
imputation.  The original dataset is NEVER used for training.

Output → private_validation/imputation_method_benchmark/

DO NOT COMMIT OUTPUT FILES — DO NOT PUSH — DO NOT USE FOR SUBMISSION
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import io
import traceback

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ─── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path("private_validation") / "imputation_method_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BANNER = (
    "\n" + "=" * 75 + "\n"
    " PRIVATE VALIDATION ONLY - DO NOT SUBMIT - DO NOT USE ORIGINAL DATASET FOR TRAINING \n"
    + "=" * 75
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

# Features to benchmark (bounce_rate is dropped; high_intent is target)
TARGET_FEATURES = ["month", "visitor_type", "product_duration", "page_value"]
CATEGORICAL_FEATURES = {"month", "visitor_type"}
NUMERIC_FEATURES     = {"product_duration", "page_value"}

# Columns with no missing values used as matching keys
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

# ─── Feature configuration ────────────────────────────────────────────────────

CAT_CFG = {
    "month": {
        "cond_col"      : "special_day_score",
        "cond_fn"       : lambda x: "special_day" if x > 0 else "regular_day",
        "knn_feats"     : [
            "num_product_pages", "exit_rate", "admin_duration", "info_duration",
            "special_day_score", "is_weekend", "operating_system", "browser",
            "region", "traffic_type", "engagement_score", "abandonment_score",
        ],
    },
    "visitor_type": {
        "cond_col"      : "traffic_type",
        "cond_fn"       : None,          # group by raw value
        "knn_feats"     : [
            "num_product_pages", "exit_rate", "admin_duration", "info_duration",
            "traffic_type", "special_day_score", "is_weekend",
            "operating_system", "browser", "region",
            "engagement_score", "abandonment_score",
        ],
    },
}

NUM_CFG = {
    "product_duration": {
        "zero_rule"   : "num_product_pages == 0",   # deterministic zero-fill rule
        "group_col"   : "num_product_pages",
        "group_bins"  : 5,                           # qcut bins for grouping
        "simple_feats": ["engagement_score"],
        "rich_feats"  : [
            "engagement_score", "num_product_pages", "admin_duration",
            "info_duration", "exit_rate", "special_day_score", "is_weekend",
            "operating_system", "browser", "region", "traffic_type",
            "abandonment_score",
        ],
    },
    "page_value": {
        "zero_rule"   : None,
        "group_col"   : "traffic_type",
        "group_bins"  : None,                        # categorical grouping
        "simple_feats": ["product_duration", "engagement_score", "exit_rate"],
        "rich_feats"  : [
            "product_duration", "engagement_score", "admin_duration",
            "info_duration", "exit_rate", "abandonment_score",
            "special_day_score", "is_weekend", "num_product_pages",
        ],
    },
}

# ─── Row matching (improved from validate_imputation_against_original.py) ─────

def _build_key(df: pd.DataFrame, cols: list) -> pd.Series:
    parts = []
    for c in cols:
        s = df[c].copy()
        if s.dtype in [float, np.float64, np.float32]:
            s = s.round(FLOAT_ROUND)
        parts.append(s.astype(str))
    return parts[0].str.cat(parts[1:], sep="|")


def match_to_original(df_assign: pd.DataFrame,
                      df_orig_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Match each shopper_train row to its counterpart in the original dataset.

    Tier 1 — bounce_rate is not NaN: match on MATCH_BASE + bounce_rate.
    Tier 2 — bounce_rate is NaN:     match on MATCH_BASE only.

    Ambiguous matches (1 train row → multiple original rows) are resolved:
      - kept if all candidates agree on every TARGET_FEATURE value
      - dropped otherwise

    Returns DataFrame with columns:
      assign_idx, orig_idx, match_tier, ambiguous, n_orig_matches,
      <TARGET_FEATURES from original>
    """
    assign = df_assign.copy().reset_index(drop=False).rename(columns={"index": "assign_idx"})
    orig   = df_orig_aligned.copy().reset_index(drop=False).rename(columns={"index": "orig_idx"})

    for c in FLOAT_COLS + ["bounce_rate"]:
        for frame in [assign, orig]:
            if c in frame.columns:
                frame[c] = frame[c].round(FLOAT_ROUND)

    orig_pull_cols = ["orig_idx", "_key"] + TARGET_FEATURES

    # Tier 1: bounce_rate observed
    t1_a = assign[assign["bounce_rate"].notna()].copy()
    t1_o = orig.copy()
    key1_cols = MATCH_BASE + ["bounce_rate"]
    t1_a["_key"] = _build_key(t1_a, key1_cols)
    t1_o["_key"] = _build_key(t1_o, key1_cols)
    m1 = t1_a[["assign_idx", "_key"]].merge(
        t1_o[[c for c in orig_pull_cols if c in t1_o.columns]],
        on="_key", how="left",
    )
    m1["match_tier"] = 1

    # Tier 2: bounce_rate missing
    t2_a = assign[assign["bounce_rate"].isna()].copy()
    t2_a["_key"] = _build_key(t2_a, MATCH_BASE)
    t2_o = orig.copy()
    t2_o["_key"] = _build_key(t2_o, MATCH_BASE)
    m2 = t2_a[["assign_idx", "_key"]].merge(
        t2_o[[c for c in orig_pull_cols if c in t2_o.columns]],
        on="_key", how="left",
    )
    m2["match_tier"] = 2

    combined = pd.concat([m1, m2], ignore_index=True)

    n_matches = combined.groupby("assign_idx")["orig_idx"].transform("count")
    combined["n_orig_matches"] = n_matches
    combined["ambiguous"]      = n_matches > 1

    def _dedup(group):
        if len(group) == 1:
            return group
        for feat in TARGET_FEATURES:
            if feat in group.columns and group[feat].nunique(dropna=False) > 1:
                return pd.DataFrame()  # conflicting originals → drop
        return group.head(1)

    combined = (
        combined
        .groupby("assign_idx", group_keys=False)
        .apply(_dedup)
        .reset_index(drop=True)
    )
    return combined


# ─── Imputation methods: categorical ─────────────────────────────────────────

def _cat_mode(df_train, feat):
    return df_train.loc[df_train[feat].notna(), feat].mode()[0]


def method_cat_A_GlobalMode(df_train, feat, missing_mask, cfg):
    """Most-frequent value across all observed rows."""
    mode_val = _cat_mode(df_train, feat)
    return pd.Series(mode_val, index=df_train.index[missing_mask])


def method_cat_B_ConditionalMode(df_train, feat, missing_mask, cfg):
    """Mode within a conditioning group (deterministic)."""
    cond_col = cfg["cond_col"]
    cond_fn  = cfg["cond_fn"]

    known = df_train[df_train[feat].notna()].copy()
    known["_grp"] = known[cond_col].apply(cond_fn) if cond_fn else known[cond_col]

    cond_modes = (
        known.groupby("_grp")[feat]
        .agg(lambda s: s.mode()[0])
    )
    global_mode = _cat_mode(df_train, feat)

    missing_sub = df_train[missing_mask].copy()
    missing_sub["_grp"] = missing_sub[cond_col].apply(cond_fn) if cond_fn else missing_sub[cond_col]

    preds = missing_sub["_grp"].map(cond_modes).fillna(global_mode)
    preds.index = df_train.index[missing_mask]
    return preds


def method_cat_C_KNN(df_train, feat, missing_mask, cfg):
    """KNN classifier with tuned k (5-fold CV)."""
    feats = cfg["knn_feats"]
    known   = df_train[df_train[feat].notna()].dropna(subset=feats)
    predict = df_train[missing_mask].dropna(subset=feats)

    preds = pd.Series(index=df_train.index[missing_mask], dtype=object)
    if len(predict) == 0 or len(known) < 5:
        preds[:] = _cat_mode(df_train, feat)
        return preds

    sc    = StandardScaler()
    X_tr  = sc.fit_transform(known[feats].values)
    X_pr  = sc.transform(predict[feats].values)
    y_tr  = known[feat].values

    best_k, best_acc = 5, 0.0
    max_k = min(21, len(known) // 5)
    for k in range(1, max(2, max_k)):
        cv_acc = cross_val_score(
            KNeighborsClassifier(n_neighbors=k, weights="distance"),
            X_tr, y_tr, cv=5, scoring="accuracy",
        ).mean()
        if cv_acc > best_acc:
            best_acc, best_k = cv_acc, k

    knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    knn.fit(X_tr, y_tr)
    preds[predict.index] = knn.predict(X_pr)
    preds = preds.fillna(_cat_mode(df_train, feat))
    return preds


def method_cat_D_RandomForest(df_train, feat, missing_mask, cfg):
    """Random-Forest classifier."""
    feats = cfg["knn_feats"]          # reuse same feature set for fair comparison
    known   = df_train[df_train[feat].notna()].dropna(subset=feats)
    predict = df_train[missing_mask].dropna(subset=feats)

    preds = pd.Series(index=df_train.index[missing_mask], dtype=object)
    if len(predict) == 0 or len(known) < 5:
        preds[:] = _cat_mode(df_train, feat)
        return preds

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=3,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(known[feats].values, known[feat].values)
    preds[predict.index] = rf.predict(predict[feats].values)
    preds = preds.fillna(_cat_mode(df_train, feat))
    return preds



def method_cat_E_XGBoost(df_train, feat, missing_mask, cfg):
    """XGBoost classifier for categorical imputation.

    Uses the same non-target feature set as KNN/RF for a fair comparison.
    Labels are encoded internally because XGBoost expects numeric class labels.
    """
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    feats = cfg["knn_feats"]
    known   = df_train[df_train[feat].notna()].dropna(subset=feats)
    predict = df_train[missing_mask].dropna(subset=feats)

    preds = pd.Series(index=df_train.index[missing_mask], dtype=object)
    if len(predict) == 0 or len(known) < 5:
        preds[:] = _cat_mode(df_train, feat)
        return preds

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(known[feat].astype(str))

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb.fit(known[feats].values, y_encoded)

    pred_encoded = xgb.predict(predict[feats].values)
    preds[predict.index] = label_encoder.inverse_transform(pred_encoded.astype(int))
    preds = preds.fillna(_cat_mode(df_train, feat))
    return preds




def method_cat_F_CatBoost(df_train, feat, missing_mask, cfg):
    """CatBoost classifier for categorical imputation."""
    if not HAS_CATBOOST:
        raise ImportError("catboost is not installed. Run: pip install catboost")

    feats = cfg["knn_feats"]
    known   = df_train[df_train[feat].notna()].dropna(subset=feats)
    predict = df_train[missing_mask].dropna(subset=feats)

    preds = pd.Series(index=df_train.index[missing_mask], dtype=object)
    if len(predict) == 0 or len(known) < 5:
        preds[:] = _cat_mode(df_train, feat)
        return preds

    model = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(known[feats], known[feat].astype(str))
    preds[predict.index] = model.predict(predict[feats]).reshape(-1)
    preds = preds.fillna(_cat_mode(df_train, feat))
    return preds


def method_cat_G_LightGBM(df_train, feat, missing_mask, cfg):
    """LightGBM classifier for categorical imputation."""
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

    feats = cfg["knn_feats"]
    known   = df_train[df_train[feat].notna()].dropna(subset=feats)
    predict = df_train[missing_mask].dropna(subset=feats)

    preds = pd.Series(index=df_train.index[missing_mask], dtype=object)
    if len(predict) == 0 or len(known) < 5:
        preds[:] = _cat_mode(df_train, feat)
        return preds

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(known[feat].astype(str))

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(known[feats].values, y_encoded)
    pred_encoded = model.predict(predict[feats].values)
    preds[predict.index] = label_encoder.inverse_transform(pred_encoded.astype(int))
    preds = preds.fillna(_cat_mode(df_train, feat))
    return preds


def method_cat_H_ExtraTrees(df_train, feat, missing_mask, cfg):
    """ExtraTrees classifier for categorical imputation."""
    feats = cfg["knn_feats"]
    known   = df_train[df_train[feat].notna()].dropna(subset=feats)
    predict = df_train[missing_mask].dropna(subset=feats)

    preds = pd.Series(index=df_train.index[missing_mask], dtype=object)
    if len(predict) == 0 or len(known) < 5:
        preds[:] = _cat_mode(df_train, feat)
        return preds

    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(known[feats].values, known[feat].values)
    preds[predict.index] = model.predict(predict[feats].values)
    preds = preds.fillna(_cat_mode(df_train, feat))
    return preds


# ─── Imputation methods: numeric ──────────────────────────────────────────────

def _apply_zero_rule(df_train, feat, missing_mask, cfg):
    """
    Returns (preds_series, remaining_mask).
    preds_series has 0.0 for rows satisfying the zero rule.
    remaining_mask is missing_mask minus those rows.
    """
    preds = pd.Series(np.nan, index=df_train.index[missing_mask], dtype=float)
    remaining_mask = missing_mask.copy()

    zero_rule = cfg.get("zero_rule")
    if zero_rule and "num_product_pages" in df_train.columns:
        zero_idx = df_train.index[missing_mask & (df_train.eval(zero_rule))]
        preds[zero_idx] = 0.0
        remaining_mask = missing_mask & (~df_train.eval(zero_rule))

    return preds, remaining_mask


def method_num_A_Median(df_train, feat, missing_mask, cfg):
    """Global median imputation."""
    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)
    median_val = df_train.loc[df_train[feat].notna(), feat].median()
    preds[df_train.index[remaining]] = median_val
    return preds


def method_num_B_ConditionalMedian(df_train, feat, missing_mask, cfg):
    """Median within a conditioning group."""
    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)

    if remaining.sum() == 0:
        return preds

    group_col  = cfg["group_col"]
    group_bins = cfg.get("group_bins")
    known      = df_train[df_train[feat].notna()].copy()
    pred_sub   = df_train[remaining].copy()
    global_med = known[feat].median()

    if group_bins is not None:
        try:
            bins = pd.qcut(df_train[group_col], q=group_bins, duplicates="drop", labels=False)
        except ValueError:
            bins = pd.cut(df_train[group_col], bins=group_bins, labels=False, include_lowest=True)
        known["_grp"]    = bins[known.index]
        pred_sub["_grp"] = bins[pred_sub.index]
    else:
        known["_grp"]    = known[group_col]
        pred_sub["_grp"] = pred_sub[group_col]

    cond_meds = known.groupby("_grp")[feat].median()
    mapped    = pred_sub["_grp"].map(cond_meds).fillna(global_med)
    preds[pred_sub.index] = mapped.values
    return preds


def method_num_C_LinearRegression(df_train, feat, missing_mask, cfg):
    """Linear regression using a small, interpretable feature set."""
    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)

    if remaining.sum() == 0:
        return preds

    simple_feats = cfg["simple_feats"]
    global_med   = df_train.loc[df_train[feat].notna(), feat].median()

    known   = df_train[df_train[feat].notna()].dropna(subset=simple_feats)
    predict = df_train[remaining].dropna(subset=simple_feats)

    if len(known) < 5 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    lr = LinearRegression()
    lr.fit(known[simple_feats].values, known[feat].values)
    preds[predict.index] = np.clip(lr.predict(predict[simple_feats].values), 0, None)
    preds = preds.fillna(global_med)
    return preds


def method_num_D_Ridge(df_train, feat, missing_mask, cfg):
    """Ridge regression on full feature set with log1p target transform."""
    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)

    if remaining.sum() == 0:
        return preds

    rich_feats = cfg["rich_feats"]
    global_med = df_train.loc[df_train[feat].notna(), feat].median()

    known   = df_train[df_train[feat].notna()].dropna(subset=rich_feats)
    predict = df_train[remaining].dropna(subset=rich_feats)

    if len(known) < 5 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    sc = StandardScaler()
    X_tr = sc.fit_transform(known[rich_feats].values)
    X_pr = sc.transform(predict[rich_feats].values)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, np.log1p(known[feat].values))
    preds[predict.index] = np.clip(np.expm1(ridge.predict(X_pr)), 0, None)
    preds = preds.fillna(global_med)
    return preds


def method_num_E_RandomForest(df_train, feat, missing_mask, cfg):
    """Random-Forest regressor with log1p target transform."""
    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)

    if remaining.sum() == 0:
        return preds

    rich_feats = cfg["rich_feats"]
    global_med = df_train.loc[df_train[feat].notna(), feat].median()

    known   = df_train[df_train[feat].notna()].dropna(subset=rich_feats)
    predict = df_train[remaining].dropna(subset=rich_feats)

    if len(known) < 5 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(known[rich_feats].values, np.log1p(known[feat].values))
    preds[predict.index] = np.clip(np.expm1(rf.predict(predict[rich_feats].values)), 0, None)
    preds = preds.fillna(global_med)
    return preds


def method_num_F_KNN(df_train, feat, missing_mask, cfg):
    """KNN regressor (k=7, distance-weighted)."""
    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)

    if remaining.sum() == 0:
        return preds

    rich_feats = cfg["rich_feats"]
    global_med = df_train.loc[df_train[feat].notna(), feat].median()

    known   = df_train[df_train[feat].notna()].dropna(subset=rich_feats)
    predict = df_train[remaining].dropna(subset=rich_feats)

    if len(known) < 10 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    sc = StandardScaler()
    X_tr = sc.fit_transform(known[rich_feats].values)
    X_pr = sc.transform(predict[rich_feats].values)

    knn = KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1)
    knn.fit(X_tr, known[feat].values)
    preds[predict.index] = np.clip(knn.predict(X_pr), 0, None)
    preds = preds.fillna(global_med)
    return preds



def method_num_G_XGBoost(df_train, feat, missing_mask, cfg):
    """XGBoost regressor with log1p target transform."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)

    if remaining.sum() == 0:
        return preds

    rich_feats = cfg["rich_feats"]
    global_med = df_train.loc[df_train[feat].notna(), feat].median()

    known   = df_train[df_train[feat].notna()].dropna(subset=rich_feats)
    predict = df_train[remaining].dropna(subset=rich_feats)

    if len(known) < 10 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb.fit(known[rich_feats].values, np.log1p(known[feat].values))
    preds[predict.index] = np.clip(
        np.expm1(xgb.predict(predict[rich_feats].values)),
        0,
        None,
    )
    preds = preds.fillna(global_med)
    return preds




def method_num_H_CatBoost(df_train, feat, missing_mask, cfg):
    """CatBoost regressor with log1p target transform."""
    if not HAS_CATBOOST:
        raise ImportError("catboost is not installed. Run: pip install catboost")

    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)
    if remaining.sum() == 0:
        return preds

    rich_feats = cfg["rich_feats"]
    global_med = df_train.loc[df_train[feat].notna(), feat].median()
    known   = df_train[df_train[feat].notna()].dropna(subset=rich_feats)
    predict = df_train[remaining].dropna(subset=rich_feats)

    if len(known) < 10 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    model = CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(known[rich_feats], np.log1p(known[feat].values))
    preds[predict.index] = np.clip(
        np.expm1(model.predict(predict[rich_feats])),
        0,
        None,
    )
    preds = preds.fillna(global_med)
    return preds


def method_num_I_LightGBM(df_train, feat, missing_mask, cfg):
    """LightGBM regressor with log1p target transform."""
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)
    if remaining.sum() == 0:
        return preds

    rich_feats = cfg["rich_feats"]
    global_med = df_train.loc[df_train[feat].notna(), feat].median()
    known   = df_train[df_train[feat].notna()].dropna(subset=rich_feats)
    predict = df_train[remaining].dropna(subset=rich_feats)

    if len(known) < 10 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    model = LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(known[rich_feats].values, np.log1p(known[feat].values))
    preds[predict.index] = np.clip(
        np.expm1(model.predict(predict[rich_feats].values)),
        0,
        None,
    )
    preds = preds.fillna(global_med)
    return preds


def method_num_J_ExtraTrees(df_train, feat, missing_mask, cfg):
    """ExtraTrees regressor with log1p target transform."""
    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)
    if remaining.sum() == 0:
        return preds

    rich_feats = cfg["rich_feats"]
    global_med = df_train.loc[df_train[feat].notna(), feat].median()
    known   = df_train[df_train[feat].notna()].dropna(subset=rich_feats)
    predict = df_train[remaining].dropna(subset=rich_feats)

    if len(known) < 10 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(known[rich_feats].values, np.log1p(known[feat].values))
    preds[predict.index] = np.clip(
        np.expm1(model.predict(predict[rich_feats].values)),
        0,
        None,
    )
    preds = preds.fillna(global_med)
    return preds


def method_num_K_HistGradientBoosting(df_train, feat, missing_mask, cfg):
    """Scikit-learn HistGradientBoosting regressor with log1p target transform."""
    preds, remaining = _apply_zero_rule(df_train, feat, missing_mask, cfg)
    if remaining.sum() == 0:
        return preds

    rich_feats = cfg["rich_feats"]
    global_med = df_train.loc[df_train[feat].notna(), feat].median()
    known   = df_train[df_train[feat].notna()].dropna(subset=rich_feats)
    predict = df_train[remaining].dropna(subset=rich_feats)

    if len(known) < 10 or len(predict) == 0:
        preds[df_train.index[remaining]] = global_med
        return preds

    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_leaf_nodes=31,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=RANDOM_SEED,
    )
    model.fit(known[rich_feats].values, np.log1p(known[feat].values))
    preds[predict.index] = np.clip(
        np.expm1(model.predict(predict[rich_feats].values)),
        0,
        None,
    )
    preds = preds.fillna(global_med)
    return preds


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_categorical(y_true: pd.Series, y_pred: pd.Series, feat: str) -> dict:
    valid = y_true.notna() & y_pred.notna()
    yt = y_true[valid].astype(str)
    yp = y_pred[valid].astype(str)
    n  = len(yt)

    if n == 0:
        return {"feature": feat, "feature_type": "categorical", "n_evaluated": 0}

    acc         = float(accuracy_score(yt, yp))
    f1_macro    = float(f1_score(yt, yp, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(yt, yp, average="weighted", zero_division=0))

    classes = sorted(yt.unique())
    per_class = {
        cls: round(float((yp[yt == cls] == yt[yt == cls]).mean()), 4)
        for cls in classes if (yt == cls).sum() > 0
    }

    cm = pd.crosstab(yt, yp, rownames=["true"], colnames=["imputed"])

    err_df = pd.DataFrame({"true": yt.values, "imputed": yp.values})
    err_df = err_df[err_df["true"] != err_df["imputed"]]
    top_errors = (
        err_df.groupby(["true", "imputed"]).size()
        .sort_values(ascending=False).head(5)
        .reset_index(name="count").to_dict("records")
    ) if len(err_df) > 0 else []

    return {
        "feature"          : feat,
        "feature_type"     : "categorical",
        "n_evaluated"      : n,
        "accuracy"         : round(acc, 4),
        "f1_macro"         : round(f1_macro, 4),
        "f1_weighted"      : round(f1_weighted, 4),
        "per_class_accuracy": per_class,
        "confusion_matrix" : cm,
        "top_errors"       : top_errors,
    }


def score_numeric(y_true: pd.Series, y_pred: pd.Series, feat: str) -> dict:
    valid = y_true.notna() & y_pred.notna()
    yt = y_true[valid].values.astype(float)
    yp = y_pred[valid].values.astype(float)
    n  = len(yt)

    if n == 0:
        return {"feature": feat, "feature_type": "numeric", "n_evaluated": 0}

    mae   = float(mean_absolute_error(yt, yp))
    rmse  = float(np.sqrt(mean_squared_error(yt, yp)))
    medae = float(np.median(np.abs(yt - yp)))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-10
    r2    = float(1 - ss_res / ss_tot)

    if n >= 3:
        pr, pp = stats.pearsonr(yt, yp)
        sr, sp = stats.spearmanr(yt, yp)
    else:
        pr = pp = sr = sp = np.nan

    exact = float((np.abs(yt - yp) < 1e-6).mean() * 100)

    return {
        "feature"        : feat,
        "feature_type"   : "numeric",
        "n_evaluated"    : n,
        "mae"            : round(float(mae),   4),
        "rmse"           : round(float(rmse),  4),
        "median_ae"      : round(float(medae), 4),
        "r2"             : round(float(r2),    4),
        "pearson_r"      : round(float(pr), 4) if not np.isnan(pr) else None,
        "pearson_p"      : round(float(pp), 6) if not np.isnan(pp) else None,
        "spearman_r"     : round(float(sr), 4) if not np.isnan(sr) else None,
        "spearman_p"     : round(float(sp), 6) if not np.isnan(sp) else None,
        "exact_match_%"  : round(exact, 2),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _safe_fname(feat: str, method: str, prefix: str, ext: str = ".png") -> str:
    safe = method.replace(" ", "_").replace("/", "_")
    return f"{prefix}_{feat}_{safe}{ext}"


def plot_numeric_scatter(y_true, y_pred, feat, method_name):
    valid = y_true.notna() & y_pred.notna()
    yt = y_true[valid].values.astype(float)
    yp = y_pred[valid].values.astype(float)
    if len(yt) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(yt, yp, alpha=0.4, s=8, color="steelblue")
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect")
    axes[0].set_xlabel("True value")
    axes[0].set_ylabel("Imputed")
    axes[0].set_title(f"Imputed vs True  (n={len(yt)})")
    axes[0].legend(fontsize=8)

    axes[1].hist(yt, bins=40, alpha=0.6, label="True",    color="steelblue")
    axes[1].hist(yp, bins=40, alpha=0.6, label="Imputed", color="salmon")
    axes[1].set_title("Distributions")
    axes[1].legend(fontsize=8)

    res = yp - yt
    axes[2].hist(res, bins=40, color="mediumseagreen", alpha=0.7)
    axes[2].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[2].set_title("Residuals (imputed − true)")
    axes[2].set_xlabel("Error")

    fig.suptitle(f"{feat}  |  {method_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / _safe_fname(feat, method_name, "scatter"), dpi=80)
    plt.close(fig)


def plot_confusion_matrix(cm, feat, method_name, acc):
    if cm is None or cm.empty:
        return

    h = max(4, len(cm.index) * 0.7)
    w = max(5, len(cm.columns) * 0.9)
    fig, ax = plt.subplots(figsize=(w, h))

    if HAS_SNS:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    else:
        ax.imshow(cm.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cm.index, fontsize=8)
        for i in range(len(cm.index)):
            for j in range(len(cm.columns)):
                ax.text(j, i, str(cm.values[i, j]), ha="center", va="center", fontsize=8)

    ax.set_title(f"{feat}  |  {method_name}  |  acc={acc:.3f}\n(rows=true, cols=imputed)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / _safe_fname(feat, method_name, "cm"), dpi=80)
    plt.close(fig)


# ─── Ranking helpers ──────────────────────────────────────────────────────────

def _rank_feature(sub: pd.DataFrame) -> pd.DataFrame:
    feat_type = sub["feature_type"].iloc[0]
    if feat_type == "categorical":
        sub = sub.sort_values(["accuracy", "f1_macro"], ascending=[False, False])
    else:
        sub = sub.sort_values(["mae", "rmse"], ascending=[True, True])
    sub = sub.reset_index(drop=True)
    sub["rank_within_feature"] = range(1, len(sub) + 1)
    return sub


def _interpret(row) -> str:
    feat_type = row.get("feature_type", "")
    rank      = row.get("rank_within_feature", "?")
    if feat_type == "categorical":
        acc = row.get("accuracy") or 0
        lbl = ("excellent" if acc >= 0.80 else "good" if acc >= 0.65
               else "moderate" if acc >= 0.50 else "poor")
        return f"Rank #{rank}: {lbl} (acc={acc:.3f}, f1_macro={row.get('f1_macro') or 0:.3f})"
    else:
        mae = row.get("mae") or float("inf")
        pr  = row.get("pearson_r") or 0
        lbl = ("excellent" if pr >= 0.85 else "good" if pr >= 0.65
               else "moderate" if pr >= 0.40 else "poor")
        return f"Rank #{rank}: {lbl} (MAE={mae:.4f}, pearson_r={pr:.3f})"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(BANNER)
    print()

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    print("1. Loading datasets...")
    df_train = pd.read_csv("shopper_train.csv")
    df_test  = pd.read_csv("shopper_test.csv")
    df_orig  = pd.read_csv("online_shoppers_intention.csv")

    print(f"   shopper_train.csv             : {df_train.shape}")
    print(f"   shopper_test.csv              : {df_test.shape}")
    print(f"   online_shoppers_intention.csv : {df_orig.shape}")

    # ── 2. Missing-value audit ────────────────────────────────────────────────
    print("\n2. Missing-value audit in shopper_train.csv:")
    miss = df_train.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    for col, cnt in miss.items():
        pct = cnt / len(df_train) * 100
        note = ""
        if col == "bounce_rate":
            note = " ← DROPPED (0.98 corr with abandonment_score); excluded from benchmark"
        elif col == "high_intent":
            note = " ← TARGET; excluded from benchmark"
        elif col in TARGET_FEATURES:
            note = " ← WILL BENCHMARK"
        print(f"   {col:20s}: {cnt:4d} ({pct:5.1f}%){note}")

    # ── 3. Align original dataset columns ────────────────────────────────────
    print("\n3. Aligning original dataset column names...")
    orig_aligned = pd.DataFrame(index=df_orig.index)
    for orig_col, assign_col in REVERSE_MAP.items():
        if orig_col in df_orig.columns:
            orig_aligned[assign_col] = df_orig[orig_col]

    for col in ["is_weekend", "high_intent"]:
        if col in orig_aligned.columns:
            orig_aligned[col] = orig_aligned[col].astype(int)

    if "BounceRates" in df_orig.columns:
        orig_aligned["bounce_rate"] = df_orig["BounceRates"]

    # ── 4. Match rows ─────────────────────────────────────────────────────────
    print("\n4. Matching shopper_train rows to online_shoppers_intention.csv...")
    matched = match_to_original(df_train, orig_aligned)

    n_total     = len(df_train)
    n_matched   = matched["assign_idx"].nunique()
    n_ambiguous = matched[matched["ambiguous"]]["assign_idx"].nunique()
    n_unmatched = n_total - n_matched
    match_pct   = 100 * n_matched / n_total

    print(f"   Total rows           : {n_total}")
    print(f"   Uniquely matched     : {n_matched}  ({match_pct:.1f}%)")
    print(f"   Ambiguous (resolved) : {n_ambiguous}")
    print(f"   Unmatched            : {n_unmatched}  ({100*n_unmatched/n_total:.1f}%)")

    if n_matched < n_total * 0.5:
        print("   WARNING: Match rate below 50%! Benchmark may not be reliable.")
        print("   Continuing on matched rows only.")

    # True-value lookup: assign_idx → original feature values
    lookup = matched.set_index("assign_idx")[TARGET_FEATURES]

    # ── 5. Define method registries ───────────────────────────────────────────
    cat_methods = {
        "A_GlobalMode"      : method_cat_A_GlobalMode,
        "B_ConditionalMode" : method_cat_B_ConditionalMode,
        "C_KNN"             : method_cat_C_KNN,
        "D_RandomForest"    : method_cat_D_RandomForest,
        "H_ExtraTrees"      : method_cat_H_ExtraTrees,
    }
    num_methods = {
        "A_Median"                  : method_num_A_Median,
        "B_ConditionalMedian"       : method_num_B_ConditionalMedian,
        "C_LinearRegression"        : method_num_C_LinearRegression,
        "D_Ridge"                   : method_num_D_Ridge,
        "E_RandomForest"            : method_num_E_RandomForest,
        "F_KNN"                     : method_num_F_KNN,
        "J_ExtraTrees"              : method_num_J_ExtraTrees,
        "K_HistGradientBoosting"    : method_num_K_HistGradientBoosting,
    }

    if HAS_XGBOOST:
        cat_methods["E_XGBoost"] = method_cat_E_XGBoost
        num_methods["G_XGBoost"] = method_num_G_XGBoost
        print("   XGBoost detected: adding XGBoost imputation methods.")
    else:
        print("   XGBoost is not installed: skipping XGBoost methods.")
        print("   To include them, run: pip install xgboost")

    if HAS_CATBOOST:
        cat_methods["F_CatBoost"] = method_cat_F_CatBoost
        num_methods["H_CatBoost"] = method_num_H_CatBoost
        print("   CatBoost detected: adding CatBoost imputation methods.")
    else:
        print("   CatBoost is not installed: skipping CatBoost methods.")
        print("   To include them, run: pip install catboost")

    if HAS_LIGHTGBM:
        cat_methods["G_LightGBM"] = method_cat_G_LightGBM
        num_methods["I_LightGBM"] = method_num_I_LightGBM
        print("   LightGBM detected: adding LightGBM imputation methods.")
    else:
        print("   LightGBM is not installed: skipping LightGBM methods.")
        print("   To include them, run: pip install lightgbm")

    print("   ExtraTrees and HistGradientBoosting are available from scikit-learn and will be included.")

    all_rows  = []           # flat list of result dicts
    all_metrics = {}         # feat → method → full metric dict (for summary file)

    # ── 6. Run benchmark ─────────────────────────────────────────────────────
    print("\n5. Running imputation benchmark...")

    for feat in TARGET_FEATURES:
        print(f"\n{'═'*65}")
        print(f"  FEATURE: {feat}")
        print(f"{'═'*65}")

        missing_mask = df_train[feat].isna()
        n_missing    = missing_mask.sum()

        # Evaluable = missing AND matched to a ground-truth row
        missing_idx  = df_train.index[missing_mask]
        eval_idx     = [i for i in missing_idx if i in lookup.index]
        n_eval       = len(eval_idx)

        print(f"  Missing in train: {n_missing}  |  Matched to original (evaluable): {n_eval}")

        if n_eval == 0:
            print("  No evaluable rows — skipping feature.")
            continue

        y_true = lookup.loc[eval_idx, feat]

        is_cat   = feat in CATEGORICAL_FEATURES
        methods  = cat_methods if is_cat else num_methods
        cfg      = CAT_CFG[feat] if is_cat else NUM_CFG[feat]
        feat_rows = []

        for method_name, method_fn in methods.items():
            print(f"\n  ▸ {method_name}", end="  ", flush=True)
            try:
                preds_all  = method_fn(df_train, feat, missing_mask, cfg)
                preds_eval = preds_all.loc[eval_idx]

                if is_cat:
                    m = score_categorical(y_true, preds_eval, feat)
                    print(f"acc={m.get('accuracy','N/A')}  "
                          f"f1_macro={m.get('f1_macro','N/A')}  "
                          f"f1_weighted={m.get('f1_weighted','N/A')}")
                    plot_confusion_matrix(
                        m.get("confusion_matrix"), feat, method_name,
                        m.get("accuracy", 0),
                    )
                    row = {
                        "feature"     : feat,
                        "method"      : method_name,
                        "feature_type": "categorical",
                        "n_evaluated" : m["n_evaluated"],
                        "accuracy"    : m.get("accuracy"),
                        "f1_macro"    : m.get("f1_macro"),
                        "f1_weighted" : m.get("f1_weighted"),
                        # numeric cols will be NaN
                        "mae": None, "rmse": None, "median_ae": None,
                        "r2": None, "pearson_r": None, "spearman_r": None,
                        "exact_match_%": None,
                    }
                else:
                    m = score_numeric(y_true, preds_eval, feat)
                    print(f"MAE={m.get('mae','N/A')}  RMSE={m.get('rmse','N/A')}  "
                          f"pearson_r={m.get('pearson_r','N/A')}  R²={m.get('r2','N/A')}")
                    plot_numeric_scatter(y_true, preds_eval, feat, method_name)
                    row = {
                        "feature"     : feat,
                        "method"      : method_name,
                        "feature_type": "numeric",
                        "n_evaluated" : m["n_evaluated"],
                        # categorical cols will be None
                        "accuracy": None, "f1_macro": None, "f1_weighted": None,
                        "mae"         : m.get("mae"),
                        "rmse"        : m.get("rmse"),
                        "median_ae"   : m.get("median_ae"),
                        "r2"          : m.get("r2"),
                        "pearson_r"   : m.get("pearson_r"),
                        "spearman_r"  : m.get("spearman_r"),
                        "exact_match_%": m.get("exact_match_%"),
                    }

                feat_rows.append(row)
                all_rows.append(row)
                all_metrics.setdefault(feat, {})[method_name] = m

            except Exception as exc:
                print(f"\n    ERROR: {exc}")
                traceback.print_exc()

    # ── 7. Rank results ───────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo results produced. Exiting.")
        return

    print(f"\n\n{'═'*65}")
    print("6. Ranking results within each feature...")

    results_df = pd.DataFrame(all_rows)
    ranked_parts = []
    for feat in TARGET_FEATURES:
        sub = results_df[results_df["feature"] == feat].copy()
        if sub.empty:
            continue
        sub = _rank_feature(sub)
        sub["short_interpretation"] = sub.apply(_interpret, axis=1)

        is_cat = sub["feature_type"].iloc[0] == "categorical"
        sub["main_metric"]      = sub["accuracy"] if is_cat else sub["mae"]
        sub["main_metric_name"] = "accuracy"       if is_cat else "mae"
        ranked_parts.append(sub)

    final_df = pd.concat(ranked_parts, ignore_index=True)

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    print(f"\n7. Saving outputs to {OUTPUT_DIR}/...")

    CSV_COLS = [
        "feature", "method", "feature_type", "n_evaluated",
        "main_metric", "main_metric_name", "rank_within_feature",
        "short_interpretation",
        "accuracy", "f1_macro", "f1_weighted",
        "mae", "rmse", "median_ae", "r2", "pearson_r", "spearman_r", "exact_match_%",
    ]
    out_cols = [c for c in CSV_COLS if c in final_df.columns]

    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    final_df[out_cols].to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Summary text
    txt_path = OUTPUT_DIR / "benchmark_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(BANNER.strip() + "\n\n")
        f.write(f"Train rows: {n_total}  |  Matched: {n_matched} ({match_pct:.1f}%)\n")
        f.write(f"Unmatched: {n_unmatched} ({100*n_unmatched/n_total:.1f}%)\n\n")

        for feat in TARGET_FEATURES:
            sub = final_df[final_df["feature"] == feat]
            if sub.empty:
                continue
            is_cat = sub["feature_type"].iloc[0] == "categorical"

            f.write(f"\n{'═'*65}\n")
            f.write(f"FEATURE: {feat}  ({sub['feature_type'].iloc[0]})\n")
            f.write(f"{'─'*65}\n")

            show_cols = (
                ["method", "rank_within_feature", "n_evaluated",
                 "accuracy", "f1_macro", "f1_weighted", "short_interpretation"]
                if is_cat else
                ["method", "rank_within_feature", "n_evaluated",
                 "mae", "rmse", "median_ae", "r2", "pearson_r", "spearman_r",
                 "short_interpretation"]
            )
            show_cols = [c for c in show_cols if c in sub.columns]
            f.write(sub[show_cols].to_string(index=False))
            f.write("\n\n")

            best     = sub[sub["rank_within_feature"] == 1].iloc[0]
            baseline = sub[sub["method"].str.startswith("A_")]
            if not baseline.empty:
                bl = baseline.iloc[0]
                f.write(f"Best method:     {best['method']}\n")
                f.write(f"  {best['short_interpretation']}\n")
                if best["method"] != bl["method"]:
                    if is_cat:
                        diff = (best["accuracy"] or 0) - (bl["accuracy"] or 0)
                        f.write(f"Baseline ({bl['method']}):  acc={bl.get('accuracy','N/A'):.4f}\n")
                        f.write(f"Improvement over baseline: {diff:+.4f} accuracy\n")
                    else:
                        diff = (bl["mae"] or 0) - (best["mae"] or 0)
                        f.write(f"Baseline ({bl['method']}):  MAE={bl.get('mae','N/A'):.4f}\n")
                        f.write(f"MAE improvement over baseline: {diff:.4f}\n")
                else:
                    f.write("Baseline IS the best method.\n")

            # Per-method details from all_metrics
            f.write(f"\n{'─'*65}\nDetailed metrics per method:\n")
            for method_name, m in all_metrics.get(feat, {}).items():
                f.write(f"\n  {method_name}:\n")
                for k, v in m.items():
                    if k not in ("feature", "feature_type", "confusion_matrix"):
                        f.write(f"    {k}: {v}\n")
                if "confusion_matrix" in m and m["confusion_matrix"] is not None:
                    f.write("    confusion_matrix:\n")
                    cm_str = m["confusion_matrix"].to_string()
                    for line in cm_str.split("\n"):
                        f.write(f"      {line}\n")

        f.write(f"\n\n{BANNER.strip()}\n")

    print(f"   Saved: {txt_path}")

    plots = sorted(OUTPUT_DIR.glob("*.png"))
    if plots:
        print(f"   Saved {len(plots)} plot(s):")
        for p in plots:
            print(f"     {p.name}")

    # ── 9. Terminal summary ───────────────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print(" BENCHMARK TERMINAL SUMMARY")
    print(f"{'═'*65}")

    for feat in TARGET_FEATURES:
        sub = final_df[final_df["feature"] == feat]
        if sub.empty:
            continue
        is_cat = sub["feature_type"].iloc[0] == "categorical"

        best     = sub[sub["rank_within_feature"] == 1].iloc[0]
        baseline = sub[sub["method"].str.startswith("A_")]
        bl       = baseline.iloc[0] if not baseline.empty else None

        print(f"\n  {feat}  ({sub['feature_type'].iloc[0]}, n_eval={sub['n_evaluated'].iloc[0]})")
        print(f"    Best method  : {best['method']}")

        if is_cat:
            print(f"    Best accuracy: {best.get('accuracy','N/A'):.4f}   "
                  f"f1_macro: {best.get('f1_macro','N/A'):.4f}")
        else:
            print(f"    Best MAE     : {best.get('mae','N/A'):.4f}   "
                  f"Pearson r: {best.get('pearson_r','N/A'):.4f}   "
                  f"R²: {best.get('r2','N/A'):.4f}")

        if bl is not None:
            if best["method"] == bl["method"]:
                print("    NOTE: Baseline IS the best — more complex methods did NOT improve")
            else:
                if is_cat:
                    diff = (best.get("accuracy") or 0) - (bl.get("accuracy") or 0)
                    print(f"    Baseline ({bl['method']}): acc={bl.get('accuracy',0):.4f}")
                    if diff > 0.02:
                        print(f"    Best BEATS baseline by +{diff:.4f} accuracy ✓")
                    elif diff > 0:
                        print(f"    Best MARGINALLY better (+{diff:.4f} acc); may not be worth complexity")
                    else:
                        print(f"    SUSPICIOUS: Best does NOT beat baseline (diff={diff:.4f}); check output")
                else:
                    bl_mae   = bl.get("mae") or float("inf")
                    best_mae = best.get("mae") or float("inf")
                    diff     = bl_mae - best_mae
                    pct_imp  = diff / (bl_mae + 1e-10) * 100
                    print(f"    Baseline ({bl['method']}): MAE={bl_mae:.4f}")
                    if diff > bl_mae * 0.02:
                        print(f"    Best BEATS baseline: −{diff:.4f} MAE ({pct_imp:.1f}% reduction) ✓")
                    elif diff > 0:
                        print(f"    Best MARGINALLY better: −{diff:.4f} MAE ({pct_imp:.1f}%); "
                              f"may not justify complexity")
                    else:
                        print(f"    SUSPICIOUS: Best does NOT beat baseline (diff={diff:.4f} MAE); "
                              f"check output")

        print(f"    Full ranking:")
        for _, r in sub.sort_values("rank_within_feature").iterrows():
            if is_cat:
                print(f"      #{int(r['rank_within_feature'])}  {r['method']:30s}"
                      f"  acc={r.get('accuracy') or 0:.4f}  f1_macro={r.get('f1_macro') or 0:.4f}")
            else:
                print(f"      #{int(r['rank_within_feature'])}  {r['method']:30s}"
                      f"  MAE={r.get('mae') or 0:.4f}  r={r.get('pearson_r') or 0:.4f}")

    print()
    print(BANNER)
    print()


if __name__ == "__main__":
    main()
