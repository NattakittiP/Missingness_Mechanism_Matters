import os
import sys
import json
import time
import warnings
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

try:
    from tqdm.auto import tqdm

except ImportError:
    def tqdm(iterable=None, total=None, desc=None, leave=True, **kw):
        return iterable if iterable is not None else range(total or 0)
    tqdm.write = print

_HERE = Path(__file__).parent.resolve()

def _find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        if (cur / "PHASE 4").exists() and (cur / "Dataset").exists():
            return cur
        cur = cur.parent
    return start.parent.parent

_PROJECT_ROOT = _find_project_root(_HERE)

_PHASE4       = _PROJECT_ROOT / "PHASE 4"

_DATASET_DIR  = _PROJECT_ROOT / "Dataset"

if not _PHASE4.exists():
    raise FileNotFoundError(
        f"Expected 'PHASE 4' folder at {_PHASE4}.\n"
        f"Project root resolved to: {_PROJECT_ROOT}\n"
        "Place this script under <project_root>/IBCAST_MECHANISM_SWEEP/Code/"
    )

if not _DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Expected 'Dataset' folder at {_DATASET_DIR}.\n"
        "Ensure datasets are placed in <project_root>/Dataset/"
    )

sys.path.insert(0, str(_PHASE4))
import jcsse_audit_runner_tqdm_hardened as _base

_base.DATASET_A_PATH = str(
    _DATASET_DIR / "full_analytic_dataset_mortality_all_admissions.csv"
)

_base.DATASET_B_PATH = str(
    _DATASET_DIR / "Synthetic_Dataset_1500_Patients_precise.csv"
)
from jcsse_audit_runner_tqdm_hardened import (
    load_dataset_A,
    build_preprocessor,
    make_model_and_grid,
    get_outer,
    calibration_split_indices,
    predict_proba_safe,
    fit_best_model_nested,
    PrefitCalibrator,
    rank_key,
    compute_ece,
    now_ts,
    MODELS,
    SEEDS_20,
    OUTER_FOLDS,
)

try:
    import scikit_posthocs as sp
    _HAS_POSTHOCS = True

except ImportError:
    _HAS_POSTHOCS = False

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True

except ImportError:
    _HAS_JOBLIB = False

MECHANISMS: List[str]  = ["MCAR", "MAR", "MNAR"]

MISS_RATES: List[float] = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

SEEDS:      List[int]   = SEEDS_20

SPLITS:     List[str]   = ["S1", "S2"]

DATASET_TAG = "A"

N_JOBS: int = 4

OUT_DIR     = _HERE.parent

FIG_DIR     = OUT_DIR / "figures"

NEM_DIR     = OUT_DIR / "nemenyi_posthoc"

_JOBLIB_TMP = OUT_DIR / "joblib_tmp"

def _present_num_cols(X_df: pd.DataFrame, num_cols: List[str]) -> List[str]:
    return [c for c in num_cols if c in X_df.columns]

def apply_mcar_injection(
    X_df: pd.DataFrame,
    rate: float,
    seed: int,
    num_cols: List[str],
) -> pd.DataFrame:
    if rate <= 0.0:
        return X_df.copy()
    X2   = X_df.copy()
    cols = _present_num_cols(X2, num_cols)
    rng  = np.random.default_rng(seed)
    for c in cols:
        mask = rng.random(len(X2)) < rate
        X2.loc[X2.index[mask], c] = np.nan
    return X2

def apply_mar_injection(
    X_df: pd.DataFrame,
    rate: float,
    seed: int,
    num_cols: List[str],
    *,
    proxy_threshold: Optional[float] = None,
    proxy_stats: Optional[Tuple[pd.Series, pd.Series]] = None,
) -> Tuple[pd.DataFrame, float, Tuple[pd.Series, pd.Series]]:
    if rate <= 0.0:
        dummy = (pd.Series(dtype=float), pd.Series(dtype=float))
        return X_df.copy(), 0.0, dummy
    X2   = X_df.copy()
    cols = _present_num_cols(X2, num_cols)
    if not cols:
        dummy = (pd.Series(dtype=float), pd.Series(dtype=float))
        return X2, 0.0, dummy
    numeric_vals = X2[cols].apply(pd.to_numeric, errors="coerce")
    if proxy_stats is None:
        col_means = numeric_vals.mean()
        col_stds  = numeric_vals.std().replace(0.0, 1.0).fillna(1.0)
        proxy_stats = (col_means, col_stds)
    else:
        col_means, col_stds = proxy_stats
    z      = (numeric_vals - col_means) / col_stds
    proxy  = z.mean(axis=1).fillna(0.0).values
    if proxy_threshold is None:
        proxy_threshold = float(np.nanmedian(proxy))
    high_sev = proxy > proxy_threshold
    rng      = np.random.default_rng(seed)
    for c in cols:
        p_high = min(rate * 2.0, 0.95)
        p_low  = rate * 0.5
        prob   = np.where(high_sev, p_high, p_low)
        mask   = rng.random(len(X2)) < prob
        X2.loc[X2.index[mask], c] = np.nan
    return X2, proxy_threshold, proxy_stats

def apply_mnar_injection(
    X_df: pd.DataFrame,
    y:    np.ndarray,
    rate: float,
    seed: int,
    num_cols: List[str],
) -> pd.DataFrame:
    if rate <= 0.0:
        return X_df.copy()
    X2   = X_df.copy()
    cols = _present_num_cols(X2, num_cols)
    y    = np.asarray(y, dtype=int)
    rng  = np.random.default_rng(seed)
    for c in cols:
        p_pos = min(rate * 2.5, 0.95)
        p_neg = rate * 0.5
        prob  = np.where(y == 1, p_pos, p_neg)
        mask  = rng.random(len(X2)) < prob
        X2.loc[X2.index[mask], c] = np.nan
    return X2

def _injection_seed(seed: int, fold_id: int, rate: float, *, train: bool) -> int:
    rate_int = int(round(rate * 1000))
    parity   = 0 if train else 1
    return (seed * 100_000 + fold_id * 2000 + rate_int * 2 + parity) & 0xFFFF_FFFF

def eval_one_setting(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    num_cols: List[str],
    cat_cols:  List[str],
    *,
    mechanism: str,
    rate:      float,
    split_key: str,
    seed:      int,
    model_key: str,
) -> List[Dict[str, Any]]:
    outer      = get_outer(split_key, seed)
    split_iter = (
        outer.split(X, y, groups) if split_key == "S2" else outer.split(X, y)
    )
    rows: List[Dict[str, Any]] = []
    for fold_id, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        X_tr_raw = X.iloc[tr_idx].copy()
        y_tr     = y[tr_idx]
        X_te_raw = X.iloc[te_idx].copy()
        y_te     = y[te_idx]
        g_tr     = groups[tr_idx] if groups is not None else None
        seed_tr = _injection_seed(seed, fold_id, rate, train=True)
        seed_te = _injection_seed(seed, fold_id, rate, train=False)
        if mechanism == "MCAR":
            X_tr = apply_mcar_injection(X_tr_raw, rate, seed_tr, num_cols)
            X_te = apply_mcar_injection(X_te_raw, rate, seed_te, num_cols)
        elif mechanism == "MAR":
            X_tr, proxy_thr, proxy_stats = apply_mar_injection(
                X_tr_raw, rate, seed_tr, num_cols,
                proxy_threshold=None,
                proxy_stats=None,
            )
            X_te, _, _ = apply_mar_injection(
                X_te_raw, rate, seed_te, num_cols,
                proxy_threshold=proxy_thr,
                proxy_stats=proxy_stats,
            )
        elif mechanism == "MNAR":
            X_tr = apply_mnar_injection(X_tr_raw, y_tr, rate, seed_tr, num_cols)
            X_te = apply_mnar_injection(X_te_raw, y_te, rate, seed_te, num_cols)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism!r}")
        pre       = build_preprocessor(
            num_cols, cat_cols, include_imputer=True, include_scaler=True
        )
        base_model, grid, do_cal = make_model_and_grid(model_key, seed)
        base_pipe = Pipeline(steps=[("pre", pre), ("clf", base_model)])
        tr_sub, cal_sub = calibration_split_indices(split_key, y_tr, g_tr, seed)
        X_tune = X_tr.iloc[tr_sub]
        y_tune = y_tr[tr_sub]
        g_tune = g_tr[tr_sub] if g_tr is not None else None
        best_pipe, best_params = fit_best_model_nested(
            base_pipe, grid, split_key, X_tune, y_tune, g_tune, seed
        )
        best_pipe.fit(X_tune, y_tune)
        if do_cal:
            calibrator = PrefitCalibrator(best_pipe, method="sigmoid")
            calibrator.fit(X_tr.iloc[cal_sub], y_tr[cal_sub])
            final_model = calibrator
        else:
            final_model = best_pipe
        p_te = predict_proba_safe(final_model, X_te)
        auc   = float(roc_auc_score(y_te, p_te))
        ap    = float(average_precision_score(y_te, p_te))
        brier = float(brier_score_loss(y_te, np.clip(p_te, 0.0, 1.0)))
        ece   = float(compute_ece(y_te, p_te))
        rows.append({
            "mechanism": mechanism,
            "rate":      rate,
            "split":     split_key,
            "model":     model_key,
            "seed":      seed,
            "fold":      fold_id,
            "auroc":     auc,
            "ap":        ap,
            "brier":     brier,
            "ece":       ece,
        })
    return rows

def aggregate_folds(raw_df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["mechanism", "rate", "split", "model", "seed"]
    agg = raw_df.groupby(gcols).agg(
        auroc_mean  = ("auroc",  "mean"),
        ap_mean     = ("ap",     "mean"),
        brier_mean  = ("brier",  "mean"),
        ece_mean    = ("ece",    "mean"),
        auroc_std   = ("auroc",  "std"),
        ap_std      = ("ap",     "std"),
        brier_std   = ("brier",  "std"),
    ).reset_index()
    for c in ["auroc_std", "ap_std", "brier_std"]:
        agg[c] = agg[c].fillna(0.0)
    return agg

def compute_winners_ibcast(summary_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["mechanism", "rate", "split", "seed"]
    winners = []
    for keys, sub in summary_df.groupby(group_cols):
        best_row, best_k = None, None
        for _, r in sub.iterrows():
            k = rank_key(r["auroc_mean"], r["ap_mean"], r["brier_mean"])
            if best_k is None or k > best_k:
                best_k = k
                best_row = r
        winners.append({
            "mechanism":    keys[0],
            "rate":         keys[1],
            "split":        keys[2],
            "seed":         int(keys[3]),
            "winner_model": best_row["model"],
            "winner_auroc": float(best_row["auroc_mean"]),
            "winner_ap":    float(best_row["ap_mean"]),
            "winner_brier": float(best_row["brier_mean"]),
        })
    return pd.DataFrame(winners)

def _model_rank_vector(sub: pd.DataFrame) -> np.ndarray:
    sub = sub.copy()
    sub["_key"] = list(zip(sub["auroc_mean"], sub["ap_mean"], -sub["brier_mean"]))
    sub = sub.sort_values("_key", ascending=False).reset_index(drop=True)
    rank_dict = {m: i + 1 for i, m in enumerate(sub["model"].tolist())}
    return np.array([rank_dict.get(m, np.nan) for m in MODELS], dtype=float)

def build_envelope(
    winners_df:  pd.DataFrame,
    summary_df:  pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for mech, split_key in itertools.product(MECHANISMS, SPLITS):
        for rate in MISS_RATES:
            flips = []
            for seed in SEEDS:
                bw = winners_df[
                    (winners_df["mechanism"] == mech) &
                    (winners_df["rate"]      == 0.0) &
                    (winners_df["split"]     == split_key) &
                    (winners_df["seed"]      == seed)
                ]
                rw = winners_df[
                    (winners_df["mechanism"] == mech) &
                    (winners_df["rate"]      == rate) &
                    (winners_df["split"]     == split_key) &
                    (winners_df["seed"]      == seed)
                ]
                if bw.empty or rw.empty:
                    continue
                flips.append(
                    int(bw.iloc[0]["winner_model"] != rw.iloc[0]["winner_model"])
                )
            flip_pct = 100.0 * np.mean(flips) if flips else np.nan
            n_seeds  = len(flips)
            tau_vals, rho_vals = [], []
            for seed in SEEDS:
                base_sub = summary_df[
                    (summary_df["mechanism"] == mech) &
                    (summary_df["rate"]      == 0.0) &
                    (summary_df["split"]     == split_key) &
                    (summary_df["seed"]      == seed)
                ]
                rate_sub = summary_df[
                    (summary_df["mechanism"] == mech) &
                    (summary_df["rate"]      == rate) &
                    (summary_df["split"]     == split_key) &
                    (summary_df["seed"]      == seed)
                ]
                if len(base_sub) < len(MODELS) or len(rate_sub) < len(MODELS):
                    continue
                r_base = _model_rank_vector(base_sub)
                r_rate = _model_rank_vector(rate_sub)
                valid  = ~(np.isnan(r_base) | np.isnan(r_rate))
                if valid.sum() < 3:
                    continue
                tau, _ = sp_stats.kendalltau(r_base[valid], r_rate[valid])
                rho, _ = sp_stats.spearmanr(r_base[valid], r_rate[valid])
                tau_vals.append(float(tau))
                rho_vals.append(float(rho))
            records.append({
                "mechanism":        mech,
                "rate":             rate,
                "split":            split_key,
                "flip_pct":         float(flip_pct) if not np.isnan(flip_pct) else np.nan,
                "kendall_tau_mean": float(np.mean(tau_vals))  if tau_vals else np.nan,
                "kendall_tau_std":  float(np.std(tau_vals))   if tau_vals else np.nan,
                "spearman_rho_mean":float(np.mean(rho_vals))  if rho_vals else np.nan,
                "spearman_rho_std": float(np.std(rho_vals))   if rho_vals else np.nan,
                "n_seeds":          n_seeds,
            })
    return pd.DataFrame(records)

def compute_margin_gap(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_seed_rows: List[Dict[str, Any]] = []
    for keys, sub in summary_df.groupby(["mechanism", "rate", "split", "seed"]):
        srt = sub.sort_values(
            ["auroc_mean", "ap_mean", "brier_mean"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if len(srt) < 2:
            continue
        top1, top2 = srt.iloc[0], srt.iloc[1]
        by_seed_rows.append({
            "mechanism":   keys[0],
            "rate":        float(keys[1]),
            "split":       keys[2],
            "seed":        int(keys[3]),
            "top1_model":  top1["model"],
            "top1_auroc":  float(top1["auroc_mean"]),
            "top2_model":  top2["model"],
            "top2_auroc":  float(top2["auroc_mean"]),
            "gap":         float(top1["auroc_mean"]) - float(top2["auroc_mean"]),
        })
    by_seed = pd.DataFrame(by_seed_rows)
    summary_rows: List[Dict[str, Any]] = []
    for keys, sub in by_seed.groupby(["mechanism", "rate", "split"]):
        gaps = sub["gap"].to_numpy(dtype=float)
        n = len(gaps)
        mean = float(np.mean(gaps))
        std  = float(np.std(gaps, ddof=1)) if n > 1 else 0.0
        ci   = 1.96 * std / np.sqrt(max(n, 1))
        top1_mode = sub["top1_model"].mode().iloc[0] if not sub["top1_model"].empty else ""
        top2_mode = sub["top2_model"].mode().iloc[0] if not sub["top2_model"].empty else ""
        summary_rows.append({
            "mechanism":     keys[0],
            "rate":          float(keys[1]),
            "split":         keys[2],
            "gap_mean":      mean,
            "gap_std":       std,
            "gap_ci95_lo":   mean - ci,
            "gap_ci95_hi":   mean + ci,
            "n_seeds":       n,
            "top1_mode":     top1_mode,
            "top2_mode":     top2_mode,
            "frac_top1_dominant": float((sub["gap"] > 0.005).mean()),
            "frac_tied_or_flipped": float((sub["gap"] <= 0.005).mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mechanism", "split", "rate"]
    ).reset_index(drop=True)
    return by_seed, summary

def plot_fig5_margin_gap(margin_summary: pd.DataFrame, fig_dir: Path) -> None:
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.suptitle(
        "Margin Gap: AUROC(top-1) − AUROC(top-2) vs. Missingness Rate\n"
        "(Mean over 20 seeds; shaded = 95% CI)",
        fontsize=11, fontweight="bold",
    )
    for ax, split_key in zip(axes, ["S1", "S2"]):
        sub = margin_summary[margin_summary["split"] == split_key]
        for mech in MECHANISMS:
            msub = sub[sub["mechanism"] == mech].sort_values("rate")
            if msub.empty:
                continue
            x = msub["rate"].to_numpy() * 100
            y = msub["gap_mean"].to_numpy()
            lo = msub["gap_ci95_lo"].to_numpy()
            hi = msub["gap_ci95_hi"].to_numpy()
            ax.plot(x, y, color=_MECH_COLORS[mech], marker=_MECH_MARKERS[mech],
                    label=mech, zorder=3)
            ax.fill_between(x, lo, hi, color=_MECH_COLORS[mech], alpha=0.18,
                            zorder=2)
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--",
                   label="_nolegend_")
        ax.set_title(
            "S1: Stratified K-Fold" if split_key == "S1" else "S2: Group K-Fold"
        )
        ax.set_xlabel("Nominal Missingness Rate (%)")
        ax.set_ylabel("AUROC margin (top-1 − top-2)")
        ax.set_xlim(-2, 53)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Mechanism", loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save_fig(plt, fig, fig_dir, "fig5_margin_gap")

def compute_per_feature_missingness_leak(
    X: pd.DataFrame,
    y: np.ndarray,
    num_cols: List[str],
    *,
    mechanisms: Optional[List[str]] = None,
    rates:      Optional[List[float]] = None,
    seeds:      Optional[List[int]]   = None,
) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    if mechanisms is None:
        mechanisms = list(MECHANISMS)
    if rates is None:
        rates = list(MISS_RATES)
    if seeds is None:
        seeds = list(SEEDS)
    cols = _present_num_cols(X, num_cols)
    y_arr = np.asarray(y, dtype=int)
    rows: List[Dict[str, Any]] = []
    for mech in mechanisms:
        for rate in rates:
            for seed in seeds:
                inj_seed = _injection_seed(seed, fold_id=0, rate=rate, train=True)
                if mech == "MCAR":
                    X_miss = apply_mcar_injection(X, rate, inj_seed, cols)
                elif mech == "MAR":
                    X_miss, _, _ = apply_mar_injection(
                        X, rate, inj_seed, cols,
                        proxy_threshold=None, proxy_stats=None,
                    )
                elif mech == "MNAR":
                    X_miss = apply_mnar_injection(X, y_arr, rate, inj_seed, cols)
                else:
                    continue
                for c in cols:
                    is_miss = X_miss[c].isna().to_numpy().astype(int)
                    miss_rate_observed = float(is_miss.mean())
                    if miss_rate_observed in (0.0, 1.0):
                        auc = float("nan")
                    else:
                        try:
                            auc = float(roc_auc_score(y_arr, is_miss))
                        except Exception:
                            auc = float("nan")
                    rows.append({
                        "mechanism":      mech,
                        "rate":           float(rate),
                        "seed":           int(seed),
                        "feature":        c,
                        "missing_rate":   miss_rate_observed,
                        "auroc_indicator": auc,
                    })
    df = pd.DataFrame(rows)
    return df

def aggregate_per_feature_leak(per_feat_df: pd.DataFrame) -> pd.DataFrame:
    agg = per_feat_df.groupby(["mechanism", "rate", "feature"]).agg(
        auroc_mean       = ("auroc_indicator", "mean"),
        auroc_std        = ("auroc_indicator", "std"),
        missing_rate_mean= ("missing_rate", "mean"),
        n_seeds          = ("seed", "nunique"),
    ).reset_index()
    agg["auroc_std"] = agg["auroc_std"].fillna(0.0)
    agg["leak_strength"] = (agg["auroc_mean"] - 0.5).abs()
    return agg

def plot_fig6_per_feature_leak(
    per_feat_agg: pd.DataFrame,
    fig_dir: Path,
    *,
    top_k: int = 12,
    rate_for_topk: float = 0.30,
) -> None:
    plt = _setup_mpl()
    pick = (
        per_feat_agg[
            (per_feat_agg["mechanism"] == "MNAR") &
            (per_feat_agg["rate"] == rate_for_topk)
        ]
        .sort_values("leak_strength", ascending=False)
        .head(top_k)["feature"]
        .tolist()
    )
    if not pick:
        return
    rates_sorted = sorted(per_feat_agg["rate"].unique().tolist())
    fig = plt.figure(figsize=(12.5, max(5.5, 0.45 * len(pick) + 2)))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.32)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[0, 1])
    mat = np.full((len(pick), len(rates_sorted)), np.nan)
    for i, feat in enumerate(pick):
        for j, r in enumerate(rates_sorted):
            row = per_feat_agg[
                (per_feat_agg["mechanism"] == "MNAR") &
                (per_feat_agg["feature"] == feat) &
                (per_feat_agg["rate"] == r)
            ]
            if not row.empty:
                mat[i, j] = float(row.iloc[0]["auroc_mean"])
    im = ax_heat.imshow(mat, vmin=0.40, vmax=0.85, cmap="YlOrRd", aspect="auto")
    ax_heat.set_xticks(range(len(rates_sorted)))
    ax_heat.set_xticklabels([f"{int(r*100)}%" for r in rates_sorted])
    ax_heat.set_yticks(range(len(pick)))
    ax_heat.set_yticklabels(pick)
    ax_heat.set_title(
        f"MNAR — Per-Feature Leak AUROC\n"
        f"(top {len(pick)} features by leak strength @ rate={int(rate_for_topk*100)}%)",
        fontsize=10, fontweight="bold",
    )
    ax_heat.set_xlabel("Nominal Missingness Rate")
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="AUROC of is_missing(·)→y")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax_heat.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="black" if v < 0.65 else "white",
                )
    for feat in pick:
        ms = per_feat_agg[
            (per_feat_agg["mechanism"] == "MNAR") &
            (per_feat_agg["feature"] == feat)
        ].sort_values("rate")
        cs = per_feat_agg[
            (per_feat_agg["mechanism"] == "MCAR") &
            (per_feat_agg["feature"] == feat)
        ].sort_values("rate")
        if not ms.empty:
            ax_line.plot(
                ms["rate"] * 100, ms["auroc_mean"],
                marker="^", linewidth=1.2, alpha=0.85,
                label=f"MNAR · {feat}",
                color=_MECH_COLORS["MNAR"],
            )
        if not cs.empty:
            ax_line.plot(
                cs["rate"] * 100, cs["auroc_mean"],
                marker="o", linewidth=0.8, alpha=0.35, linestyle=":",
                color=_MECH_COLORS["MCAR"],
                label="_nolegend_",
            )
    ax_line.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    ax_line.set_xlim(-2, 53)
    ax_line.set_ylim(0.40, 0.90)
    ax_line.set_title(
        "Indicator AUROC vs. Rate\n(green = MNAR per-feature; blue = MCAR per-feature, ref)",
        fontsize=10, fontweight="bold",
    )
    ax_line.set_xlabel("Nominal Missingness Rate (%)")
    ax_line.set_ylabel("AUROC of is_missing(feature) → y")
    ax_line.grid(True, alpha=0.3)
    fig.suptitle(
        "Figure 6 — Per-Feature Missingness Leak under MNAR\n"
        "(features whose 'is-missing' mask alone predicts mortality drive the shortcut)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(plt, fig, fig_dir, "fig6_per_feature_mnar_leak")

def run_friedman_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for mech, rate, split_key in itertools.product(MECHANISMS, MISS_RATES, SPLITS):
        sub = summary_df[
            (summary_df["mechanism"] == mech) &
            (summary_df["rate"]      == rate) &
            (summary_df["split"]     == split_key)
        ]
        pivot = sub.pivot_table(index="seed", columns="model", values="auroc_mean")
        pivot = pivot[[m for m in MODELS if m in pivot.columns]].dropna()
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            continue
        try:
            stat, p = sp_stats.friedmanchisquare(
                *[pivot[m].values for m in pivot.columns]
            )
        except Exception:
            stat, p = np.nan, np.nan
        avg_ranks = {
            m: float(pivot[m].rank(ascending=False).mean())
            for m in pivot.columns
        }
        records.append({
            "mechanism":      mech,
            "rate":           rate,
            "split":          split_key,
            "friedman_stat":  float(stat),
            "p_value":        float(p),
            "significant_p05": bool(p < 0.05) if not np.isnan(p) else False,
            **{f"avg_rank_{m}": v for m, v in avg_ranks.items()},
        })
    return pd.DataFrame(records)

def _nemenyi_manual(data: pd.DataFrame) -> pd.DataFrame:
    k  = data.shape[1]
    N  = data.shape[0]
    ar = data.rank(axis=1, ascending=True).mean(axis=0)
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    models = list(data.columns)
    pmat   = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            z = abs(ar.iloc[i] - ar.iloc[j]) / se
            p = float(2.0 * (1.0 - sp_stats.norm.cdf(z)))
            pmat[i, j] = p
            pmat[j, i] = p
    return pd.DataFrame(pmat, index=models, columns=models)

def run_nemenyi_posthoc(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _nemenyi_fn = sp.posthoc_nemenyi_friedman if _HAS_POSTHOCS else _nemenyi_manual
    for mech, rate, split_key in itertools.product(MECHANISMS, MISS_RATES, SPLITS):
        sub = summary_df[
            (summary_df["mechanism"] == mech) &
            (summary_df["rate"]      == rate) &
            (summary_df["split"]     == split_key)
        ]
        pivot = sub.pivot_table(index="seed", columns="model", values="auroc_mean")
        pivot = pivot[[m for m in MODELS if m in pivot.columns]].dropna()
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            continue
        try:
            _, p = sp_stats.friedmanchisquare(
                *[pivot[m].values for m in pivot.columns]
            )
            if np.isnan(p) or p >= 0.05:
                continue
            pmat = _nemenyi_fn(pivot)
            fname = out_dir / f"nemenyi_{mech}_rate{int(rate*100):02d}_{split_key}.csv"
            if isinstance(pmat, pd.DataFrame):
                pmat.to_csv(fname)
            else:
                pd.DataFrame(
                    pmat, index=pivot.columns, columns=pivot.columns
                ).to_csv(fname)
        except Exception:
            pass

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "lines.linewidth":   1.8,
        "lines.markersize":  6,
    })
    return plt

_MECH_COLORS  = {"MCAR": "#1f77b4", "MAR": "#ff7f0e", "MNAR": "#2ca02c"}

_MECH_MARKERS = {"MCAR": "o",       "MAR": "s",       "MNAR": "^"}

_MODEL_LABELS = {
    "lr_l2": "LR-L2", "svm_linear_cal": "SVM",
    "rf": "RF", "xgb": "XGBoost", "extratrees": "ExtraT",
}

def _save_fig(plt, fig, fig_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{stem}.{ext}")
    plt.close(fig)

def plot_fig1_envelope_flip(envelope_df: pd.DataFrame, fig_dir: Path) -> None:
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.suptitle(
        "Winner Flip Rate vs. Missingness Rate by Mechanism\n"
        "(MIMIC-IV v3.1, N=14 081, 20 seeds, Protocol P0)",
        fontsize=11, fontweight="bold",
    )
    for ax, split_key in zip(axes, ["S1", "S2"]):
        sub = envelope_df[envelope_df["split"] == split_key]
        for mech in MECHANISMS:
            msub = sub[sub["mechanism"] == mech].sort_values("rate")
            if msub.empty:
                continue
            ax.plot(
                msub["rate"] * 100,
                msub["flip_pct"],
                color=_MECH_COLORS[mech],
                marker=_MECH_MARKERS[mech],
                label=mech,
                zorder=3,
            )
        ax.set_title(
            "S1: Stratified K-Fold" if split_key == "S1" else "S2: Group K-Fold"
        )
        ax.set_xlabel("Nominal Missingness Rate (%)")
        ax.set_ylabel("Winner Flip Rate (%)")
        ax.set_xlim(-2, 53)
        ax.set_ylim(-5, 105)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Mechanism", loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_fig(plt, fig, fig_dir, "fig1_envelope_flip")

def plot_fig2_kendall_heatmap(envelope_df: pd.DataFrame, fig_dir: Path) -> None:
    plt = _setup_mpl()
    rate_labels = [f"{int(r*100)}%" for r in MISS_RATES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.suptitle(
        "Kendall's τ of 5-Model Ranking vs. Rate-0 Baseline\n"
        "(Mean over 20 seeds;  τ = 1 → ranking fully preserved)",
        fontsize=11, fontweight="bold",
    )
    for ax, split_key in zip(axes, ["S1", "S2"]):
        sub = envelope_df[envelope_df["split"] == split_key]
        matrix = []
        for mech in MECHANISMS:
            msub = sub[sub["mechanism"] == mech].sort_values("rate")
            matrix.append(msub["kendall_tau_mean"].tolist())
        mat = np.array(matrix, dtype=float)
        im  = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(MISS_RATES)))
        ax.set_xticklabels(rate_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(MECHANISMS)))
        ax.set_yticklabels(MECHANISMS)
        ax.set_xlabel("Missingness Rate")
        ax.set_title(
            "S1: Stratified K-Fold" if split_key == "S1" else "S2: Group K-Fold"
        )
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v   = mat[i, j]
                txt = f"{v:.2f}" if not np.isnan(v) else "—"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black" if abs(v) < 0.7 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Kendall's τ")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save_fig(plt, fig, fig_dir, "fig2_kendall_tau_heatmap")

def plot_fig3_auroc_boxplot(
    summary_df: pd.DataFrame,
    rate: float,
    fig_dir: Path,
) -> None:
    plt   = _setup_mpl()
    sub   = summary_df[summary_df["rate"] == rate]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"AUROC Distribution by Model & Mechanism at {int(rate*100)}% Missingness\n"
        "(20 seeds, Protocol P0, Dataset A — MIMIC-IV v3.1)",
        fontsize=11, fontweight="bold",
    )
    x_pos  = np.arange(len(MODELS))
    width  = 0.25
    for ax, split_key in zip(axes, ["S1", "S2"]):
        ssub = sub[sub["split"] == split_key]
        for i, mech in enumerate(MECHANISMS):
            msub = ssub[ssub["mechanism"] == mech]
            data_per_model = [
                msub[msub["model"] == m]["auroc_mean"].values for m in MODELS
            ]
            ax.boxplot(
                data_per_model,
                positions=x_pos + (i - 1) * width,
                widths=width * 0.85,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
                boxprops={"facecolor": _MECH_COLORS[mech], "alpha": 0.70},
            )
            ax.plot([], [], color=_MECH_COLORS[mech], linewidth=6,
                    alpha=0.7, label=mech)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_MODEL_LABELS.get(m, m) for m in MODELS])
        ax.set_xlabel("Model")
        ax.set_ylabel("AUROC (mean over outer folds)")
        ax.set_title(
            "S1: Stratified K-Fold" if split_key == "S1" else "S2: Group K-Fold"
        )
        ax.legend(title="Mechanism", loc="lower right")
        ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save_fig(plt, fig, fig_dir, f"fig3_auroc_boxplot_rate{int(rate*100):02d}")

def plot_fig4_mechanism_divergence(
    envelope_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    plt  = _setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.suptitle(
        "Differential Ranking Stability: Structured vs. Random Missingness\n"
        r"(Δτ = τ[MAR or MNAR] − τ[MCAR];  >0 → structured mechanism more stable)",
        fontsize=11, fontweight="bold",
    )
    delta_style = {
        "MAR − MCAR":  {"color": _MECH_COLORS["MAR"],  "marker": "s"},
        "MNAR − MCAR": {"color": _MECH_COLORS["MNAR"], "marker": "^"},
    }
    for ax, split_key in zip(axes, ["S1", "S2"]):
        sub  = envelope_df[envelope_df["split"] == split_key]
        mcar = sub[sub["mechanism"] == "MCAR"].sort_values("rate")
        mar  = sub[sub["mechanism"] == "MAR"].sort_values("rate")
        mnar = sub[sub["mechanism"] == "MNAR"].sort_values("rate")
        rates_pct = mcar["rate"].values * 100
        if not mar.empty and not mcar.empty:
            d = mar["kendall_tau_mean"].values - mcar["kendall_tau_mean"].values
            ax.plot(rates_pct, d, label="MAR − MCAR",
                    **delta_style["MAR − MCAR"])
            ax.fill_between(rates_pct, d, alpha=0.12,
                            color=delta_style["MAR − MCAR"]["color"])
        if not mnar.empty and not mcar.empty:
            d = mnar["kendall_tau_mean"].values - mcar["kendall_tau_mean"].values
            ax.plot(rates_pct, d, label="MNAR − MCAR",
                    **delta_style["MNAR − MCAR"])
            ax.fill_between(rates_pct, d, alpha=0.12,
                            color=delta_style["MNAR − MCAR"]["color"])
        ax.axhline(0, color="grey", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Nominal Missingness Rate (%)")
        ax.set_ylabel("Δ Kendall's τ")
        ax.set_title(
            "S1: Stratified K-Fold" if split_key == "S1" else "S2: Group K-Fold"
        )
        ax.set_xlim(-2, 53)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save_fig(plt, fig, fig_dir, "fig4_mechanism_divergence")

def _safe_run(
    X, y, groups, num_cols, cat_cols,
    mechanism, rate, split_key, seed, model_key,
) -> List[Dict[str, Any]]:
    try:
        return eval_one_setting(
            X, y, groups, num_cols, cat_cols,
            mechanism=mechanism, rate=rate,
            split_key=split_key, seed=seed, model_key=model_key,
        )
    except Exception as exc:
        tqdm.write(
            f"  [WARN] {mechanism} rate={rate:.2f} {split_key} "
            f"seed={seed} model={model_key}: {exc}"
        )
        return []

def main() -> None:
    t0 = time.time()
    tqdm.write(f"[{now_ts()}] ══ IBCAST Mechanism Sweep — starting ══")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NEM_DIR.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"[{now_ts()}] Loading Dataset A (MIMIC-IV v3.1)…")
    X, y, groups, num_cols, cat_cols = load_dataset_A()
    tqdm.write(
        f"  N={len(y):,}  pos={int(y.sum())}  ({100*y.mean():.2f}%)  "
        f"num_feats={len(num_cols)}  cat_feats={len(cat_cols)}"
    )
    jobs = list(itertools.product(MECHANISMS, MISS_RATES, SPLITS, SEEDS, MODELS))
    tqdm.write(
        f"[{now_ts()}] Total jobs: {len(jobs):,}  "
        f"({len(MECHANISMS)} mechs × {len(MISS_RATES)} rates × "
        f"{len(SPLITS)} splits × {len(SEEDS)} seeds × {len(MODELS)} models)"
    )
    raw_path       = OUT_DIR / "metrics_raw.csv"
    completed_keys: set = set()
    first_write    = not raw_path.exists()
    if raw_path.exists():
        try:
            existing = pd.read_csv(raw_path)
            for _, r in existing.iterrows():
                completed_keys.add(
                    (r["mechanism"], float(r["rate"]), r["split"],
                     int(r["seed"]), r["model"])
                )
            tqdm.write(
                f"[{now_ts()}] Resuming — {len(completed_keys):,} jobs already done."
            )
        except Exception:
            pass
    pending = [
        j for j in jobs
        if (j[0], j[1], j[2], j[3], j[4]) not in completed_keys
    ]
    tqdm.write(f"[{now_ts()}] Pending jobs: {len(pending):,}")
    pbar = tqdm(total=len(pending), desc="Mechanism sweep", dynamic_ncols=True)
    BATCH_SIZE = max(N_JOBS * 4, 4)
    if N_JOBS > 1 and _HAS_JOBLIB:
        tqdm.write(f"[{now_ts()}] Parallel mode: n_jobs={N_JOBS}")
        _JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
        os.environ["JOBLIB_TEMP_FOLDER"] = str(_JOBLIB_TMP)
        tqdm.write(f"[{now_ts()}] joblib temp folder → {_JOBLIB_TMP}")
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            tqdm.write(
                f"[{now_ts()}] Running batch "
                f"{start + 1:,}–{start + len(batch):,} / {len(pending):,}"
            )
            results = Parallel(n_jobs=N_JOBS, temp_folder=str(_JOBLIB_TMP), verbose=0)(
                delayed(_safe_run)(
                    X, y, groups, num_cols, cat_cols,
                    mech, rate, split_key, seed, model_key
                )
                for mech, rate, split_key, seed, model_key in batch
            )
            all_rows = [row for fold_rows in results for row in fold_rows]
            if all_rows:
                df_new = pd.DataFrame(all_rows)
                df_new.to_csv(
                    raw_path,
                    mode="a" if not first_write else "w",
                    header=first_write,
                    index=False,
                )
                first_write = False
            pbar.update(len(batch))
    else:
        tqdm.write(f"[{now_ts()}] Serial mode (joblib disabled or N_JOBS<=1).")
        for mech, rate, split_key, seed, model_key in pending:
            pbar.set_postfix(
                mech=mech, rate=f"{rate:.2f}", sp=split_key,
                s=seed, m=model_key[:5], refresh=False,
            )
            fold_rows = _safe_run(
                X, y, groups, num_cols, cat_cols,
                mech, rate, split_key, seed, model_key,
            )
            if fold_rows:
                df_chunk = pd.DataFrame(fold_rows)
                df_chunk.to_csv(
                    raw_path,
                    mode="a" if not first_write else "w",
                    header=first_write,
                    index=False,
                )
                first_write = False
            pbar.update(1)
    pbar.close()
    tqdm.write(f"[{now_ts()}] Sweep complete.")
    tqdm.write(f"[{now_ts()}] Loading raw metrics…")
    raw_df = pd.read_csv(raw_path)
    tqdm.write(f"  {len(raw_df):,} fold-level rows")
    tqdm.write(f"[{now_ts()}] Aggregating folds…")
    summary_df = aggregate_folds(raw_df)
    summary_df.to_csv(OUT_DIR / "summary_by_setting.csv", index=False)
    tqdm.write(f"[{now_ts()}] Computing winners…")
    winners_df = compute_winners_ibcast(summary_df)
    winners_df.to_csv(OUT_DIR / "winners_by_seed.csv", index=False)
    tqdm.write(f"[{now_ts()}] Building robustness envelope…")
    envelope_df = build_envelope(winners_df, summary_df)
    envelope_df.to_csv(OUT_DIR / "envelope_by_mechanism.csv", index=False)
    tqdm.write(f"[{now_ts()}] Friedman tests…")
    friedman_df = run_friedman_tests(summary_df)
    friedman_df.to_csv(OUT_DIR / "friedman_results.csv", index=False)
    tqdm.write(f"[{now_ts()}] Nemenyi posthoc (significant conditions)…")
    run_nemenyi_posthoc(summary_df, NEM_DIR)
    tqdm.write(f"[{now_ts()}] Generating figures…")
    for fig_fn, label in [
        (lambda: plot_fig1_envelope_flip(envelope_df, FIG_DIR),
         "Fig 1: Envelope flip%"),
        (lambda: plot_fig2_kendall_heatmap(envelope_df, FIG_DIR),
         "Fig 2: Kendall τ heatmap"),
        (lambda: plot_fig3_auroc_boxplot(summary_df, 0.30, FIG_DIR),
         "Fig 3: AUROC boxplot @ 30%"),
        (lambda: plot_fig4_mechanism_divergence(envelope_df, FIG_DIR),
         "Fig 4: Mechanism divergence Δτ"),
    ]:
        try:
            fig_fn()
            tqdm.write(f"  ✓ {label}")
        except Exception as exc:
            tqdm.write(f"  [WARN] {label}: {exc}")
    elapsed = (time.time() - t0) / 60.0
    tqdm.write(f"\n{'═'*68}")
    tqdm.write(f"  IBCAST Mechanism Sweep — COMPLETE  ({elapsed:.1f} min)")
    tqdm.write(f"{'═'*68}")
    tqdm.write(f"  Output dir  : {(OUT_DIR).resolve()}")
    tqdm.write(f"  Raw rows    : {len(raw_df):,}")
    tqdm.write(f"  Summary rows: {len(summary_df):,}")
    tqdm.write(f"  Winners     : {len(winners_df):,}")
    if not envelope_df.empty:
        tqdm.write("\n  — Flip % (all splits, first 4 rates) —")
        try:
            flip_pivot = envelope_df.pivot_table(
                index=["mechanism", "split"], columns="rate", values="flip_pct"
            )
            cols4 = [c for c in flip_pivot.columns if c <= 0.30]
            tqdm.write(flip_pivot[cols4].round(1).to_string())
        except Exception:
            pass
    if not friedman_df.empty:
        sig = friedman_df[friedman_df["significant_p05"]]
        tqdm.write(
            f"\n  Friedman p<0.05: {len(sig)}/{len(friedman_df)} conditions"
        )
    tqdm.write(f"\n  Figures: {FIG_DIR.resolve()}")
    tqdm.write(f"{'═'*68}\n")

if __name__ == "__main__":
    main()
