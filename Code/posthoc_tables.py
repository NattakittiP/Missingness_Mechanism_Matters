from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\Sabi\Desktop\Non-Name-Yet\IBCAST_MECHANISM_SWEEP")

RATES_OF_INTEREST = [0.30, 0.40, 0.50]

MECHANISMS = ["MCAR", "MAR", "MNAR"]

SPLITS = ["S1", "S2"]

def read_csv_required(filename: str) -> pd.DataFrame:
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path.resolve()}")
    return pd.read_csv(path)

def fmt_pct(x):
    if pd.isna(x):
        return np.nan
    return round(float(x), 2)

def fmt4(x):
    if pd.isna(x):
        return np.nan
    return round(float(x), 4)

def normalize_rate(df: pd.DataFrame, col: str = "rate") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(float).round(2)
    return df

def build_main_mechanism_table() -> pd.DataFrame:
    envelope = read_csv_required("envelope_by_mechanism.csv")
    margin = read_csv_required("margin_gap_summary.csv")
    envelope = normalize_rate(envelope, "rate")
    margin = normalize_rate(margin, "rate")
    envelope = envelope[envelope["rate"].isin(RATES_OF_INTEREST)].copy()
    margin = margin[margin["rate"].isin(RATES_OF_INTEREST)].copy()
    need_env = [
        "mechanism", "rate", "split",
        "flip_pct", "kendall_tau_mean", "spearman_rho_mean", "n_seeds"
    ]
    need_margin = [
        "mechanism", "rate", "split",
        "gap_mean", "gap_ci95_lo", "gap_ci95_hi",
        "top1_mode", "top2_mode",
        "frac_top1_dominant", "frac_tied_or_flipped"
    ]
    missing_env = [c for c in need_env if c not in envelope.columns]
    missing_margin = [c for c in need_margin if c not in margin.columns]
    if missing_env:
        raise ValueError(f"envelope_by_mechanism.csv missing columns: {missing_env}")
    if missing_margin:
        raise ValueError(f"margin_gap_summary.csv missing columns: {missing_margin}")
    table = envelope[need_env].merge(
        margin[need_margin],
        on=["mechanism", "rate", "split"],
        how="left"
    )
    table["rate_pct"] = table["rate"] * 100
    table["flip_pct"] = table["flip_pct"].map(fmt_pct)
    table["kendall_tau_mean"] = table["kendall_tau_mean"].map(fmt4)
    table["spearman_rho_mean"] = table["spearman_rho_mean"].map(fmt4)
    table["auroc_margin_pp"] = table["gap_mean"] * 100
    table["auroc_margin_ci95_lo_pp"] = table["gap_ci95_lo"] * 100
    table["auroc_margin_ci95_hi_pp"] = table["gap_ci95_hi"] * 100
    for c in ["auroc_margin_pp", "auroc_margin_ci95_lo_pp", "auroc_margin_ci95_hi_pp"]:
        table[c] = table[c].map(fmt4)
    table["frac_top1_dominant"] = table["frac_top1_dominant"].map(fmt4)
    table["frac_tied_or_flipped"] = table["frac_tied_or_flipped"].map(fmt4)
    table = table[
        [
            "mechanism", "split", "rate_pct", "n_seeds",
            "flip_pct",
            "kendall_tau_mean",
            "spearman_rho_mean",
            "top1_mode", "top2_mode",
            "auroc_margin_pp",
            "auroc_margin_ci95_lo_pp",
            "auroc_margin_ci95_hi_pp",
            "frac_top1_dominant",
            "frac_tied_or_flipped",
        ]
    ].sort_values(["mechanism", "split", "rate_pct"]).reset_index(drop=True)
    return table

def build_shortcut_evidence_table() -> pd.DataFrame:
    leak = read_csv_required("per_feature_leak_summary.csv")
    leak = normalize_rate(leak, "rate")
    leak = leak[leak["rate"].isin(RATES_OF_INTEREST)].copy()
    need = [
        "mechanism", "rate", "feature",
        "auroc_mean", "auroc_std",
        "missing_rate_mean",
        "leak_strength"
    ]
    missing = [c for c in need if c not in leak.columns]
    if missing:
        raise ValueError(f"per_feature_leak_summary.csv missing columns: {missing}")
    rows = []
    for (mechanism, rate), sub in leak.groupby(["mechanism", "rate"]):
        sub = sub.copy()
        sub_valid = sub.dropna(subset=["auroc_mean"])
        if sub_valid.empty:
            continue
        top_row = sub_valid.sort_values("auroc_mean", ascending=False).iloc[0]
        rows.append({
            "mechanism": mechanism,
            "rate_pct": rate * 100,
            "n_features": int(sub_valid["feature"].nunique()),
            "mean_indicator_auroc": float(sub_valid["auroc_mean"].mean()),
            "median_indicator_auroc": float(sub_valid["auroc_mean"].median()),
            "max_indicator_auroc": float(sub_valid["auroc_mean"].max()),
            "top_feature": top_row["feature"],
            "top_feature_indicator_auroc": float(top_row["auroc_mean"]),
            "top_feature_missing_rate": float(top_row["missing_rate_mean"]),
            "mean_abs_deviation_from_chance": float((sub_valid["auroc_mean"] - 0.5).abs().mean()),
        })
    table = pd.DataFrame(rows)
    numeric_cols = [
        "rate_pct",
        "mean_indicator_auroc",
        "median_indicator_auroc",
        "max_indicator_auroc",
        "top_feature_indicator_auroc",
        "top_feature_missing_rate",
        "mean_abs_deviation_from_chance",
    ]
    for c in numeric_cols:
        table[c] = table[c].map(fmt4)
    table = table.sort_values(["mechanism", "rate_pct"]).reset_index(drop=True)
    return table

def build_calibration_winner_table() -> pd.DataFrame:
    winners = read_csv_required("winners_by_seed.csv")
    summary = read_csv_required("summary_by_setting.csv")
    winners = normalize_rate(winners, "rate")
    summary = normalize_rate(summary, "rate")
    winners = winners[winners["rate"].isin(RATES_OF_INTEREST)].copy()
    summary = summary[summary["rate"].isin(RATES_OF_INTEREST)].copy()
    need_w = ["mechanism", "rate", "split", "seed", "winner_model"]
    need_s = [
        "mechanism", "rate", "split", "seed", "model",
        "auroc_mean", "ap_mean", "brier_mean", "ece_mean"
    ]
    missing_w = [c for c in need_w if c not in winners.columns]
    missing_s = [c for c in need_s if c not in summary.columns]
    if missing_w:
        raise ValueError(f"winners_by_seed.csv missing columns: {missing_w}")
    if missing_s:
        raise ValueError(f"summary_by_setting.csv missing columns: {missing_s}")
    merged = winners[need_w].merge(
        summary[need_s],
        left_on=["mechanism", "rate", "split", "seed", "winner_model"],
        right_on=["mechanism", "rate", "split", "seed", "model"],
        how="left"
    )
    rows = []
    for (mechanism, rate, split), sub in merged.groupby(["mechanism", "rate", "split"]):
        sub = sub.copy()
        winner_counts = sub["winner_model"].value_counts()
        mode_winner = winner_counts.index[0]
        mode_winner_count = int(winner_counts.iloc[0])
        rows.append({
            "mechanism": mechanism,
            "split": split,
            "rate_pct": rate * 100,
            "n_seeds": int(sub["seed"].nunique()),
            "mode_winner": mode_winner,
            "mode_winner_count": mode_winner_count,
            "mode_winner_pct": 100 * mode_winner_count / max(1, sub["seed"].nunique()),
            "winner_auroc_mean": float(sub["auroc_mean"].mean()),
            "winner_auroc_std": float(sub["auroc_mean"].std(ddof=1)),
            "winner_ap_mean": float(sub["ap_mean"].mean()),
            "winner_ap_std": float(sub["ap_mean"].std(ddof=1)),
            "winner_brier_mean": float(sub["brier_mean"].mean()),
            "winner_brier_std": float(sub["brier_mean"].std(ddof=1)),
            "winner_ece_mean": float(sub["ece_mean"].mean()),
            "winner_ece_std": float(sub["ece_mean"].std(ddof=1)),
        })
    table = pd.DataFrame(rows)
    for c in table.columns:
        if c.endswith("_mean") or c.endswith("_std"):
            table[c] = table[c].map(fmt4)
    table["mode_winner_pct"] = table["mode_winner_pct"].map(fmt_pct)
    table["rate_pct"] = table["rate_pct"].map(fmt4)
    table = table.sort_values(["mechanism", "split", "rate_pct"]).reset_index(drop=True)
    return table

def build_paper_ready_summary(
    main_table: pd.DataFrame,
    shortcut_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
) -> pd.DataFrame:
    shortcut_keep = shortcut_table[
        [
            "mechanism", "rate_pct",
            "mean_indicator_auroc",
            "max_indicator_auroc",
            "top_feature",
            "top_feature_indicator_auroc",
        ]
    ].copy()
    combined = main_table.merge(
        shortcut_keep,
        on=["mechanism", "rate_pct"],
        how="left"
    )
    cal_keep = calibration_table[
        [
            "mechanism", "split", "rate_pct",
            "mode_winner",
            "mode_winner_pct",
            "winner_auroc_mean",
            "winner_ap_mean",
            "winner_brier_mean",
            "winner_ece_mean",
        ]
    ].copy()
    combined = combined.merge(
        cal_keep,
        on=["mechanism", "split", "rate_pct"],
        how="left"
    )
    return combined.sort_values(["mechanism", "split", "rate_pct"]).reset_index(drop=True)

def export_latex_table(df: pd.DataFrame, filename: str, caption: str, label: str):
    latex = df.to_latex(
        index=False,
        escape=False,
        longtable=False,
        float_format="%.4f",
        caption=caption,
        label=label,
    )
    path = BASE_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    return path

def main():
    print("Building IBCAST post-hoc tables...")
    main_table = build_main_mechanism_table()
    shortcut_table = build_shortcut_evidence_table()
    calibration_table = build_calibration_winner_table()
    paper_ready = build_paper_ready_summary(
        main_table,
        shortcut_table,
        calibration_table,
    )
    main_table.to_csv(BASE_DIR / "ibcast_main_mechanism_table.csv", index=False)
    shortcut_table.to_csv(BASE_DIR / "ibcast_shortcut_evidence_table.csv", index=False)
    calibration_table.to_csv(BASE_DIR / "ibcast_calibration_winner_table.csv", index=False)
    paper_ready.to_csv(BASE_DIR / "ibcast_paper_ready_summary.csv", index=False)
    export_latex_table(
        paper_ready,
        "ibcast_paper_ready_summary.tex",
        caption=(
            "Mechanism-specific robustness summary under matched missingness rates. "
            "Winner flip, ranking stability, AUROC margin, missingness-indicator shortcut signal, "
            "and winner-level calibration are summarized across 20 seeds."
        ),
        label="tab:ibcast_mechanism_summary",
    )
    print("Done.")
    print("Wrote:")
    print("  - ibcast_main_mechanism_table.csv")
    print("  - ibcast_shortcut_evidence_table.csv")
    print("  - ibcast_calibration_winner_table.csv")
    print("  - ibcast_paper_ready_summary.csv")
    print("  - ibcast_paper_ready_summary.tex")

if __name__ == "__main__":
    main()
