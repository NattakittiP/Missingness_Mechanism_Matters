import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd

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

sys.path.insert(0, str(_HERE))

sys.path.insert(0, str(_PROJECT_ROOT / "PHASE 4"))
from ibcast_mechanism_sweep import (
    aggregate_folds,
    compute_winners_ibcast,
    build_envelope,
    run_friedman_tests,
    run_nemenyi_posthoc,
    compute_margin_gap,
    compute_per_feature_missingness_leak,
    aggregate_per_feature_leak,
    plot_fig1_envelope_flip,
    plot_fig2_kendall_heatmap,
    plot_fig3_auroc_boxplot,
    plot_fig4_mechanism_divergence,
    plot_fig5_margin_gap,
    plot_fig6_per_feature_leak,
    OUT_DIR as DEFAULT_OUT_DIR,
    FIG_DIR as DEFAULT_FIG_DIR,
    NEM_DIR as DEFAULT_NEM_DIR,
)

try:
    from jcsse_audit_runner_tqdm_hardened import load_dataset_A
    _HAS_DATASET_A = True

except Exception as _e:
    _HAS_DATASET_A = False
    _DATASET_A_ERR = str(_e)

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def read_csv_with_progress(path: Path, chunksize: int = 50_000) -> pd.DataFrame:
    try:
        total_bytes = os.path.getsize(path)
    except OSError:
        total_bytes = None
    chunks = []
    rows = 0
    bar = tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Reading {path.name}",
        leave=True,
    )
    pos_after = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunks.append(chunk)
        rows += len(chunk)
        if total_bytes is not None:
            try:
                pos_now = chunk.memory_usage(deep=True).sum()
            except Exception:
                pos_now = 0
        bar.set_postfix(rows=f"{rows:,}", refresh=False)
        if total_bytes is not None:
            new_pos = min(total_bytes, int(rows * (total_bytes / max(rows + 1, 1))))
            delta = new_pos - pos_after
            pos_after = new_pos
            if delta > 0:
                bar.update(delta)
    if total_bytes is not None and pos_after < total_bytes:
        bar.update(total_bytes - pos_after)
    bar.close()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw",
        type=str,
        default=str(DEFAULT_OUT_DIR / "metrics_raw.csv"),
        help="Path to metrics_raw.csv (default: D:\\IBCAST_MECHANISM_SWEEP\\metrics_raw.csv)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory (default: D:\\IBCAST_MECHANISM_SWEEP)",
    )
    args = ap.parse_args()
    raw_path = Path(args.raw)
    out_dir  = Path(args.out_dir)
    fig_dir  = out_dir / "figures"
    nem_dir  = out_dir / "nemenyi_posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    nem_dir.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"metrics_raw.csv not found at: {raw_path}\n"
            "Pass --raw <path> if the file is elsewhere."
        )
    tqdm.write(f"[{_now()}] ══ IBCAST post-process recovery ══")
    stages = [
        "Load metrics_raw.csv",
        "Aggregate folds",
        "Compute winners",
        "Build envelope",
        "Friedman tests",
        "Nemenyi posthoc",
        "Generate figures (1-4)",
        "Margin gap analysis (Fig 5)",
        "Per-feature MNAR leak (Fig 6)",
    ]
    master = tqdm(total=len(stages), desc="IBCAST post-process", position=0, leave=True)
    master.set_description(f"[1/7] {stages[0]}")
    tqdm.write(f"[{_now()}] Reading raw metrics: {raw_path}")
    raw_df = read_csv_with_progress(raw_path)
    tqdm.write(f"  rows: {len(raw_df):,}")
    n_before = len(raw_df)
    raw_df = raw_df.dropna(subset=["auroc"])
    n_dropped = n_before - len(raw_df)
    if n_dropped:
        tqdm.write(f"  dropped {n_dropped:,} rows with NaN AUROC (failed fits).")
    expected = 3 * 7 * 2 * 20 * 5 * 5
    tqdm.write(
        f"  coverage: {len(raw_df):,} / {expected:,} ideal fold rows "
        f"({100.0 * len(raw_df) / expected:.1f}%)"
    )
    keys = ["mechanism", "rate", "split", "model", "seed"]
    completed = raw_df.groupby(keys).size().reset_index(name="n_folds")
    tqdm.write(
        f"  completed (mech,rate,split,model,seed) cells: {len(completed):,} "
        f"/ ideal {3*7*2*5*20:,}"
    )
    master.update(1)
    master.set_description(f"[2/7] {stages[1]}")
    tqdm.write(f"[{_now()}] Aggregating folds…")
    summary_df = aggregate_folds(raw_df)
    summary_df.to_csv(out_dir / "summary_by_setting.csv", index=False)
    tqdm.write(f"  → summary_by_setting.csv  ({len(summary_df):,} rows)")
    master.update(1)
    master.set_description(f"[3/7] {stages[2]}")
    tqdm.write(f"[{_now()}] Computing winners…")
    winners_df = compute_winners_ibcast(summary_df)
    winners_df.to_csv(out_dir / "winners_by_seed.csv", index=False)
    tqdm.write(f"  → winners_by_seed.csv  ({len(winners_df):,} rows)")
    try:
        win_pivot = (
            winners_df
            .groupby(["mechanism", "rate", "split"])["winner_model"]
            .agg(lambda s: s.value_counts().to_dict())
            .unstack("split")
        )
        tqdm.write("\n  winners by (mechanism,rate,split):")
        tqdm.write(win_pivot.to_string())
    except Exception as e:
        tqdm.write(f"  [warn] could not build winners pivot: {e}")
    master.update(1)
    master.set_description(f"[4/7] {stages[3]}")
    tqdm.write(f"\n[{_now()}] Building robustness envelope…")
    envelope_df = build_envelope(winners_df, summary_df)
    envelope_df.to_csv(out_dir / "envelope_by_mechanism.csv", index=False)
    tqdm.write(f"  → envelope_by_mechanism.csv  ({len(envelope_df):,} rows)")
    try:
        flip_pivot = envelope_df.pivot_table(
            index=["mechanism", "split"], columns="rate", values="flip_pct"
        )
        tqdm.write("\n  flip % by (mechanism,split) × rate:")
        tqdm.write(flip_pivot.round(1).to_string())
    except Exception:
        pass
    master.update(1)
    master.set_description(f"[5/7] {stages[4]}")
    tqdm.write(f"\n[{_now()}] Running Friedman tests…")
    friedman_df = run_friedman_tests(summary_df)
    friedman_df.to_csv(out_dir / "friedman_results.csv", index=False)
    if not friedman_df.empty:
        sig = friedman_df[friedman_df["significant_p05"]]
        tqdm.write(f"  Friedman p<0.05: {len(sig)}/{len(friedman_df)} conditions")
    master.update(1)
    master.set_description(f"[6/7] {stages[5]}")
    tqdm.write(f"[{_now()}] Running Nemenyi posthoc (significant conditions only)…")
    run_nemenyi_posthoc(summary_df, nem_dir)
    n_nem = len(list(nem_dir.glob("*.csv")))
    tqdm.write(f"  → {n_nem} nemenyi files in {nem_dir}")
    master.update(1)
    master.set_description(f"[7/9] {stages[6]}")
    tqdm.write(f"\n[{_now()}] Generating figures (1-4)…")
    fig_specs = [
        (lambda: plot_fig1_envelope_flip(envelope_df, fig_dir),
         "fig1_envelope_flip"),
        (lambda: plot_fig2_kendall_heatmap(envelope_df, fig_dir),
         "fig2_kendall_tau_heatmap"),
        (lambda: plot_fig3_auroc_boxplot(summary_df, 0.30, fig_dir),
         "fig3_auroc_boxplot_rate30"),
        (lambda: plot_fig4_mechanism_divergence(envelope_df, fig_dir),
         "fig4_mechanism_divergence"),
    ]
    fig_bar = tqdm(fig_specs, desc="figures", position=1, leave=False)
    for fn, label in fig_bar:
        fig_bar.set_postfix_str(label, refresh=True)
        try:
            fn()
            tqdm.write(f"  ✓ {label}")
        except Exception as e:
            tqdm.write(f"  [WARN] {label}: {e}")
    fig_bar.close()
    master.update(1)
    master.set_description(f"[8/9] {stages[7]}")
    tqdm.write(f"\n[{_now()}] Margin gap analysis (top-1 vs top-2 AUROC)…")
    try:
        margin_by_seed, margin_summary = compute_margin_gap(summary_df)
        margin_by_seed.to_csv(out_dir / "margin_gap_by_seed.csv", index=False)
        margin_summary.to_csv(out_dir / "margin_gap_summary.csv", index=False)
        tqdm.write(
            f"  → margin_gap_by_seed.csv     ({len(margin_by_seed):,} rows)"
        )
        tqdm.write(
            f"  → margin_gap_summary.csv     ({len(margin_summary):,} rows)"
        )
        try:
            piv = margin_summary.pivot_table(
                index=["mechanism", "split"],
                columns="rate",
                values="gap_mean",
            )
            tqdm.write("\n  mean AUROC margin (top-1 − top-2) by (mech,split) × rate:")
            tqdm.write((piv * 100).round(2).to_string() + "  (×100, i.e. AUROC pp)")
        except Exception:
            pass
        plot_fig5_margin_gap(margin_summary, fig_dir)
        tqdm.write(f"  ✓ fig5_margin_gap")
    except Exception as e:
        tqdm.write(f"  [WARN] margin gap stage failed: {e}")
    master.update(1)
    master.set_description(f"[9/9] {stages[8]}")
    tqdm.write(f"\n[{_now()}] Per-feature MNAR leak analysis…")
    if not _HAS_DATASET_A:
        tqdm.write(
            f"  [SKIP] Could not import load_dataset_A: {_DATASET_A_ERR}\n"
            "         Place this script under "
            "<project>/IBCAST_MECHANISM_SWEEP/Code/ so 'PHASE 4' is reachable."
        )
    else:
        try:
            tqdm.write(f"  Loading Dataset A (MIMIC-IV v3.1)…")
            X, y, _groups, num_cols, _cat_cols = load_dataset_A()
            tqdm.write(
                f"    N={len(y):,}  pos={int(y.sum())}  "
                f"({100*y.mean():.2f}%)  num_feats={len(num_cols)}"
            )
            tqdm.write(f"  Computing per-feature indicator AUROC "
                       f"(3 mechs × 7 rates × 20 seeds × {len(num_cols)} feats)…")
            per_feat_raw = compute_per_feature_missingness_leak(
                X, y, num_cols,
            )
            per_feat_raw.to_csv(out_dir / "per_feature_leak_by_seed.csv", index=False)
            tqdm.write(
                f"    → per_feature_leak_by_seed.csv  ({len(per_feat_raw):,} rows)"
            )
            per_feat_agg = aggregate_per_feature_leak(per_feat_raw)
            per_feat_agg.to_csv(out_dir / "per_feature_leak_summary.csv", index=False)
            tqdm.write(
                f"    → per_feature_leak_summary.csv  ({len(per_feat_agg):,} rows)"
            )
            try:
                top = (
                    per_feat_agg[
                        (per_feat_agg["mechanism"] == "MNAR") &
                        (per_feat_agg["rate"] == 0.30)
                    ]
                    .sort_values("auroc_mean", ascending=False)
                    .head(10)[["feature", "auroc_mean", "missing_rate_mean"]]
                )
                tqdm.write("\n  top 10 MNAR-leaky features @ rate=0.30:")
                tqdm.write(top.round(3).to_string(index=False))
            except Exception:
                pass
            plot_fig6_per_feature_leak(per_feat_agg, fig_dir, top_k=12, rate_for_topk=0.30)
            tqdm.write(f"  ✓ fig6_per_feature_mnar_leak")
        except Exception as e:
            tqdm.write(f"  [WARN] per-feature leak stage failed: {e}")
    master.update(1)
    master.close()
    tqdm.write(f"\n{'═'*68}")
    tqdm.write(f"  IBCAST post-process — DONE")
    tqdm.write(f"{'═'*68}")
    tqdm.write(f"  Output dir : {out_dir}")
    tqdm.write(f"  Figures    : {fig_dir}")
    tqdm.write(f"  Nemenyi    : {nem_dir}")

if __name__ == "__main__":
    main()
