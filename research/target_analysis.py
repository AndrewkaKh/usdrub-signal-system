"""Target distribution analysis for IV delta.

Computes summary statistics for the raw delta-IV series and its
rolling-normalised (z-score) transform, then saves results to CSV.

Run via:  python model_runner.py --analyze-target

Outputs (written to data/exports/)
-----------------------------------
target_analysis_stats.csv    — scalar summary statistics
target_analysis_series.csv   — per-row delta and z-score time series
target_analysis_quantiles.csv — threshold recommendations at various flat-zone sizes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_EXPORTS_DIR = Path(__file__).parents[1] / 'data' / 'exports'


def analyze_target(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    """Analyse the distribution of raw delta IV and its z-score variant.

    Parameters
    ----------
    df         : prepared dataset; must contain 'date', 'iv_1m',
                 and 'target_delta_iv_1m' columns.
    output_dir : directory for CSV output (default: data/exports/).

    Returns
    -------
    dict with keys 'stats', 'stats_path', 'series_path', 'quantile_path'.
    """
    if output_dir is None:
        output_dir = _EXPORTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    delta = df['target_delta_iv_1m'].dropna()

    # ── 1. Basic statistics ───────────────────────────────────────────────────
    stats: dict = {
        'n': int(len(delta)),
        'mean': float(delta.mean()),
        'std': float(delta.std()),
        'min': float(delta.min()),
        'max': float(delta.max()),
        'median': float(delta.median()),
        'q05': float(delta.quantile(0.05)),
        'q10': float(delta.quantile(0.10)),
        'q25': float(delta.quantile(0.25)),
        'q75': float(delta.quantile(0.75)),
        'q90': float(delta.quantile(0.90)),
        'q95': float(delta.quantile(0.95)),
        'skewness': float(delta.skew()),
        'kurtosis': float(delta.kurt()),
    }

    # ── 2. Fraction of near-zero changes at fixed thresholds ─────────────────
    abs_thresholds = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]
    for t in abs_thresholds:
        key = f'frac_abs_lt_{int(t * 1000):03d}mpp'
        stats[key] = round(float((delta.abs() < t).mean()), 4)

    # ── 3. Rolling-normalised z-scores ────────────────────────────────────────
    # past_delta = iv(t) - iv(t-1): causal, known at time t
    past_delta = df['iv_1m'].diff()
    rolling_std = past_delta.rolling(20, min_periods=5).std()

    valid_mask = rolling_std.notna() & df['target_delta_iv_1m'].notna()
    z = df.loc[valid_mask, 'target_delta_iv_1m'] / rolling_std[valid_mask]

    stats.update({
        'z_n': int(len(z)),
        'z_mean': float(z.mean()),
        'z_std': float(z.std()),
        'z_q05': float(z.quantile(0.05)),
        'z_q10': float(z.quantile(0.10)),
        'z_q25': float(z.quantile(0.25)),
        'z_q75': float(z.quantile(0.75)),
        'z_q90': float(z.quantile(0.90)),
        'z_q95': float(z.quantile(0.95)),
    })

    # Fraction near zero in z-score units
    for z_thresh in [0.25, 0.50, 0.75, 1.00, 1.50]:
        key = f'frac_abs_z_lt_{str(z_thresh).replace(".", "_")}'
        stats[key] = round(float((z.abs() < z_thresh).mean()), 4)

    # ── 4. Save scalar stats ──────────────────────────────────────────────────
    stats_path = output_dir / 'target_analysis_stats.csv'
    pd.DataFrame([stats]).to_csv(stats_path, index=False, sep=';')

    # ── 5. Per-row time series ────────────────────────────────────────────────
    series_df = df[['date', 'iv_1m', 'target_delta_iv_1m']].copy()
    series_df['past_delta'] = past_delta.values
    series_df['rolling_std_20'] = rolling_std.values
    # z-score: NaN where rolling_std not yet available
    series_df['z_score'] = (series_df['target_delta_iv_1m'] / rolling_std).values

    series_path = output_dir / 'target_analysis_series.csv'
    series_df.to_csv(series_path, index=False, sep=';')

    # ── 6. Threshold recommendation table ────────────────────────────────────
    # For each candidate flat-zone size (as fraction of all observations),
    # show what absolute threshold and z-threshold that corresponds to.
    rows = []
    for flat_frac in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        # Symmetric: flat = |delta| < threshold
        # threshold = quantile of |delta| at (flat_frac) level
        abs_thresh = float(delta.abs().quantile(flat_frac))
        z_thresh = float(z.abs().quantile(flat_frac)) if len(z) > 0 else float('nan')
        empirical_flat = float((delta.abs() < abs_thresh).mean())
        rows.append({
            'flat_zone_target_frac': flat_frac,
            'abs_threshold': round(abs_thresh, 5),
            'z_threshold': round(z_thresh, 3),
            'empirical_flat_frac': round(empirical_flat, 4),
            'n_flat': int((delta.abs() < abs_thresh).sum()),
            'n_down': int((delta < -abs_thresh).sum()),
            'n_up': int((delta > abs_thresh).sum()),
        })

    quantile_path = output_dir / 'target_analysis_quantiles.csv'
    quantile_df = pd.DataFrame(rows)
    quantile_df.to_csv(quantile_path, index=False, sep=';')

    return {
        'stats': stats,
        'stats_path': str(stats_path),
        'series_path': str(series_path),
        'quantile_path': str(quantile_path),
    }


def print_analysis(result: dict) -> None:
    """Pretty-print analysis results to stdout."""
    s = result['stats']
    print('\n=== Target Delta-IV Distribution ===')
    print(f'  Observations : {s["n"]}')
    print(f'  Mean         : {s["mean"]:+.5f}  ({s["mean"]*100:+.3f}%)')
    print(f'  Std          : {s["std"]:.5f}  ({s["std"]*100:.3f}%)')
    print(f'  Median       : {s["median"]:+.5f}')
    print(f'  [Q5, Q95]    : [{s["q05"]:+.5f}, {s["q95"]:+.5f}]')
    print(f'  Skewness     : {s["skewness"]:+.3f}')
    print(f'  Kurtosis     : {s["kurtosis"]:+.3f}')

    print('\n--- Near-zero fraction (|delta| < threshold) ---')
    keys = [k for k in s if k.startswith('frac_abs_lt_')]
    for k in sorted(keys):
        thresh_mpp = int(k.split('_')[-1].replace('mpp', ''))
        thresh_pct = thresh_mpp / 10
        print(f'  < {thresh_mpp:3d} mpp ({thresh_pct:.1f}bps): {s[k]*100:.1f}%')

    print('\n--- Z-score statistics (delta / rolling_std20) ---')
    print(f'  Z std        : {s["z_std"]:.3f}  (ideal ~ 1.0)')
    print(f'  Z [Q5, Q95]  : [{s["z_q05"]:+.2f}, {s["z_q95"]:+.2f}]')
    print('\n--- Near-zero fraction in z-score units ---')
    z_keys = [k for k in s if k.startswith('frac_abs_z_lt_')]
    for k in sorted(z_keys):
        thresh_str = k.replace('frac_abs_z_lt_', '').replace('_', '.')
        print(f'  |z| < {thresh_str}: {s[k]*100:.1f}%')

    print('\n--- Threshold recommendation (flat-zone sizing) ---')
    print(f'  {"flat%":>6}  {"abs_thresh":>10}  {"z_thresh":>8}  {"n_down":>6}  {"n_flat":>6}  {"n_up":>6}')
    import csv
    with open(result['quantile_path'], newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            print(
                f'  {float(row["flat_zone_target_frac"])*100:5.0f}%  '
                f'{float(row["abs_threshold"]):.5f}     '
                f'{float(row["z_threshold"]):7.3f}   '
                f'{row["n_down"]:>6}  {row["n_flat"]:>6}  {row["n_up"]:>6}'
            )

    print(f'\nSaved: {result["stats_path"]}')
    print(f'       {result["series_path"]}')
    print(f'       {result["quantile_path"]}')
