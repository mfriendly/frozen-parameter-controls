"""Collect all experiment results from run dirs into a single CSV.

Config keys are extracted by flattening the full YAML; every config field
becomes a column.

Usage:
    python collect_results.py              # collect all, save to results.csv
    python collect_results.py --seeded-only  # skip runs with seed=null
"""
import argparse
import math
import re
import yaml
import pandas as pd
from pathlib import Path

LOGS_ROOT = Path(__file__).parent / "logs"
OUT_CSV = Path(__file__).parent / "results.csv"

# Config subtrees to flatten (prefix → dot-separated columns).
# Keys not matching any prefix are stored with their full path.
# Internal/verbose subtrees to skip entirely:
SKIP_PREFIXES = {
    "hydra", "tags", "default_tags", "run_tags",
    "run.dir", "run.name",   # run_path already captured separately
}

# Config keys to drop entirely from output
DROP_KEYS = {"workers", "num_threads"}

# Metric columns to drop (redundant or unscaled)
DROP_METRICS = {"test_loss", "test_mae_first", "test_mae_last"}

# Desired leading columns (in order); remaining config cols follow alphabetically
LEAD_COLS = [
    "run_time", "seed",
    "dataset.name", "model.name", "rewiring.method",
    "test_mae", "test_rmse", "test_mape",
    "args_string", "run_path",
]


def flatten_config(cfg: dict, parent: str = "", sep: str = ".") -> dict:
    """Recursively flatten nested dict with dot-separated keys."""
    out = {}
    for k, v in cfg.items():
        full_key = f"{parent}{sep}{k}" if parent else k
        if any(full_key == s or full_key.startswith(s + sep)
               for s in SKIP_PREFIXES):
            continue
        if full_key in DROP_KEYS:
            continue
        if isinstance(v, dict):
            out.update(flatten_config(v, full_key, sep))
        else:
            out[full_key] = v
    return out


def read_config(config_path: Path) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw or {}


def extract_run_time(run_dir: Path) -> str:
    """Extract KST timestamp from run dir path (YYYY-MM-DD/HH-MM-SS pattern)."""
    parts = run_dir.parts
    for i, p in enumerate(parts):
        if re.match(r"\d{4}-\d{2}-\d{2}$", p) and i + 1 < len(parts):
            t = parts[i + 1]
            if re.match(r"\d{2}-\d{2}-\d{2}$", t):
                date_str = p.replace("-", "")[2:]   # YYMMDD
                time_str = t.replace("-", "")        # HHMMSS
                return f"{date_str}_{time_str}"
    return ""


def _label_distilled_runs(flat: dict) -> None:
    """External fixed adj, no bilevel policy — label distilled, not vanilla (none)."""
    if flat.get("rewiring.method") != "none":
        return
    adj = flat.get("adj_load")
    if adj is None or adj == "":
        return
    if isinstance(adj, str) and adj.strip().lower() in ("null", "none", "~"):
        return
    flat["rewiring.method"] = "distilled"


def build_args_string(flat: dict) -> str:
    """Build a compact human-readable config summary from flattened config."""
    want = [
        ("dataset.name",                   "dataset"),
        ("model.name",                     "model"),
        ("seed",                           "seed"),
        ("rewiring.method",                "rewiring"),
        ("adj_load",                       "adj_load"),
        ("model.hparams.hidden_size",      "hidden"),
        ("model.hparams.n_layers",         "n_layers"),
        ("rewiring.bilevel_method",        "bilevel_method"),
        ("rewiring.inner_steps",           "inner_steps"),
        ("rewiring.outer_lr",              "outer_lr"),
        ("rewiring.outer_warmup_epochs",   "warmup_epochs"),
        ("rewiring.policy_num_new_edges",  "num_new_edges"),
        ("rewiring.policy_type",           "policy_type"),
        ("rewiring.disable_reweight",      "disable_reweight"),
        ("rewiring.freeze_phi",            "freeze_phi"),
        ("rewiring.outer_loss",            "outer_loss"),
        ("epochs",                         "epochs"),
        ("window",                         "window"),
    ]
    parts = []
    for cfg_key, label in want:
        v = flat.get(cfg_key)
        if v is None:
            continue
        parts.append(f"{label}={v}")
    return " ".join(parts)


def extract_test_metrics(metrics_csv: Path) -> dict:
    df = pd.read_csv(metrics_csv)
    test_cols = [c for c in df.columns if c.startswith("test_")]
    if not test_cols:
        return {}
    test_row = df[test_cols].dropna(how="all")
    if test_row.empty:
        return {}
    row = test_row.iloc[-1]
    out = {c: float(row[c]) for c in test_cols
           if not pd.isna(row[c]) and c not in DROP_METRICS}
    if "test_mse" in out:
        out["test_rmse"] = math.sqrt(out.pop("test_mse"))
    return out


def _build_row(config_path: Path, skip_null_seed: bool = False):
    """Build a single result row from a run directory. Returns None if invalid."""
    run_dir = config_path.parent
    metrics_path = run_dir / "lightning_logs" / "version_0" / "metrics.csv"

    if not metrics_path.exists():
        return None
    test_metrics = extract_test_metrics(metrics_path)
    if not test_metrics:
        return None

    cfg = read_config(config_path)
    raw_seed = cfg.get("seed")
    if skip_null_seed and raw_seed is None:
        return None

    flat = flatten_config(cfg)
    _label_distilled_runs(flat)

    # Normalise seed
    seed = "none" if flat.get("seed") is None else int(flat["seed"])
    flat["seed"] = seed

    row = {"run_time": extract_run_time(run_dir)}
    row.update(flat)
    row["args_string"] = build_args_string(flat)
    row["run_path"] = str(run_dir.relative_to(LOGS_ROOT))
    row.update(test_metrics)
    return row


def _sort_and_save(df: pd.DataFrame):
    sort_keys = [c for c in ["dataset.name", "model.name", "rewiring.method", "seed"]
                 if c in df.columns]
    df = df.sort_values(sort_keys).reset_index(drop=True)

    # Build final column order:
    # 1. LEAD_COLS that exist
    # 2. remaining config cols (sorted, excluding test_* and already-placed)
    # 3. any extra test_* not already in LEAD_COLS
    lead = [c for c in LEAD_COLS if c in df.columns]
    placed = set(lead)
    test_extra = sorted(c for c in df.columns
                        if c.startswith("test_") and c not in placed)
    rest = sorted(c for c in df.columns
                  if c not in placed and not c.startswith("test_"))
    df = df[lead + rest + test_extra]
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    return df


def append_run(run_dir: Path):
    """Append a single completed run to results.csv (called after each experiment)."""
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return
    row = _build_row(config_path)
    if row is None:
        return

    new_df = pd.DataFrame([row])

    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV, dtype={"seed": str})
        if "run_path" in existing.columns and row["run_path"] in existing["run_path"].values:
            return
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df

    df = _sort_and_save(df)
    print(f"[collect] Appended run → {OUT_CSV} (total {len(df)} rows)")


def collect(skip_null_seed: bool = False):
    rows = []
    for config_path in sorted(LOGS_ROOT.rglob("config.yaml")):
        row = _build_row(config_path, skip_null_seed=skip_null_seed)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No results found.")
        return

    df = _sort_and_save(pd.DataFrame(rows))
    print(f"Saved {len(df)} rows → {OUT_CSV}")

    show_cols = [c for c in ["dataset.name", "model.name", "rewiring.method", "seed",
                              "test_mae", "run_time"] if c in df.columns]
    print(df[show_cols].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeded-only", action="store_true",
                        help="Skip runs where seed was null in config")
    args = parser.parse_args()
    collect(skip_null_seed=args.seeded_only)
