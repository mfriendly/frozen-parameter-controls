"""
AQ-437 adjacency anatomy and bandwidth comparison.

Reads the AirQuality dataset from tsl and writes all outputs under --outdir.

Outputs:
    {outdir}/anatomy_summary.csv          (adjacency statistics at the default bandwidth)
    {outdir}/bandwidth_ablation.csv       (AirQuality bandwidth table)
    {outdir}/aq_block_diagonal.pdf        (adjacency reordered by city)
    {outdir}/aq_block_diagonal.png        (preview)
    {outdir}/distances_intra_inter.csv    (intra- and inter-city distance statistics)
    {outdir}/anatomy_log.txt              (full numeric log)

Run:
    python scripts/aq_anatomy.py --outdir results/aq_anatomy --tau 0.1
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import scipy.sparse.csgraph as csgraph
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
import mpl_config  # noqa: F401

from tsl.datasets import AirQuality


def kernel_adj(dist, theta, tau):
    K = np.exp(-(dist ** 2) / (theta ** 2))
    K[K < tau] = 0.0
    np.fill_diagonal(K, 0.0)
    return K


def training_adjacency(aq, config_path):
    """Adjacency used in training: tsl get_connectivity with the connectivity settings of the dataset config."""
    connectivity = yaml.safe_load(open(config_path))["dataset"]["connectivity"]
    A = np.asarray(aq.get_connectivity(**connectivity, layout="dense"), dtype=np.float64)
    assert np.array_equal(A > 0, (A > 0).T), "training adjacency pattern is not symmetric"
    assert not np.any(np.diag(A)), "training adjacency has a nonzero diagonal"
    return A, connectivity


def assign_city_ids(dist, eps):
    """Cluster stations into cities via single-linkage on a precomputed distance matrix."""
    n = dist.shape[0]
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < eps:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def adjacency_metrics(A, city_ids):
    nz = A > 0
    n_components, _ = csgraph.connected_components(A, directed=False)
    deg = nz.sum(axis=1)
    mean_deg = deg.mean()

    same_city = city_ids[:, None] == city_ids[None, :]
    iu = np.triu_indices_from(A, k=1)
    w_pairs = A[iu]
    same_pairs = same_city[iu]
    intra = w_pairs[(w_pairs > 0) & same_pairs]
    inter = w_pairs[(w_pairs > 0) & (~same_pairs)]
    intra_med = float(np.median(intra)) if intra.size else 0.0
    inter_med = float(np.median(inter)) if inter.size else 0.0
    ratio = intra_med / inter_med if inter_med > 0 else float("inf")

    return {
        "n_nodes": int(A.shape[0]),
        "n_edges": int(nz.sum() // 2),
        "n_components": int(n_components),
        "mean_degree": float(mean_deg),
        "median_degree": float(np.median(deg)),
        "max_degree": int(deg.max()),
        "intra_n": int(intra.size),
        "inter_n": int(inter.size),
        "intra_median": intra_med,
        "inter_median": inter_med,
        "intra_inter_ratio": float(ratio) if np.isfinite(ratio) else None,
        "n_isolated": int((deg == 0).sum()),
    }

def reordered_heatmap(A, city_ids, out_path):
    order = np.argsort(city_ids, kind="stable")
    Ar = A[order][:, order]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(Ar > 0, cmap="Greys", interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
# def reordered_heatmap(A, city_ids, out_path):
#     order = np.argsort(city_ids, kind="stable")
#     Ar = A[order][:, order]
#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.imshow(Ar > 0, cmap="Greys", interpolation="nearest", aspect="equal")
#     ax.set_xticks([]); ax.set_yticks([])
#     # ax.set_title(f"AQ-437 adjacency (reordered by city, {len(np.unique(city_ids))} cities)",
#     #              fontsize=20)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=300)
#     fig.savefig(out_path.replace(".pdf", ".png"), dpi=300)
#     plt.close(fig)


def write_csv(rows, fieldnames, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def percentile_theta(dist, p):
    iu = np.triu_indices_from(dist, k=1)
    return float(np.percentile(dist[iu], p))


def per_city_std(dist, city_ids):
    """Bandwidth = mean of within-city std (excluding singletons)."""
    stds = []
    for c in np.unique(city_ids):
        idx = np.where(city_ids == c)[0]
        if idx.size < 2:
            continue
        sub = dist[np.ix_(idx, idx)]
        iu = np.triu_indices_from(sub, k=1)
        if iu[0].size:
            stds.append(float(np.std(sub[iu])))
    return float(np.mean(stds)) if stds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tau", type=float, default=0.1, help="kernel sparsification threshold")
    ap.add_argument("--city_eps_km", type=float, default=80.0,
                    help="single-linkage distance for city clustering, km")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "anatomy_log.txt"
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    aq = AirQuality()
    dist = np.asarray(aq.dist, dtype=np.float64)
    n = dist.shape[0]
    log(f"n_stations = {n}")
    log(f"dist matrix: shape={dist.shape}, dtype={dist.dtype}")
    log(f"dist stats: min={dist.min():.4f}, max={dist.max():.4f}, "
        f"median={np.median(dist):.4f}")
    assert n == 437, f"expected 437 stations, got {n}"
    assert dist.shape == (n, n), f"expected square distance matrix, got {dist.shape}"

    city_ids = assign_city_ids(dist, eps=args.city_eps_km)
    n_cities = int(np.unique(city_ids).size)
    log(f"n_cities (single-linkage @ {args.city_eps_km}) = {n_cities}")

    config_path = Path(__file__).resolve().parent.parent / "config" / "dataset" / "airquality.yaml"
    A_default, connectivity = training_adjacency(aq, config_path)
    default_theta = float(np.std(aq.dist[:36, :36]))
    log(f"dataset default adjacency: get_connectivity({connectivity}) from {config_path.name}; "
        f"theta used by tsl = std(dist[:36, :36]) = {default_theta:.6f}")

    iu_full = np.triu_indices(n, k=1)

    bandwidths = {
        "aq437_std": float(np.std(dist[iu_full])),
        "percentile_p25": percentile_theta(dist, 25),
        "percentile_p50": percentile_theta(dist, 50),
        "percentile_p95": percentile_theta(dist, 95),
        "per_city_std": per_city_std(dist, city_ids),
    }

    log("\nbandwidth choices (same units as aq.dist):")
    for k, v in bandwidths.items():
        log(f"  {k:30s} = {v:.6f}")

    m = adjacency_metrics(A_default, city_ids)
    m["bandwidth"] = "dataset default"
    m["theta"] = default_theta
    rows = [m]
    log(f"\n[dataset default] theta={default_theta:.6f}")
    for k, v in m.items():
        if k not in ("bandwidth", "theta"):
            log(f"  {k:20s} = {v}")
    for name, theta in bandwidths.items():
        if theta <= 0:
            continue
        A = kernel_adj(dist, theta, args.tau)
        m = adjacency_metrics(A, city_ids)
        m["bandwidth"] = name
        m["theta"] = theta
        rows.append(m)
        log(f"\n[{name}] theta={theta:.6f}")
        for k, v in m.items():
            if k in ("bandwidth", "theta"):
                continue
            log(f"  {k:20s} = {v}")

    write_csv(
        rows,
        fieldnames=[
            "bandwidth", "theta",
            "n_nodes", "n_edges", "n_components", "n_isolated",
            "mean_degree", "median_degree", "max_degree",
            "intra_n", "inter_n", "intra_median", "inter_median", "intra_inter_ratio",
        ],
        path=outdir / "bandwidth_ablation.csv",
    )

    summary = adjacency_metrics(A_default, city_ids)
    summary["theta"] = default_theta
    summary["n_cities"] = n_cities
    write_csv(
        [summary],
        fieldnames=[
            "theta", "n_nodes", "n_cities", "n_edges", "n_components", "n_isolated",
            "mean_degree", "median_degree", "max_degree",
            "intra_n", "inter_n", "intra_median", "inter_median", "intra_inter_ratio",
        ],
        path=outdir / "anatomy_summary.csv",
    )

    iu = np.triu_indices_from(A_default, k=1)
    same = (city_ids[:, None] == city_ids[None, :])[iu]
    pos = A_default[iu] > 0
    raw_rows = [
        {"pair_type": "intra_city", "n_pairs": int((pos & same).sum()),
         "mean_d": float(dist[iu][same].mean()),
         "max_d": float(dist[iu][same].max()),
         "median_d_squared_over_theta_squared":
             float(np.median(dist[iu][same] ** 2 / default_theta ** 2)),
         "median_kernel_value":
             float(np.median(np.exp(-dist[iu][same] ** 2 / default_theta ** 2)))},
        {"pair_type": "inter_city", "n_pairs": int((pos & (~same)).sum()),
         "mean_d": float(dist[iu][~same].mean()),
         "max_d": float(dist[iu][~same].max()),
         "median_d_squared_over_theta_squared":
             float(np.median(dist[iu][~same] ** 2 / default_theta ** 2)),
         "median_kernel_value":
             float(np.median(np.exp(-dist[iu][~same] ** 2 / default_theta ** 2)))},
    ]
    write_csv(
        raw_rows,
        fieldnames=["pair_type", "n_pairs", "mean_d", "max_d",
                    "median_d_squared_over_theta_squared", "median_kernel_value"],
        path=outdir / "distances_intra_inter.csv",
    )

    reordered_heatmap(A_default, city_ids, str(outdir / "aq_block_diagonal.pdf"))

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nartifacts written to {outdir}")


if __name__ == "__main__":
    main()