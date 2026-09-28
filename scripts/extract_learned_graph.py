"""Extract learned adjacency A_phi from bilevel checkpoint and save as .npy.

Loads the checkpoint, rebuilds the RelativeReweightPolicy, runs forward()
to produce A_phi, and saves it.

Usage:
    python scripts/extract_learned_graph.py \
        --ckpt logs/bilevel/mptcn_diff/pems04/.../epoch=91-step=192280.ckpt \
        --out learned_graphs/pems04_bilevel_seed42.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsl.metrics.torch import metrics as _tsl_metrics
_safe_classes = [getattr(_tsl_metrics, c) for c in dir(_tsl_metrics)
                 if isinstance(getattr(_tsl_metrics, c, None), type)]
torch.serialization.add_safe_globals(_safe_classes)

from experiments.run_realworld_rewired import ThresholdMAPE
torch.serialization.add_safe_globals([ThresholdMAPE])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']

    adj_dense = state['adj_dense']
    N = adj_dense.shape[0]
    print(f"adj_dense: {adj_dense.shape}, nnz={int((adj_dense > 0).sum())}")

    policy_keys = [k for k in state if k.startswith('policy.')]
    print(f"Policy keys: {len(policy_keys)}")

    from lib.nn.rewiring.policy import RelativeReweightPolicy, EdgeReweightPolicy

    num_new_edges = 0
    if 'policy.add_head.0.weight' in state:
        num_new_edges = 20
    disable_reweight = False

    policy_state = {}
    for k, v in state.items():
        if k.startswith('policy.'):
            policy_state[k[len('policy.'):]] = v

    use_relative = 'score_head.0.weight' in policy_state
    if use_relative:
        print("Detected: RelativeReweightPolicy")
        policy = RelativeReweightPolicy(
            num_nodes=N, adj_dense=adj_dense, seq_len=12,
            hidden_dim=64, temperature=1.0,
            num_new_edges=num_new_edges, disable_reweight=disable_reweight)
    else:
        print("Detected: EdgeReweightPolicy")
        policy = EdgeReweightPolicy(
            num_nodes=N, adj_dense=adj_dense, seq_len=12,
            hidden_dim=64, temperature=1.0,
            num_new_edges=num_new_edges)

    policy.load_state_dict(policy_state)
    policy.eval()

    with torch.no_grad():
        adj_s, adj_t, scores_s, scores_t = policy(adj_dense)

    adj_phi = adj_s.numpy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, adj_phi)

    nnz = int((adj_phi > 1e-6).sum())
    orig_nnz = int((adj_dense.numpy() > 0).sum())
    diff = np.abs(adj_phi - adj_dense.numpy())
    diff_at_orig = diff[adj_dense.numpy() > 0]

    print(f"Saved: {args.out}")
    print(f"  shape={adj_phi.shape}, nnz={nnz} (orig={orig_nnz}, added={nnz - orig_nnz})")
    print(f"  weight range: [{adj_phi[adj_phi > 1e-6].min():.6f}, {adj_phi.max():.6f}]")
    print(f"  mean abs diff at original edges: {diff_at_orig.mean():.6f}")
    print(f"  max abs diff: {diff_at_orig.max():.6f}")


if __name__ == '__main__':
    main()
