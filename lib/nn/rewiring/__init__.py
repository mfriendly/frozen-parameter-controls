"""Graph rewiring module.

Provides static rewiring baselines and learned bilevel rewiring.

Static methods (preprocessing):
  - fosr:  spectral gap maximization (Karhadkar et al., LoG 2023)
  - sdrf:  stochastic discrete Ricci flow (Topping et al., ICLR 2022)
  - borf:  batched Ollivier-Ricci flow (Nguyen et al., ICML 2023)
  - gtr:   greedy total resistance (Black et al., ICML 2023)

Learned methods (during training):
  - bilevel: task-aware bilevel rewiring
"""
import numpy as np
import scipy.sparse as sp
from tsl import logger

from .baselines import fosr_rewire, sdrf_rewire, borf_rewire, gtr_rewire


def adj_to_edge_index(adj):
    """Convert scipy sparse or dense adjacency to (edge_index_np, num_nodes).

    Args:
        adj: scipy sparse matrix, numpy array, or tuple (edge_index, edge_weight).

    Returns:
        edge_index_np: (2, M) numpy int64 array.
        num_nodes: int.
    """
    if isinstance(adj, tuple):
        # (edge_index, edge_weight) format
        edge_index, edge_weight = adj
        if hasattr(edge_index, 'numpy'):
            edge_index = edge_index.numpy()
        num_nodes = int(edge_index.max()) + 1
        return edge_index.astype(np.int64), num_nodes

    if sp.issparse(adj):
        adj_coo = adj.tocoo()
        edge_index = np.stack([adj_coo.row, adj_coo.col]).astype(np.int64)
        return edge_index, adj.shape[0]

    if isinstance(adj, np.ndarray):
        rows, cols = np.nonzero(adj)
        edge_index = np.stack([rows, cols]).astype(np.int64)
        return edge_index, adj.shape[0]

    # torch_sparse SparseTensor
    if hasattr(adj, 'coo'):
        row, col, val = adj.coo()
        edge_index = np.stack([row.numpy(), col.numpy()]).astype(np.int64)
        return edge_index, adj.size(0)

    raise TypeError(f"Unsupported adjacency type: {type(adj)}")


def edge_index_to_sparse(edge_index_np, num_nodes, edge_weight=None):
    """Convert (2, M) edge_index to scipy CSR sparse matrix."""
    if edge_weight is None:
        edge_weight = np.ones(edge_index_np.shape[1])
    return sp.csr_matrix(
        (edge_weight, (edge_index_np[0], edge_index_np[1])),
        shape=(num_nodes, num_nodes)
    )


def apply_rewiring(adj, cfg):
    """Apply static graph rewiring as preprocessing.

    Args:
        adj: adjacency in any supported format (scipy sparse, numpy, tuple).
        cfg: OmegaConf rewiring config with at least `method` key.

    Returns:
        adj_rewired: adjacency in scipy CSR format.
        edge_type: (M,) numpy array (0=original, 1=added).
        stats: dict with rewiring statistics.
    """
    method = cfg.method
    if method == 'none':
        edge_index_np, num_nodes = adj_to_edge_index(adj)
        edge_type = np.zeros(edge_index_np.shape[1], dtype=np.int64)
        return adj, edge_type, {'method': 'none', 'edges_added': 0}

    edge_index_np, num_nodes = adj_to_edge_index(adj)
    n_orig = edge_index_np.shape[1]

    if method == 'fosr':
        ei_new, et_new = fosr_rewire(
            edge_index_np, num_nodes,
            num_iterations=cfg.num_iterations,
            initial_power_iters=cfg.initial_power_iters,
        )
    elif method == 'sdrf':
        ei_new, et_new = sdrf_rewire(
            edge_index_np, num_nodes,
            num_iterations=cfg.num_iterations,
            curvature=cfg.curvature,
            remove_edges=cfg.remove_edges,
            removal_bound=cfg.removal_bound,
            tau=cfg.tau,
        )
    elif method == 'borf':
        ei_new, et_new = borf_rewire(
            edge_index_np, num_nodes,
            loops=cfg.loops,
            batch_add=cfg.batch_add,
            batch_remove=cfg.batch_remove,
        )
    elif method == 'gtr':
        ei_new, et_new = gtr_rewire(
            edge_index_np, num_nodes,
            num_edges=cfg.num_edges,
        )
    elif method in ('bilevel', 'drew'):
        # These are not preprocessing methods — handled at model/training level
        edge_type = np.zeros(edge_index_np.shape[1], dtype=np.int64)
        return adj, edge_type, {'method': method, 'edges_added': 0}
    else:
        raise ValueError(f"Unknown rewiring method: {method}")

    n_new = ei_new.shape[1]
    n_added = int((et_new == 1).sum())
    logger.info(f"Rewiring [{method}]: {n_orig} -> {n_new} edges "
                f"(+{n_added} added)")

    adj_rewired = edge_index_to_sparse(ei_new, num_nodes)
    return adj_rewired, et_new, {
        'method': method,
        'edges_original': n_orig,
        'edges_total': n_new,
        'edges_added': n_added,
    }
