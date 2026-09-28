"""Static graph rewiring baselines.

Ports of FoSR, SDRF, BORF, GTR to a unified interface for ST forecasting.
All methods operate on the spatial adjacency graph as a preprocessing step.

Dependencies:
  - FoSR: numba
  - SDRF: numba
  - BORF: numba, ot (POT), networkx
  - GTR:  torch, networkx
"""
import numpy as np
import torch
import networkx as nx
from numba import jit, prange


# =============================================================================
# FoSR  (Karhadkar et al., LoG 2023)
# Greedy spectral gap maximization via Fiedler vector
# =============================================================================

@jit(nopython=True)
def _fosr_compute_degrees(edge_index, num_nodes):
    degrees = np.zeros(num_nodes)
    m = edge_index.shape[1]
    for i in range(m):
        degrees[edge_index[0, i]] += 1
    return degrees


@jit(nopython=True)
def _fosr_adj_mul(edge_index, x):
    n = x.size
    y = np.zeros(n)
    m = edge_index.shape[1]
    for i in range(m):
        y[edge_index[0, i]] += x[edge_index[1, i]]
    return y


@jit(nopython=True)
def _fosr_choose_edge(x, edge_index, degrees):
    n = x.size
    m = edge_index.shape[1]
    y = x / ((degrees + 1) ** 0.5)
    products = np.outer(y, y)
    for i in range(m):
        products[edge_index[0, i], edge_index[1, i]] = np.inf
    for i in range(n):
        products[i, i] = np.inf
    idx = np.argmin(products)
    return (idx % n, idx // n)


@jit(nopython=True)
def _fosr_add_edge(edge_index, u, v):
    new_edge = np.array([[u, v], [v, u]])
    return np.concatenate((edge_index, new_edge), axis=1)


@jit(nopython=True)
def _fosr_core(edge_index, edge_type, x, num_iterations, initial_power_iters):
    n = np.max(edge_index) + 1
    degrees = _fosr_compute_degrees(edge_index, n)
    # Power iteration for Fiedler vector
    for _ in range(initial_power_iters):
        x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5) / np.sum(degrees)
        y = x + _fosr_adj_mul(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
        x = y / np.linalg.norm(y)
    # Greedy edge addition
    for _ in range(num_iterations):
        i, j = _fosr_choose_edge(x, edge_index, degrees)
        edge_index = _fosr_add_edge(edge_index, i, j)
        degrees[i] += 1
        degrees[j] += 1
        edge_type = np.append(edge_type, 1)
        edge_type = np.append(edge_type, 1)
        # Update Fiedler vector
        x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5) / np.sum(degrees)
        y = x + _fosr_adj_mul(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
        x = y / np.linalg.norm(y)
    return edge_index, edge_type


def fosr_rewire(edge_index_np, num_nodes, num_iterations=20,
                initial_power_iters=10):
    """FoSR: First-order Spectral Rewiring.

    Args:
        edge_index_np: (2, M) numpy int array.
        num_nodes: int.
        num_iterations: number of edges to add.
        initial_power_iters: power iteration warmup steps.

    Returns:
        edge_index: (2, M') numpy int array (M' = M + 2*num_iterations).
        edge_type:  (M',) numpy int array (0=original, 1=added).
    """
    x = 2 * np.random.random(num_nodes) - 1
    edge_type = np.zeros(edge_index_np.shape[1], dtype=np.int64)
    return _fosr_core(edge_index_np.astype(np.int64), edge_type, x,
                      num_iterations, initial_power_iters)


# =============================================================================
# SDRF  (Topping et al., ICLR 2022)
# Stochastic Discrete Ricci Flow via balanced Forman curvature
# =============================================================================

@jit(nopython=True)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    for i in prange(N):
        for j in prange(N):
            if A[i, j] == 0:
                C[i, j] = 0
                continue
            if d_in[i] > d_out[j]:
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]
            if d_max * d_min == 0:
                C[i, j] = 0
                continue
            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP
                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP
            C[i, j] = ((2 / d_max) + (2 / d_min) - 2
                        + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j])
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_curvature(A, C=None):
    """Compute balanced Forman curvature for all edges."""
    N = A.shape[0]
    A2 = A @ A
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = np.zeros((N, N))
    _balanced_forman_curvature(A, A2, d_in, d_out, N, C)
    return C


@jit(nopython=True)
def _bfc_post_delta(A, A2, d_in_x, d_out_y, N, D, x, y,
                    i_neighbors, j_neighbors, dim_i, dim_j):
    """Efficient curvature change computation after adding edge (i,j)."""
    for I in prange(dim_i):
        for J in prange(dim_j):
            i = i_neighbors[I]
            j = j_neighbors[J]
            if (i == j) or (A[i, j] != 0):
                D[I, J] = -1000
                continue
            d_in = d_in_x
            d_out = d_out_y
            if j == x:
                d_in += 1
            elif i == y:
                d_out += 1
            if d_in * d_out == 0:
                D[I, J] = 0
                continue
            d_max = max(d_in, d_out)
            d_min = min(d_in, d_out)
            A2_x_y = A2[x, y]
            if (x == i) and (A[j, y] != 0):
                A2_x_y += A[j, y]
            elif (y == j) and (A[x, i] != 0):
                A2_x_y += A[x, i]
            sharp_ij = 0
            lambda_ij = 0
            for z in range(N):
                A_z_y = A[z, y]
                A_x_z = A[x, z]
                A2_z_y = A2[z, y]
                A2_x_z = A2[x, z]
                if (z == i) and (y == j):
                    A_z_y += 1
                if (x == i) and (z == j):
                    A_x_z += 1
                if (z == i) and (A[j, y] != 0):
                    A2_z_y += A[j, y]
                if (x == i) and (A[j, z] != 0):
                    A2_x_z += A[j, z]
                if (y == j) and (A[z, i] != 0):
                    A2_z_y += A[z, i]
                if (z == j) and (A[x, i] != 0):
                    A2_x_z += A[x, i]
                TMP = A_z_y * (A2_x_z - A_x_z) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP
                TMP = A_x_z * (A2_z_y - A_z_y) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP
            D[I, J] = ((2 / d_max) + (2 / d_min) - 2
                        + (2 / d_max + 1 / d_min) * A2_x_y * A[x, y])
            if lambda_ij > 0:
                D[I, J] += sharp_ij / (d_max * lambda_ij)


def _softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()


def sdrf_rewire(edge_index_np, num_nodes, num_iterations=20,
                curvature='bfc', remove_edges=True, removal_bound=0.5,
                tau=1):
    """SDRF: Stochastic Discrete Ricci Flow.

    Args:
        edge_index_np: (2, M) numpy int array (undirected edges).
        num_nodes: int.
        num_iterations: number of rewiring iterations.
        curvature: 'bfc' for balanced Forman curvature.
        remove_edges: whether to remove positively curved edges.
        removal_bound: curvature threshold for edge removal.
        tau: softmax temperature for candidate selection.

    Returns:
        edge_index: (2, M') numpy int array.
        edge_type:  (M',) numpy int array (0=original, 1=added).
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in zip(edge_index_np[0], edge_index_np[1]):
        if i != j:
            A[i, j] = A[j, i] = 1.0

    G = nx.from_numpy_array(A)
    C = np.zeros((num_nodes, num_nodes))
    edge_type_list = [0] * len(G.edges()) * 2  # undirected

    for _ in range(num_iterations):
        can_add = True
        balanced_forman_curvature(A, C=C)

        # Find minimum curvature edge
        ix_min = C.argmin()
        x = ix_min // num_nodes
        y = ix_min % num_nodes

        x_neighbors = list(G.neighbors(x)) + [x]
        y_neighbors = list(G.neighbors(y)) + [y]

        # Find candidates
        candidates = [(i, j) for i in x_neighbors for j in y_neighbors
                       if (i != j) and (not G.has_edge(i, j))]

        if len(candidates):
            A2 = A @ A
            d_in = A[:, x].sum()
            d_out = A[y].sum()
            D = np.zeros((len(x_neighbors), len(y_neighbors)))
            _bfc_post_delta(A, A2, d_in, d_out, num_nodes, D, x, y,
                            np.array(x_neighbors), np.array(y_neighbors),
                            len(x_neighbors), len(y_neighbors))
            improvements = [
                (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)]
                for (i, j) in candidates
            ]
            k, l = candidates[
                np.random.choice(len(candidates),
                                 p=_softmax(np.array(improvements), tau=tau))
            ]
            G.add_edge(k, l)
            A[k, l] = A[l, k] = 1.0
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax()
            x2 = ix_max // num_nodes
            y2 = ix_max % num_nodes
            if C[x2, y2] > removal_bound and G.has_edge(x2, y2):
                G.remove_edge(x2, y2)
                A[x2, y2] = A[y2, x2] = 0.0
            elif not can_add:
                break

    # Convert back to edge_index
    edges = np.array(list(G.edges())).T
    edge_index_out = np.concatenate([edges, edges[::-1]], axis=1)

    # Mark edge types: reconstruct from original
    orig_set = set()
    for i in range(edge_index_np.shape[1]):
        orig_set.add((int(edge_index_np[0, i]), int(edge_index_np[1, i])))
    edge_type_out = np.array([
        0 if (edge_index_out[0, i], edge_index_out[1, i]) in orig_set else 1
        for i in range(edge_index_out.shape[1])
    ], dtype=np.int64)

    return edge_index_out, edge_type_out


# =============================================================================
# BORF  (Nguyen et al., ICML 2023)
# Batched Ollivier-Ricci Flow
# =============================================================================

def borf_rewire(edge_index_np, num_nodes, loops=10, batch_add=4,
                batch_remove=2):
    """BORF: Batched Ollivier-Ricci Flow.

    Requires: pip install POT (optimal transport)

    Args:
        edge_index_np: (2, M) numpy int array.
        num_nodes: int.
        loops: number of rewiring iterations.
        batch_add: edges to add per iteration.
        batch_remove: edges to remove per iteration.

    Returns:
        edge_index: (2, M') numpy int array.
        edge_type:  (M',) numpy int array.
    """
    import ot  # POT library for optimal transport

    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in zip(edge_index_np[0], edge_index_np[1]):
        if i != j:
            A[i, j] = A[j, i] = 1.0
    G = nx.from_numpy_array(A)

    orig_set = set()
    for i in range(edge_index_np.shape[1]):
        orig_set.add((int(edge_index_np[0, i]), int(edge_index_np[1, i])))

    for _ in range(loops):
        # Compute Ollivier-Ricci curvature for all edges
        curvatures = {}
        for (u, v) in G.edges():
            u_nbrs = list(G.neighbors(u))
            v_nbrs = list(G.neighbors(v))
            du, dv = len(u_nbrs), len(v_nbrs)
            if du == 0 or dv == 0:
                curvatures[(u, v)] = 0.0
                continue
            # Uniform distributions on neighborhoods
            mu = np.ones(du) / du
            mv = np.ones(dv) / dv
            # Distance matrix between neighborhoods
            dist = np.zeros((du, dv))
            for ii, ni in enumerate(u_nbrs):
                for jj, nj in enumerate(v_nbrs):
                    dist[ii, jj] = nx.shortest_path_length(G, ni, nj)
            cost = ot.emd2(mu, mv, dist)
            curvatures[(u, v)] = 1.0 - cost

        sorted_edges = sorted(curvatures.keys(), key=lambda e: curvatures[e])
        most_neg = sorted_edges[:batch_add]
        most_pos = sorted_edges[-batch_remove:]

        # Add edges at most negatively curved locations
        for (u, v) in most_neg:
            u_nbrs = list(G.neighbors(u))
            v_nbrs = list(G.neighbors(v))
            best_pair = None
            best_dist = -1
            for ni in u_nbrs:
                for nj in v_nbrs:
                    if ni != nj and not G.has_edge(ni, nj):
                        d = nx.shortest_path_length(G, ni, nj)
                        if d > best_dist:
                            best_dist = d
                            best_pair = (ni, nj)
            if best_pair is not None:
                G.add_edge(*best_pair)

        # Remove most positively curved edges
        for (u, v) in most_pos:
            if G.has_edge(u, v) and G.degree(u) > 1 and G.degree(v) > 1:
                G.remove_edge(u, v)

    # Convert back
    edges = np.array(list(G.edges())).T
    if edges.size == 0:
        return edge_index_np, np.zeros(edge_index_np.shape[1], dtype=np.int64)
    edge_index_out = np.concatenate([edges, edges[::-1]], axis=1)
    edge_type_out = np.array([
        0 if (edge_index_out[0, i], edge_index_out[1, i]) in orig_set else 1
        for i in range(edge_index_out.shape[1])
    ], dtype=np.int64)
    return edge_index_out, edge_type_out


# =============================================================================
# GTR  (Black et al., ICML 2023)
# Greedy Total Resistance minimization
# =============================================================================

def gtr_rewire(edge_index_np, num_nodes, num_edges=20):
    """GTR: Greedy Total Resistance rewiring.

    Adds edges that maximally decrease total effective resistance.
    Uses Woodbury identity for efficient pseudoinverse updates.

    Args:
        edge_index_np: (2, M) numpy int array.
        num_nodes: int.
        num_edges: number of edges to add.

    Returns:
        edge_index: (2, M') numpy int array.
        edge_type:  (M',) numpy int array.
    """
    # Build adjacency and Laplacian
    A = np.zeros((num_nodes, num_nodes))
    for i, j in zip(edge_index_np[0], edge_index_np[1]):
        if i != j:
            A[i, j] = A[j, i] = 1.0

    G = nx.from_numpy_array(A)
    L = torch.tensor(nx.laplacian_matrix(G).todense().astype(np.float64),
                     dtype=torch.float64)
    pinv = torch.linalg.pinv(L, hermitian=True)
    sq_pinv = pinv @ pinv

    # Edge mask: 1 where edge can be added
    edge_mask = (~L.bool()).float()
    # Component mask: only add within connected components
    comp_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    for comp in nx.connected_components(G):
        nodes = list(comp)
        for ni in nodes:
            for nj in nodes:
                comp_mask[ni, nj] = 1.0

    added_edges = []

    for _ in range(num_edges):
        pinv_diag = torch.diagonal(pinv)
        R = pinv_diag.unsqueeze(0) + pinv_diag.unsqueeze(1) - 2 * pinv
        sq_diag = torch.diagonal(sq_pinv)
        B = sq_diag.unsqueeze(0) + sq_diag.unsqueeze(1) - 2 * sq_pinv
        # Decrease in total resistance when adding edge (s,t)
        diff = B / (1 + R)
        masked = diff * edge_mask * comp_mask
        flat_idx = torch.argmax(masked).item()
        s, t = divmod(flat_idx, num_nodes)

        added_edges.append((s, t))
        added_edges.append((t, s))

        # Woodbury update for pinv
        v = pinv[:, s] - pinv[:, t]
        eff_res = v[s] - v[t]
        pinv = pinv - (1 / (1 + eff_res)) * torch.outer(v, v)

        # Woodbury update for sq_pinv
        x_vec = torch.zeros(num_nodes, dtype=torch.float64)
        x_vec[s], x_vec[t] = 1, -1
        y_vec = L[:, s] - L[:, t]
        U = torch.stack([x_vec, y_vec + x_vec], dim=1)
        V = torch.stack([y_vec + x_vec, x_vec])
        left = sq_pinv @ U
        center = torch.inverse(torch.eye(2, dtype=torch.float64) + V @ sq_pinv @ U)
        right = V @ sq_pinv
        sq_pinv = sq_pinv - left @ center @ right

        # Update Laplacian
        L[s, s] += 1
        L[s, t] -= 1
        L[t, s] -= 1
        L[t, t] += 1
        # Update edge mask
        edge_mask[s, t] = 0
        edge_mask[t, s] = 0

    # Reconstruct edge_index
    new_edges = np.array(added_edges, dtype=np.int64).T if added_edges else np.zeros((2, 0), dtype=np.int64)
    edge_index_out = np.concatenate([edge_index_np, new_edges], axis=1)
    edge_type_out = np.concatenate([
        np.zeros(edge_index_np.shape[1], dtype=np.int64),
        np.ones(new_edges.shape[1], dtype=np.int64),
    ])
    return edge_index_out, edge_type_out
