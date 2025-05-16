
import numpy as np
import scipy.sparse as sp
import networkx as nx
import logging
from collections import namedtuple, defaultdict

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("graph_generation")

# Quadruplet definition
Quadruplet = namedtuple('Quadruplet', ['matrix_idx', 'row', 'col', 'value'])


def _quadruplets_from_edge_pairs(matrix_idx: int,
                                 edge_pairs: np.ndarray,
                                 weights: np.ndarray,
                                 num_vertices: int) -> list[Quadruplet]:
    """
    Build Laplacian quadruplets from edge pairs and weights.
    Each undirected edge contributes two off-diagonals and accumulates on the diagonal.
    """
    diag = np.zeros(num_vertices, dtype=float)
    quads: list[Quadruplet] = []
    for (i, j), w in zip(edge_pairs, weights):
        diag[i] += w
        diag[j] += w
        quads.append(Quadruplet(matrix_idx, i, j, -w))
        quads.append(Quadruplet(matrix_idx, j, i, -w))
    # diagonal entries
    for i, val in enumerate(diag):
        if val != 0:
            quads.append(Quadruplet(matrix_idx, i, i, val))
    return quads


def weight_graph_lap_from_edge_list_quad(edge_list, num_vertices, matrix_idx=0):
    """
    Quadruplets from a list of .i/.j/.weight edges.
    """
    pairs = np.array([[e.i, e.j] for e in edge_list], dtype=int)
    weights = np.array([e.weight for e in edge_list], dtype=float)
    return _quadruplets_from_edge_pairs(matrix_idx, pairs, weights, num_vertices)


def weight_graph_lap_from_edges_quad(edges: list[tuple[int,int]],
                                     weights: np.ndarray,
                                     num_vertices: int,
                                     matrix_idx=0):
    """
    Quadruplets from arrays of (i,j) and corresponding weights.
    """
    pairs = np.array(edges, dtype=int)
    return _quadruplets_from_edge_pairs(matrix_idx, pairs, weights, num_vertices)


def generate_test_matrices(
    n: int = 100,
    m: int = 50,
    r_factor: float = 1.25,
    Wmax: float = 5.0,
    seed: int = 42
) -> tuple[list[Quadruplet], list[Quadruplet], np.ndarray]:
    """
    Generate n random Laplacian matrices (with random offsets) as quadruplets,
    plus a prior identity matrix (index n).
    Returns (all_quads, prior_quads, weights).
    """
    rng = np.random.default_rng(seed)
    r = r_factor * np.sqrt(np.log(m)/(np.pi*m))
    logger.info(f"Generating {n} test matrices of size {m}x{m} as quadruplets")

    # build geometric graph
    points = rng.random((m,2))
    G = nx.random_geometric_graph(m, r, pos={i:points[i] for i in range(m)})
    attempts = 0
    while not nx.is_connected(G) and attempts<10:
        r *= 1.1
        G = nx.random_geometric_graph(m, r, pos={i:points[i] for i in range(m)})
        attempts+=1
    if not nx.is_connected(G):
        logger.warning("Graph disconnected: adding edges to connect.")
        comps = list(nx.connected_components(G))
        for comp in comps[1:]:
            u = rng.choice(list(comps[0])); v = rng.choice(list(comp))
            G.add_edge(u, v, weight=rng.uniform(1,Wmax))

    for u,v in G.edges(): G[u][v]['weight']=rng.uniform(1,Wmax)
    all_edges = list(G.edges())
    if len(all_edges)<n:
        all_edges *= (n//len(all_edges)+1)
    rng.shuffle(all_edges)
    partitions = np.array_split(all_edges, n)

    all_quads: list[Quadruplet] = []
    weights = np.zeros(n, dtype=float)
    for i, part in enumerate(partitions):
        # collect edge pairs and weights
        pairs = np.array(part, dtype=int)
        ws = np.array([G[u][v]['weight'] for u,v in part], dtype=float)
        # Laplacian quads
        quads = _quadruplets_from_edge_pairs(i, pairs, ws, m)
        # add small offset to diagonal to ensure PD
        offset = rng.uniform(15,20)
        for j in range(m):
            quads.append(Quadruplet(i, j, j, offset))
        all_quads.extend(quads)
        weights[i] = ws.sum()
        if (i+1)%20==0: logger.info(f"Generated {i+1}/{n} matrices")

    # prior = identity
    prior_quads = [Quadruplet(n, i, i, 1.0) for i in range(m)]
    return all_quads, prior_quads, weights


def quadruplets_to_sparse(quadruplets: list[Quadruplet],
                          num_matrices: int,
                          matrix_size: int) -> list[sp.csr_matrix]:
    """
    Group quadruplets by matrix_idx and build sparse CSR matrices.
    """
    groups: dict[int,list[Quadruplet]] = defaultdict(list)
    for q in quadruplets:
        groups[q.matrix_idx].append(q)

    mats: list[sp.csr_matrix] = []
    for idx in range(num_matrices):
        qs = groups.get(idx, [])
        if qs:
            rows = [q.row for q in qs]
            cols = [q.col for q in qs]
            data = [q.value for q in qs]
            mat = sp.coo_matrix((data,(rows,cols)), shape=(matrix_size,matrix_size))
            mats.append(mat.tocsr())
        else:
            mats.append(sp.csr_matrix((matrix_size,matrix_size)))
    return mats


def generate_test_problem_constraints(
    n: int,
    weights: np.ndarray = None,
    k: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build constraint matrix A and bounds b for cardinality k and weight budgets.
    """
    constraints, bounds = [], []
    if k is not None:
        constraints.append(np.ones(n))
        bounds.append(k)
    if weights is not None:
        total = weights.sum()
        constraints.append(weights)
        bounds.append(0.3*total)
        sq = weights**2
        constraints.append(sq)
        bounds.append(0.3*sq.sum())
    A = np.vstack(constraints) if constraints else np.zeros((0,n))
    b = np.array(bounds).reshape(-1,1) if bounds else np.zeros((0,1))
    return A, b
