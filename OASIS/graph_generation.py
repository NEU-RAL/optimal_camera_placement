"""
Graph Generation Module

This module provides functions for generating test instances for optimization problems,
including generating information matrices and test graphs.
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("graph_generation")

def weight_graph_lap_from_edge_list(edge_list, num_vertices):
    """
    Create a weighted graph Laplacian from a list of weighted edges.
    
    Args:
        edge_list: List of weighted edges (each with attributes i, j, weight)
        num_vertices: Number of vertices in the graph
        
    Returns:
        sp.spmatrix: The weighted graph Laplacian
    """
    L = sp.lil_matrix((num_vertices, num_vertices))
    
    for edge in edge_list:
        i, j, weight = edge.i, edge.j, edge.weight
        L[i, i] += weight
        L[j, j] += weight
        L[i, j] -= weight
        L[j, i] -= weight
    
    return L.tocsr()

def weight_graph_lap_from_edges(edges, weights, num_vertices):
    """
    Create a weighted graph Laplacian from edges and weights.
    
    Args:
        edges: Array of edges [(i,j), ...]
        weights: Array of weights
        num_vertices: Number of vertices in the graph
        
    Returns:
        sp.spmatrix: The weighted graph Laplacian
    """
    L = sp.lil_matrix((num_vertices, num_vertices))
    
    for (i, j), weight in zip(edges, weights):
        L[i, i] += weight
        L[j, j] += weight
        L[i, j] -= weight
        L[j, i] -= weight
    
    return L.tocsr()

def generate_test_matrices(
    n: int = 100,   # Number of matrices to generate
    m: int = 50,    # Size of each matrix (dimension)
    r_factor: float = 1.25,  # Connectivity radius factor
    Wmax: float = 5.0,  # Maximum edge weight
    seed: int = 42  # Random seed
) -> tuple:
    """
    Generate test matrices for optimization problems using a graph-based approach.
    
    Args:
        n: Number of matrices to generate
        m: Dimension of each matrix
        r_factor: Connectivity radius factor
        Wmax: Maximum edge weight
        seed: Random seed
        
    Returns:
        Tuple containing:
        - List of sparse information matrices
        - Prior information matrix
        - Weights of matrices
    """
    np.random.seed(seed)
    
    # Calculate r based on the number of vertices to ensure connectivity
    r = r_factor * np.sqrt(np.log(m) / (np.pi * m))
    
    logger.info(f"Generating {n} test matrices of size {m}x{m}")
    logger.info(f"Parameters: r={r:.4f}, Wmax={Wmax}")
    
    # Generate a connected geometric random graph
    vertices = np.random.uniform(0, 1, size=(m, 2))
    G = nx.random_geometric_graph(m, r, pos={i: vertices[i] for i in range(m)})
    
    # Make sure the graph is connected
    attempts = 0
    while not nx.is_connected(G) and attempts < 10:
        r *= 1.1
        G = nx.random_geometric_graph(m, r, pos={i: vertices[i] for i in range(m)})
        attempts += 1
    
    if not nx.is_connected(G):
        logger.warning("Could not generate a connected graph. Adding minimal edges to connect components.")
        # Add minimal edges to make the graph connected
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            # Add an edge between a random node in component 0 and component i
            u = np.random.choice(list(components[0]))
            v = np.random.choice(list(components[i]))
            G.add_edge(u, v)
    
    # Assign random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(1, Wmax)
    
    # Get all edges and partition them into n subsets
    all_edges = list(G.edges())
    np.random.shuffle(all_edges)
    
    # Handle case where n > number of edges
    if len(all_edges) < n:
        logger.warning(f"Number of edges ({len(all_edges)}) < number of matrices ({n}). Some matrices will be duplicated.")
        # Duplicate edges to reach n
        all_edges = all_edges * (n // len(all_edges) + 1)
        all_edges = all_edges[:n]
    
    edge_partitions = np.array_split(all_edges, n)
    
    # Generate matrices with different non-zero min eigenvalues
    inf_mats = []
    weights = []
    
    for i in range(n):
        # Create subgraph from this partition
        subgraph = nx.Graph()
        subgraph.add_nodes_from(range(m))  # Add all nodes
        
        # Add the edges from this partition
        for u, v in edge_partitions[i]:
            subgraph.add_edge(u, v, weight=G[u][v]['weight'])
        
        # Generate the Laplacian
        L = nx.laplacian_matrix(subgraph).astype(float)
        
        # Use a random offset to create different minimum eigenvalues
        # This ensures all matrices have different positive min eigenvalues
        min_eig_offset = np.random.uniform(0.05, 1.0)
        
        A = L + min_eig_offset * sp.eye(m)
        
        # Make sure it's symmetric
        A = (A + A.T) / 2
        inf_mats.append(A)
        
        # Store the "weight" of this matrix (can be used for constraints)
        total_weight = sum(G[u][v]['weight'] for u, v in subgraph.edges())
        weights.append(total_weight)
        
        if (i+1) % 20 == 0:
            logger.info(f"Generated {i+1}/{n} matrices")
    
    # Generate initial prior matrix H0 (identity matrix)
    H0 = sp.eye(m)
    
    return inf_mats, H0, np.array(weights)

def generate_pose_graph(
    num_poses: int,
    loop_closure_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Generate a synthetic pose graph for SLAM-like problems.
    
    Args:
        num_poses: Number of poses in the graph
        loop_closure_ratio: Ratio of loop closures to odometry edges
        seed: Random seed
        
    Returns:
        Tuple containing:
        - List of odometry measurements
        - List of loop closure measurements
        - Number of poses
    """
    np.random.seed(seed)
    
    # Create a simple class for edge measurement
    class Measurement:
        def __init__(self, i, j, weight=1.0):
            self.i = i
            self.j = j
            self.weight = weight
    
    # Generate sequential odometry measurements
    odom_measurements = [Measurement(i, i+1, 1.0) for i in range(num_poses-1)]
    
    # Generate random loop closures
    num_loop_closures = int(num_poses * loop_closure_ratio)
    lc_measurements = []
    
    for _ in range(num_loop_closures):
        i = np.random.randint(0, num_poses)
        j = np.random.randint(0, num_poses)
        
        # Ensure i != j and |i-j| > 1 (not adjacent poses)
        if i != j and abs(i-j) > 1:
            # Random weight for loop closures
            weight = np.random.uniform(0.5, 2.0)
            lc_measurements.append(Measurement(min(i,j), max(i,j), weight))
    
    return odom_measurements, lc_measurements, num_poses

def generate_test_problem_constraints(
    n: int,
    weights: np.ndarray = None,
    k: int = None
) -> tuple:
    """
    Generate constraints for test problems.
    
    Args:
        n: Number of variables (matrices/edges)
        weights: Optional weights for the elements (used for budget constraints)
        k: Optional cardinality constraint (number of elements to select)
        
    Returns:
        Tuple containing:
        - Constraint matrix A
        - Constraint bounds b
    """
    constraints = []
    bounds = []
    
    # If k is provided, add cardinality constraint
    if k is not None:
        constraints.append(np.ones(n))
        bounds.append(k)
    
    # If weights are provided, add budget constraint
    if weights is not None:
        # Total weight constraint (allow using 30% of total weight)
        total_weight = np.sum(weights)
        max_weight_allowed = 0.3 * total_weight
        
        constraints.append(weights)
        bounds.append(max_weight_allowed)
        
        # Add a second constraint on squared weights
        squared_weights = weights**2
        total_squared = np.sum(squared_weights)
        max_squared_allowed = 0.3 * total_squared
        
        constraints.append(squared_weights)
        bounds.append(max_squared_allowed)
    
    # Stack all constraints into a matrix
    A = np.vstack(constraints) if constraints else np.zeros((0, n))
    b = np.array(bounds).reshape(-1, 1) if bounds else np.zeros((0, 1))
    
    return A, b

def generate_3d_grid_graph(nx: int, ny: int, nz: int, weight_range=(1.0, 2.0), seed=None):
    """
    Generate a 3D grid graph with random edge weights.
    
    Args:
        nx: Number of points in x dimension
        ny: Number of points in y dimension
        nz: Number of points in z dimension
        weight_range: Range for random edge weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - NetworkX graph
        - Dictionary of node positions (for visualization)
    """
    import networkx as nx
    import numpy as np
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create a 3D grid graph
    G = nx.grid_graph(dim=[nx, ny, nz])
    
    # Generate positions (for visualization)
    pos = {(x, y, z): np.array([x, y, z]) for x in range(nx) for y in range(ny) for z in range(nz)}
    
    # Add random weights to edges
    min_weight, max_weight = weight_range
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(min_weight, max_weight)
    
    return G, pos

def generate_3d_sphere_graph(n_points: int, radius: float = 1.0, 
                           connection_radius: float = None, 
                           weight_range=(1.0, 2.0), seed=None):
    """
    Generate a 3D sphere graph with random edge weights.
    Points are distributed randomly on the surface of a sphere and connected if they
    are within a specified distance of each other.
    
    Args:
        n_points: Number of points on the sphere
        radius: Radius of the sphere
        connection_radius: Distance threshold for connecting points 
                          (default: auto-calculated to ensure connectivity)
        weight_range: Range for random edge weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - NetworkX graph
        - Dictionary of node positions (for visualization)
    """
    import networkx as nx
    import numpy as np
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points on a sphere
    # (using the Gaussian sampling method for uniform distribution on a sphere)
    points = np.random.randn(n_points, 3)
    
    # Normalize points to lie on the sphere's surface
    norms = np.sqrt(np.sum(points**2, axis=1))
    points = points / norms[:, np.newaxis] * radius
    
    # Create the graph
    G = nx.Graph()
    
    # Add nodes with positions
    for i in range(n_points):
        G.add_node(i, pos=points[i])
    
    # Automatically calculate connection radius if not specified
    if connection_radius is None:
        # Heuristic: Use a radius that will likely ensure connectivity
        # based on average nearest neighbor distance
        connection_radius = radius * np.sqrt(4 * np.pi / n_points) * 1.5
    
    # Connect nearby nodes
    min_weight, max_weight = weight_range
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.linalg.norm(points[i] - points[j])
            if dist <= connection_radius:
                weight = np.random.uniform(min_weight, max_weight)
                G.add_edge(i, j, weight=weight, distance=dist)
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        
        # If not all nodes are in the largest component
        if len(largest_component) < n_points:
            logger.warning(f"Original graph was not connected. Connecting components...")
            
            # Connect smaller components to the largest one
            for component in components:
                if component != largest_component:
                    # Find the shortest edge between components
                    min_dist = float('inf')
                    edge_to_add = None
                    
                    for i in component:
                        for j in largest_component:
                            dist = np.linalg.norm(points[i] - points[j])
                            if dist < min_dist:
                                min_dist = dist
                                edge_to_add = (i, j)
                    
                    # Add the edge
                    if edge_to_add:
                        i, j = edge_to_add
                        weight = np.random.uniform(min_weight, max_weight)
                        G.add_edge(i, j, weight=weight, distance=min_dist)
    
    # Create position dictionary for visualization
    pos = {i: points[i] for i in range(n_points)}
    
    return G, pos

def generate_3d_sphere_pose_graph(n_poses: int, radius: float = 10.0,
                                odometry_noise: float = 0.1, 
                                loop_closure_ratio: float = 0.1,
                                loop_closure_max_dist: float = None,
                                seed: int = None):
    """
    Generate a SLAM-like pose graph on a 3D sphere.
    
    This function creates a path that traverses randomly on a sphere,
    with odometry edges between consecutive poses and loop closure edges
    for poses that are close in space but far in trajectory.
    
    Args:
        n_poses: Number of poses
        radius: Radius of the sphere
        odometry_noise: Standard deviation of noise added to odometry measurements
        loop_closure_ratio: Ratio of loop closures to odometry edges
        loop_closure_max_dist: Maximum distance for loop closures (default: 20% of diameter)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - List of odometry measurements (each with attributes i, j, weight)
        - List of loop closure measurements (each with attributes i, j, weight)
        - Array of 3D pose positions (n_poses x 3)
    """
    import networkx as nx
    import numpy as np
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create a simple class for edge measurement
    class Measurement:
        def __init__(self, i, j, weight=1.0):
            self.i = i
            self.j = j
            self.weight = weight
    
    # Generate points on a sphere with a continuous path
    sphere_graph, pos = generate_3d_sphere_graph(
        n_points=n_poses * 5,  # Generate more points than needed
        radius=radius,
        connection_radius=radius * 0.5,  # Create a reasonably dense graph
        seed=seed
    )
    
    # Find a spanning tree to ensure connectivity
    mst = nx.minimum_spanning_tree(sphere_graph)
    
    # Find a long path in the graph (for odometry)
    longest_path = []
    for start in mst.nodes():
        # Find the longest path starting from this node
        paths = nx.single_source_shortest_path(mst, start)
        furthest_node = max(paths.items(), key=lambda x: len(x[1]))
        if len(furthest_node[1]) > len(longest_path):
            longest_path = furthest_node[1]
        
        # Stop once we find a path of sufficient length
        if len(longest_path) >= n_poses:
            break
    
    # Trim to desired length
    path = longest_path[:n_poses]
    
    # Extract positions
    positions = np.array([pos[node] for node in path])
    
    # Create odometry measurements
    odom_measurements = []
    for i in range(n_poses - 1):
        # Compute distance between consecutive poses
        dist = np.linalg.norm(positions[i+1] - positions[i])
        
        # Add noise to weight
        weight = 1.0 / (dist + np.random.normal(0, odometry_noise))
        weight = max(0.1, weight)  # Ensure positive weight
        
        odom_measurements.append(Measurement(i, i+1, weight))
    
    # Create loop closure measurements
    num_loop_closures = int(n_poses * loop_closure_ratio)
    
    # Default max distance for loop closure if not specified
    if loop_closure_max_dist is None:
        loop_closure_max_dist = 2 * radius * 0.2  # 20% of diameter
    
    lc_measurements = []
    
    # Calculate pairwise distances
    dists = {}
    for i in range(n_poses):
        for j in range(i+2, n_poses):  # Skip adjacent poses
            dist = np.linalg.norm(positions[i] - positions[j])
            dists[(i, j)] = dist
    
    # Sort by distance
    close_pairs = [(i, j) for (i, j), dist in dists.items() 
                  if dist < loop_closure_max_dist]
    
    # Randomly select pairs for loop closures
    if close_pairs:
        selected_pairs = np.random.choice(
            len(close_pairs),
            size=min(num_loop_closures, len(close_pairs)),
            replace=False
        )
        
        for idx in selected_pairs:
            i, j = close_pairs[idx]
            dist = dists[(i, j)]
            
            # Weight inversely proportional to distance
            weight = 1.0 / (dist + np.random.normal(0, odometry_noise))
            weight = max(0.1, weight)  # Ensure positive weight
            
            lc_measurements.append(Measurement(i, j, weight))
    
    return odom_measurements, lc_measurements, positions