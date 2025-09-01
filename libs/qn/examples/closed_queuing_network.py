import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from libs.qn.examples.controller import cpu_threshold_controller, cpu_hpa, constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork

def example1_system_mu() -> np.ndarray:
    return np.array([4.0])

def example1() -> ClosedQueuingNetwork:
    stations = 1
    
    P = np.array([
        [0]
    ])
    
    entry_p = np.array([1.0])
    
    mu = np.array([3.6])
    
    max_cores = np.array([16.0])
    min_cores = np.array([1.0])
    
    network = ClosedQueuingNetwork(
        stations, P, entry_p, mu, max_cores, min_cores, 500, 1
    )
    
    network.set_controllers(
        [constant_controller(network, 0, network.max_users)] + 
        [cpu_hpa(network, 1)]
        #[cpu_threshold_controller(network, 1, 0.3, 0.7, 1)]
    )
    
    return network

def example2_system_mu() -> np.ndarray:
    return np.array([
        8, 
        4, 
        4
    ])

def example2() -> ClosedQueuingNetwork:
    stations = 3

    P = np.array([
        [0, 0.6, 0.4],
        [0.3, 0, 0],
        [0.2, 0, 0]
    ])

    entry_p = np.array([
        1.0,
        0.0,
        0.0
    ])

    mu = np.array([
        8,
        4,
        4
    ])
    
    max_cores = np.array([
        8.0,
        8.0,
        8.0
    ])
    
    min_cores = np.array([
        1.0,
        1.0,
        1.0
    ])
    
    max_users = 80
    
    skewness = 15
    
    network = ClosedQueuingNetwork(
        stations, P, entry_p, mu, max_cores, min_cores, max_users, 1, skewness
    )

    return network

def example7_sparse() -> ClosedQueuingNetwork:
    stations = 7

    P = np.array([
        [0.0,  0.2,  0.1,  0.2,  0.0,  0.4,  0.1], 
        [0.0,  0.0,  0.0,  0.3,  0.0,  0.0,  0.3], 
        [0.2,  0.0,  0.0,  0.0,  0.6,  0.0,  0.0], 
        [0.0,  0.0,  0.0,  0.0,  0.2,  0.0,  0.0], 
        [0.0,  0.4,  0.2,  0.0,  0.0,  0.0,  0.0], 
        [0.25, 0.0,  0.25, 0.0,  0.3,  0.0,  0.0], 
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.8,  0.0]  
    ])

    # Entry probabilities (all users enter at station 0, for example).
    entry_p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Service rates for each of the 7 stations.
    mu = np.array([10.0, 1.5, 2.0, 6.0, 3.5, 5.0, 6.0])

    # Core constraints (min and max).
    max_cores = np.array([16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0])
    min_cores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Create the closed queuing network (e.g., 500 users total, 1 iteration).
    network = ClosedQueuingNetwork(
        stations,
        P,
        entry_p,
        mu,
        max_cores,
        min_cores,
        100,  # total users
        1     # assignment iterations
    )

    return network

def random_qn(stations) -> ClosedQueuingNetwork:

    probabilities = generate_matrix(stations + 1, k = 2)
    entry_p = probabilities[0, 1:]
    P = probabilities[1:, 1:]

    mu = np.random.uniform(1.0, 10.0, size=stations)

    max_cores = np.array([8.0]*stations)
    min_cores = np.ones(stations)
    
    skewness = 15
    max_users = 80
    think_time = np.random.uniform(0.5, 2.0)

    network = ClosedQueuingNetwork(
        stations,
        P,
        entry_p,
        mu,
        max_cores,
        min_cores,
        max_users,  
        think_time,
        skewness
    )

    return network

def generate_matrix(n, k=10, seed=None):
    assert k >= 1 and k < n, "Cannot enforce zero diagonal if k >= n"
    rng = np.random.default_rng(seed)
    P = lil_matrix((n, n))
    
    # Step 1: Strongly connected via random cycle (avoiding self-loops)
    perm = rng.permutation(n)
    for i in range(n):
        src = perm[i]
        dst = perm[(i + 1) % n]
        if src != dst:
            P[src, dst] = 1.0  # placeholder value

    # Step 2: Add extra random edges, avoiding self-loops and duplicates
    for i in range(n):
        existing = set(P.rows[i])
        existing.add(i)  # block diagonal
        candidates = list(set(range(n)) - existing)
        needed = np.random.randint(0, k)   # Randomly choose how many edges to add
        if needed > 0 and len(candidates) >= needed:
            new_cols = rng.choice(candidates, size=needed, replace=False)
            for j in new_cols:
                P[i, j] = 1.0

    # Step 3: Assign random weights and normalize rows
    for i in range(n):
        row_indices = P.rows[i]
        weights = rng.random(len(row_indices))
        weights /= weights.sum()
        for idx, val in zip(row_indices, weights):
            P[i, idx] = val

    P_dense = P.toarray()
    np.fill_diagonal(P_dense, 0.0)
    P_dense /= P_dense.sum(axis=1, keepdims=True) 
    P_rounded = np.round(P_dense, 2)

    return P_rounded

def acmeair_qn():
    stations = 9
    
    P = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # entry
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # auth
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # book flights
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # cancel booking
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # get reward miles
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # query flights
        [2/3, 0.0, 0.0, 0.0, 1/3, 0.0, 0.0, 0.0, 0.0, 0.0],  # update miles
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # update profile
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # validate id
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # view profile
    ])

    P[0][[1, 2, 3, 5, 7, 9]] = 1/6

    mu = np.random.uniform(1, 8.0, size=stations)
    
    rts = np.array([103.7, 101.5, 72.6, 31.8, 56.8, 27.5, 124.6, 52.3, 92.8])
    mu = 1 / (rts*5/1000)
    
    print(mu)

    max_cores = np.array([4.0] * stations)
    min_cores = np.array([1.0] * stations)
    
    network = ClosedQueuingNetwork(
        stations,
        P[1:, 1:],
        P[0, 1:],
        mu,
        max_cores,
        min_cores,
        60,  # total users
        1.0,     # assignment iterations
        30
    )

    return network