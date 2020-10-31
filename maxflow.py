import numpy as np
from scipy.sparse import coo_matrix



# Can't use np.inf here because we need inf * False = 0
inf = 1e32
tol = 1e-8

def closure_to_maxflow(edges, weights, check_trivial=False):
    """
    Returns a sparse adjacency matrix that represents the closure graph
    as a max-flow problem.
    """

    s = len(weights)
    t = len(weights) + 1

    edges = list(edges)
    edge_caps = [inf] * len(edges)
    
    if check_trivial:
        tmp = np.array(weights)
        assert np.any(tmp > tol) and np.any(tmp < -tol), \
            "Trivial problem: all weights have same sign"
        del tmp

    for i, w in enumerate(weights):
        if w < - tol:
            edges.append((s, i))
            edge_caps.append(-w)
        elif w > tol:
            edges.append((i, t))
            edge_caps.append(w)
            
    # Add backflow edges
    edge_caps += [0] * len(edges)
    edges += [x[::-1] for x in edges]

    # Build adjacency matrix
    # am = np.zeros((len(weights) + 2, len(weights) + 2))

    # for w, (a, b) in zip(edge_caps, edges):
    #     am[a, b] = w

    am = coo_matrix((edge_caps,
                     list(map(list, zip(*edges)))))
    return am.tocsr()


def dfs_ap(am, s=None, t=None):
    """
    Uses depth-first search and fattest-path rule to find an augmenting s-t path
    in the adjacency matrix am. Returns the path and its bottleneck capacity.
    """
    n = am.shape[1]
    
    assert am.shape[0] == n, \
        "Adjacency matrix must be square"
        
    if s==None: s = n - 2
    if t==None: t = n - 1
        
    assert s < n and t < n, \
        "Invalid source and/or sink node"
    
    path = []
    capacities = []
    visited = np.array([False] * n) # am.shape[1] might be good enough
    
    next_node = s
    while not visited.all():
        visited[next_node] = True
        path.append(next_node)
        choices = am.getrow(next_node).multiply(~visited)
        
        if np.any(choices.toarray() > tol):
            argmax = choices.argmax()
            capacities.append(am[next_node, argmax])
            next_node = argmax
            if next_node == t:
                path.append(t)
                break
            
        else:
            if len(path) > 1: 
                next_node = path.pop(-2)
                path.pop(-1)
                capacities.pop(-1)
            else:
                capacities.append(0)
                break
    
    return path, min(capacities)


def bfs(am, s=None):
    """
    Uses breadth-first search to create a boolean index of which nodes are accessible from s.
    """
    n = am.shape[1]
    
    assert am.shape[0] == n, \
        "Adjacency matrix must be square"
        
    if s==None: s = n - 2
        
    assert s < n, \
        "Invalid source and/or sink node"
    
    visited = np.array([False] * n)
    
    visited[s] = True
    while True:
        n_visited = visited.sum()
        for i in np.arange(n)[visited]:
            visited += (am.getrow(i) > tol).toarray().flatten()
            if visited.all():
                break
        if visited.sum() == n_visited:
            break

    return visited


def maxflow(am_in, s=None, t=None, inplace=False): 
    """
    Computes the maximum flow through the adjacency matrix using Ford-Fulkerson
    algorithm. Returns the updated adjacency matrix and S, the set of nodes on
    the source side of the minimal cut. am_in is copied by default; set inplace
    to true to modify am_in in place and return only S.
    """
    n = am_in.shape[1]
    
    assert am_in.shape[0] == n, \
        "Adjacency matrix must be square"
        
    if s==None: s = n - 2
    if t==None: t = n - 1
        
    assert s < n and t < n, \
        "Invalid source and/or sink node"
    
    if inplace:
        am = am_in
    else:
        am = am_in.copy()
    
    while(True):
        path, cap = dfs_ap(am, s, t)

        if path[-1] != t:
            break

        for i in range(len(path) - 1):
            am[path[i], path[i + 1]] -= cap
            am[path[i + 1], path[i]] += cap   

    S = bfs(am, s)

    if inplace:
        return S
    else: 
        return am, S