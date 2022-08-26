import numpy as np
from scipy.optimize import linprog as minimize
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from maxflow import *

"""
This module provides a lightweight implementation of the Gale-Shapley stable assignment
(proposal) algorithm and a few tools for exploring the stable assignment polytope,
including a modified implementation of the optimal stable marriage algorithm (rotation
algorithm) described in the following reference:

    Irving, Robert W., Paul Leather, and Dan Gusfield. 1987. “An Efficient Algorithm for
    the ‘Optimal’ Stable Marriage.” Journal of the Association for Computing Machinery 34,
    no. 3 (July): 532–43.

Many of the imports above are used only in the vizualization methods; if you don't need
visualization, omitting the networksx and matplotlib imports is OK.
"""


# Global config
default_method = 'simplex'


def find_rotations(shortlists, verbose=False):
    """
    Arguments:

        shortlists      Reduced list of candidates' favorite reviewers (or v.v.),
                        in descending order of preference.

    Returns:

        rotation_idx    List of candidates or reviewers comprising a rotation.

    Finds rotations in a set of shortlists.
    """

    # No rotations possible if preference lists shorter than 2
    visited = [len(k) < 2 for k in shortlists]
    rotation_idx = []
    flag = []
    i = 0

    while False in visited:
        match = None
        if visited[i]:
            flag = []
            i = visited.index(False)

        else:
            flag.append(i)
            visited[i] = True
            for j, j_prefs in enumerate(shortlists):
                if j_prefs:
                    if j_prefs[0] == shortlists[i][1]:
                        match = j
                        break

            if match is None:
                break

            if verbose:
                print("C{}'s next choice is R{}, currently matched with C{}"
                      .format(i, shortlists[i][1], match))

            if match in flag:
                rotation_idx.append(flag[flag.index(match):])
                if verbose:
                    print("  Found a rotation involving candidates {}".format(rotation_idx[-1]))
                visited[match] = True
                flag = []

            else:
                i = match

    return rotation_idx


def elim_rotation(cand_shortlists, reviewer_shortlists, rotation):
    """
    Arguments:

        cand_shortlists        Reduced list of candidates' favorite reviewers,
                               in descending order of preference.

        reviewer_shortlists    Nested list of reviewers' favorite candidates,
                               in descending order of preference.

        rotation               List of candidates involved in rotation to be eliminated.

    Returns:

        removals               Tuples specifying pairs of candidates and reviewers
                               who are eliminated by this rotation. Includes the
                               participants in the rotation themselves.

        cands and reviewers are modified in-place.

    Eliminates the rotation specified from the shortlists. Undefined behavior
    if the rotation is not actually present in the cand_shortlists graph.
    """

    # Candidates and their new matches
    rotation_tuples = [(c, cand_shortlists[c][0]) for c in rotation]
    removals = []

    for i, (c, r) in enumerate(rotation_tuples):
        # Remove p (anyone same or worse than rotation[i]) from r's list
        for _ in range(len(reviewer_shortlists[r]) - reviewer_shortlists[r].index(rotation[i-1]) -1):
            p = reviewer_shortlists[r].pop(-1)
            # Remove r from candidate p's list as well, if it appears
            if r in cand_shortlists[p]:
                removals.append((p, r))
                cand_shortlists[p].remove(r)

    return removals


def preprocessor(cands, reviewers):
    """
    Arguments:

        cands           Nested list of candidates' favorite reviewers,
                        in descending order of preference.

        reviewers       Nested list of reviewers' favorite candidates,
                        in descending order of preference.

    Returns:

        cands and reviewers are modified in-place.

    Preprocesses desk rejects. When a reviewer has excluded a candidate from
    their ranking, this function automatically removes that reviewer from the
    candidate's list. Occurs in place, and only cands is modified.

    This is not strictly necessary for the proposal algorithm to find stable
    assignments, but it must be done to search for rotations, since the proposal
    algorithm does not visit pairings that are further down the list than the
    candidate-optimal stable pairings.
    """

    n_cands = len(cands)
    for r, r_prefs in enumerate(reviewers):
        for c in range(n_cands):
            if c not in r_prefs and r in cands[c]:
                cands[c].remove(r)


def proposal(cands, reviewers, cand_capacity=None, reviewer_capacity=None, verbose=False):
    """
    Finds a stable pairing of candidates and reviewers using the proposal (Gale-Shapley)
    algorithm.

    Arguments:

        cands           Nested list of candidates' favorite reviewers,
                        in descending order of preference.

        reviewers       Nested list of reviewers' favorite candidates,
                        in descending order of preference.

        cand_capacity       Max number of reviewers matched to each candidate
                            in final assignment (defaults to 1).

        reviewer_capacity   Max number of candidates matched to each candidate
                            in final assignment (defaults to 1).

        verbose         If True, prints results of each round.

    Returns:

        matches         List of tuples giving candidate-optimal matches.

        reduction       Cands after deletions; each candidate's list of
                        reviewers who did not reject them (distinct from
                        shortlists, given elsewhere).

    One or both of cand_capacity or reviewer_capacity must be None, or else the GS
    algorithm is unsuitable.
    """

    assert cand_capacity is None or reviewer_capacity is None, \
        "Many-to-many matching not supported by proposal algorithm"
    assert max([max(i) for i in cands if i]) < len(reviewers), \
        "Candidates ranked more reviewers than exist"
    assert max([max(i) for i in reviewers if i]) < len(cands), \
        "Reviewers ranked more candidates than exist"

    if cand_capacity is None:
        cand_capacity = [1] * len(cands)
    else:
        assert len(cand_capacity) == len(cands)
    if reviewer_capacity is None:
        reviewer_capacity = [1] * len(reviewers)
    else:
        assert len(reviewer_capacity) == len(reviewers)

    reduction = deepcopy(cands)
    nit = 0
    removals = [0]
    dummy = len(cands)

    while removals:
        removals = []
        if verbose:
            print("Reduced candidate lists:")
            print(reduction)
            print("Results of round {}:".format(nit))
        for i, i_prefs in enumerate(reviewers):
            rejections = []

            # Candidates propose to their cc top picks
            for j, j_prefs in enumerate(reduction):
                if i in j_prefs[:min(cand_capacity[j], len(j_prefs))]:
                    rejections.append(j)
                    if verbose:
                        print("  Candidate {} proposed to reviewer {}".format(j, i))

            # Assign "desk rejects" (those who are unacceptable to i) a large
            # dummy rank
            i_ranks_proposals = [(i_prefs.index(j) if j in i_prefs else dummy)
                                 for j in rejections]

            # Reviewers reject all but the rc best candidates
            for rc in range(reviewer_capacity[i]):
                if rejections:
                    maybe = min(i_ranks_proposals)
                    if maybe != dummy:
                        rejections.remove(i_prefs[maybe])
                        i_ranks_proposals.remove(maybe)

            for k in rejections:
                if verbose:
                    print("    Reviewer {} rejected candidate {}".format(i, k))
                removals.append((k, i))

        for k, i in removals:
            reduction[k].remove(i)
        nit += 1

    matches = []
    for c, c_prefs in enumerate(reduction):
        for j in c_prefs[:min(cand_capacity[c], len(c_prefs))]:
            matches.append((c, j))

    return matches, reduction


def osa_from_rotation_digraph(edges, rotation_poset, rotation_weights,
                              rotation_depths, rotation_key,
                              method=default_method, verbose=False):
    """
    Arguments:

        edges               Set of directed edges in the subgraph.

        rotation_poset      List of tuples comprising rotations. The first entry in each
                            tuple is a candidate, and the second entry is the reviewer
                            that candidate is currently matched with.

        rotation_weights    Weight of each rotation (node).

        rotation_depths     Depth of each rotation (node). Used to check which nodes
                            are immediate predecessors of others, and also handy
                            for visualizations.

        rotation_key        Convenience list indicating which rotations (by index)
                            appear at each depth.

        method              'hone' to use heuristic (moving cones) algorithm, 'simplex'
                            to use generic simplex, 'maxflow' to use network simplex 
                            (Ford-Fulkerson) algorithm.

        verbose=False       Self-explanatory.

    Returns:

        r_in_opt            Boolean index of which rotations are included in the
                            optimal assignment.

        rotation_members    Lists of members in each rotation, for easy transfer
                            to elim_rotations().

    You can use the output of assignment.rotation_digraph() as the input.
    """

    assert method in 'hone maxflow simplex'.split(), \
        "Unknown method {}".format(method)

    # A few trivial cases that tend to produce bugs
    tmp = np.array(rotation_weights)
    if np.all(tmp < tol):
        r_in_opt = np.zeros_like(tmp, dtype=bool)
    elif np.all(tmp > -tol):
        r_in_opt = np.ones_like(tmp, dtype=bool)

    elif method == 'hone':  # Moving cones
        r_in_opt = np.zeros_like(rotation_weights, dtype=bool)

        predecessors = [set() for _ in rotation_weights]

        # Immediate predecessors of each node
        for a, b in edges:
            predecessors[b].add(a)

        # Function that recursively finds all of a node's predecessors.
        # Unused and not particularly efficient.
        # def pred(i):
        #     return predecessors[i].union(*(pred(j) for j in predecessors[i]))

        # Marks a node and all of its predecessors as part of the optimal set.
        def pred_marker(i):
            r_in_opt[i] = True
            for j in predecessors[i]:
                pred_marker(j)

        # Calculates the sum of the weights of a node and all of its predecessors
        # not currently in the optimal set, i.e. the node's mass at current solution.
        def pred_mass(inp):
            def recursor(i):
                nonlocal out
                out += rotation_weights[i]
                for j in predecessors[i]:
                    if not r_in_opt[j]:
                        recursor(j)
            out = 0
            recursor(inp)
            return out

        while True:
            sum0 = r_in_opt.sum()
            if verbose:
                print("\nCurrent solution includes these rotations: \n", r_in_opt)
            for i, w_i in enumerate(rotation_weights):
                # Neither of these checks are necessary but improves performance
                if w_i >= 0 and not r_in_opt[i]:
                    pm = pred_mass(i)
                    if verbose:
                        print("  Including rotation {} will yield change utility of {}."
                              .format(i, pm))
                    if pm >= 0:
                        if verbose:
                            print("    Let's include it.")
                        pred_marker(i)

            # Stop when optimal set not augmented
            if r_in_opt.sum() == sum0:
                if verbose:
                    print("\nSolution did not change. All done.")
                break
        
    elif method=='maxflow':     # Network flow method
        am = closure_to_maxflow(edges, rotation_weights)
        S = maxflow(am, inplace=True, verbose=verbose)
        r_in_opt = ~S[:-2]

    else:    # Simplex method
        c = -np.array(rotation_weights)

        A_ub = np.zeros((len(edges), len(rotation_weights)))
        for k, (i, j) in enumerate(edges):
            A_ub[k, j] = 1
            A_ub[k, i] = -1
        b_ub = np.zeros(len(edges))

        bounds = [(0, 1) for _ in rotation_weights]

        out = minimize(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                       method='simplex',
                       callback=lambda o: print(o, end="\n\n") if verbose else None)

        # Get boolean idx of which rotations are used in optimal assignment
        r_in_opt = out.x.round() == 1

    rotation_members = np.array([[a[0] for a in rotation_tuples]
                                 for rotation_tuples in rotation_poset],
                                dtype=list)[r_in_opt].tolist()

    return r_in_opt, rotation_members


def viz_prefs(cands, reviewers, kwargs={}):
    """
    Arguments:

        cands           Nested list of candidates' favorite reviewers,
                        in descending order of preference.

        reviewers       Nested list of reviewers' favorite candidates,
                        in descending order of preference.

    Returns:

        fig, ax         Matplotlib plot.

    Color-coded representation of the reviewers' and candidates' preferences.
    Currently works only for square data.
    """

    fig, ax = plt.subplots(1, 2, **kwargs)

    ins = [cands, reviewers]
    labels = ["Candidates", "Reviewers"]

    for k in range(2):

        im = ax[k].imshow(ins[k], cmap="winter")
        ax[k].set_xticks([])
        ax[k].set_yticks([])

        for i, cands in enumerate(ins[k]):
            for j, reviewers in enumerate(ins[1 - k]):
                text = ax[k].text(j, i, cands[j],
                                  ha="center", va="center", color="w")

        ax[k].set_title(labels[k])
        fig.tight_layout()

    return fig, ax


class assignment:

    def __init__(self, cands, reviewers,
                 cand_capacity=None, reviewer_capacity=None, cost_arr=None):
        assert max([max(i) for i in cands]) <= len(reviewers) - 1, \
            "Candidates ranked more reviewers than exist"
        assert max([max(i) for i in reviewers]) <= len(cands) - 1, \
            "Reviewers ranked more candidates than exist"

        self.cands = cands
        self.reviewers = reviewers
        self.cand_capacity = cand_capacity
        self.reviewer_capacity = reviewer_capacity
        self.cost_arr = cost_arr
        self.shortlists_internal = None

    def proposal(self, reverse=False, verbose=False):
        """
        Arguments:

            reverse=False   If true, runs proposal algorithm in the reverse direction.
                            Returns also appear in reverse order.

            verbose=False   If True, prints results of each round.

        Returns:

            matches         List of tuples giving candidate-optimal matches.

            reduction       Cands after deletions; each candidate's list of
                            reviewers who did not reject them (distinct from
                            shortlists, given elsewhere).

        Convenience wrapper to run proposal algorithm on inputs.
        """

        if reverse:
            return proposal(self.reviewers, self.cands,
                            self.reviewer_capacity, self.cand_capacity, verbose)
        else:
            return proposal(self.cands, self.reviewers,
                            self.cand_capacity, self.reviewer_capacity, verbose)

    def cost(self, pairings=None, reverse=False, cost_arr=None):
        """
        Arguments:

            pairings=None   List of tuples (i, j), where i is a candidate and j is
                            the reviewer they are matched with. Uses the proposal
                            algorithm pairings if none provided.

            reverse=False   Enable if reviewer indices are given first.

            cost_arr=self.cost_arr   Array of costs associated with each pairing; sum of candidate
                                     and reviewer rankings used if none supplied.

        Returns:

            c        Value of cost functions.

        Returns the cost of (sum of ranks or sum of array entries associated with given
        pairings) associated with the given assignment.
        """

        if cost_arr is None:
            cost_arr = self.cost_arr

        if pairings is None:
            pairings = self.proposal(reverse=reverse)[0]

        if reverse:
            pairings = [(j, i) for i, j in pairings]

        c = 0

        if cost_arr is not None:
            for i, j in pairings:
                c += cost_arr[i, j]

        else:
            for i, j in pairings:
                c += self.cands[i].index(j) + self.reviewers[j].index(i)

        return c

    def shortlists(self, reverse=False, verbose=False):
        """
        Arguments:

            reverse=False       Runs proposal algorithm in the reverse direction.

            verbose=False       Self-explanatory.

        Returns:

            cand_shortlists         Candidate shortlists.

            reviewer_shortlists     Reviewer shortlists.

        The output is also written to self.shortlists_internal, if self.shortlists_internal
        has not been defined yet.

        After running the proposal algorithm, we obtain a list of candidate-pessimal matches.
        We may create (candidate-oriented) shortlists for both groups by removing any matches
        worse than these from the reviewers' rankings, and removing the same from the
        candidates' reduced lists. See Irving et al., 534.
        """

        if reverse:
            reviewers_pp = deepcopy(self.reviewers)
            preprocessor(reviewers_pp, self.cands)
            cand_shortlists = proposal(reviewers_pp, self.cands,
                                       self.reviewer_capacity, self.cand_capacity, verbose)[1]
            reviewer_shortlists = deepcopy(self.cands)

        else:
            cands_pp = deepcopy(self.cands)
            preprocessor(cands_pp, self.reviewers)
            cand_shortlists = proposal(cands_pp, self.reviewers,
                                       self.cand_capacity, self.reviewer_capacity, verbose)[1]
            reviewer_shortlists = deepcopy(self.reviewers)

        for c, c_prefs in enumerate(cand_shortlists):
            if c_prefs:
                dex = reviewer_shortlists[c_prefs[0]].index(c)
                for j in reviewer_shortlists[c_prefs[0]][dex + 1:]:
                    if c_prefs[0] in cand_shortlists[j]:
                        cand_shortlists[j].remove(c_prefs[0])
                reviewer_shortlists[c_prefs[0]] = reviewer_shortlists[c_prefs[0]][:dex + 1]

        # We need this in self.osa()
        if self.shortlists_internal is None:
            self.shortlists_internal = deepcopy((cand_shortlists, reviewer_shortlists))

        return cand_shortlists, reviewer_shortlists

    def xshortlists(self, verbose=False):
        """
        Arguments:

            verbose=False       Self-explanatory.

        Returns:

            cand_shortlists, reviewer_shortlists      See below.

        If we run the proposal algorithm both ways, we get lists of candidate- and reviewer-
        pessimal matches. Removing matches worse than these from the other party's rankings
        yields a unique pair of "extra short lists." This further reduction is not necessary
        for the rotation algorithm, but provided for interest and further study.
        """

        cand_shortlists = proposal(self.cands, self.reviewers,
                                   self.cand_capacity, self.reviewer_capacity, verbose)[1]
        reviewer_shortlists = proposal(self.reviewers, self.cands,
                                       self.reviewer_capacity, self.cand_capacity, verbose)[1]

        for c, c_prefs in enumerate(cand_shortlists):
            if c_prefs:
                dex = reviewer_shortlists[c_prefs[0]].index(c)
                reviewer_shortlists[c_prefs[0]] = reviewer_shortlists[c_prefs[0]][:dex + 1]

        for r, r_prefs in enumerate(reviewer_shortlists):
            if r_prefs:
                dex = cand_shortlists[r_prefs[0]].index(r)
                cand_shortlists[r_prefs[0]] = cand_shortlists[r_prefs[0]][:dex + 1]

        return cand_shortlists, reviewer_shortlists

    def rotate(self, reverse=False, verbose=False):
        """
        Arguments:

            reverse=False       Runs proposal algorithm in the reverse direction.

            verbose=False       Self-explanatory.

        Returns:

            rotation_poset      List of tuples comprising rotations. The 0th entry in each
                                tuple is a candidate, and the 1st entry is the reviewer
                                that candidate is currently matched with.

            rotation_removals   List of tuples comprising matches eliminated by a rotation
                                (including the participants in the rotation themselves). See
                                See Irving et al., 535.

            rotation_weights    Weight of each rotation in the poset.

            rotation_depths     Depth of each rotation in the poset graph; a lower bound on
                                the number of immediate predecessors.

        Discovers the rotations leading from one set of shortlists to
        the other.
        """

        if verbose:
            print("Generating shortlists")

        G_shortlists, H_shortlists = self.shortlists(reverse, verbose)
        depth = 0

        if reverse:
            cands = self.reviewers
            reviewers = self.cands
        else:
            cands = self.cands
            reviewers = self.reviewers

        rotation_poset = []
        rotation_removals = []
        rotation_weights = []
        rotation_depths = []

        while True:
            if verbose:
                print("\nFinding rotations")
            rotations = find_rotations(G_shortlists, verbose)
            if rotations:
                if verbose:
                    print("\nEliminating rotations")
                for r in rotations:
                    rotation_poset.append([(c, G_shortlists[c][0]) for c in r])

                    # Decrease in cost function associated with eliminating this rotation
                    weight = 0
                    if self.cost_arr is not None:
                        for c in r:
                            weight += (self.cost_arr[c, G_shortlists[c][0]]
                                       - self.cost_arr[c, G_shortlists[c][1]])
                    else:
                        for c in r:
                            weight += (cands[c].index(G_shortlists[c][0])
                                       - cands[c].index(G_shortlists[c][1])
                                       + reviewers[G_shortlists[c][0]].index(c)
                                       - reviewers[G_shortlists[c][1]].index(c))
                    rotation_weights.append(weight)
                    rotation_depths.append(depth)

                    if verbose:
                        print("  Eliminating rotation {}, weight {}".format(r, weight))

                    rotation_removals.append(elim_rotation(G_shortlists, H_shortlists, r))

                if verbose:
                    print("\nShortlists after eliminating rotations at depth {}".format(depth))
                    print(G_shortlists)
                    print(H_shortlists)

                depth += 1

            else:
                break

        return rotation_poset, rotation_removals, rotation_weights, rotation_depths

    def rotation_digraph(self, reverse=False, verbose=False):
        """
        Arguments:

            reverse=False       Runs proposal algorithm in the reverse direction.

            verbose=False       Self-explanatory.

        Returns:

            edges               Set of directed edges in the subgraph.

            rotation_poset      List of tuples comprising rotations. The first entry in each
                                tuple is a candidate, and the second entry is the reviewer
                                that candidate is currently matched with. (*)

            rotation_weights    Weight of each rotation (node). (*)

            rotation_depths     Depth of each rotation (node). Used to check which nodes
                                are immediate predecessors of others, and also handy
                                for visualizations. (*)

            rotation_key        Convenience list indicating which rotations (by index)
                                appear at each depth.

        Generates the data for the directed subgraph used to model the optimal marriage
        a maximum-flow problem. The starred returns are simply passed through from
        self.rotate().
        """

        rotation_poset, rotation_removals, rotation_weights, rotation_depths = \
            self.rotate(reverse, verbose)

        if not rotation_poset:
            if verbose:
                print("No rotations present")
            return [[], [], [], [], []]

        if reverse:
            cands = self.reviewers
        else:
            cands = self.cands

        edges = set()

        if verbose:
            print("\nDiscovering edges")

        # Group together the idxs of the rotations at each depth
        d_max = rotation_depths[-1]
        rotation_key = [[] for _ in range(d_max + 1)]
        for r_id, depth in enumerate(rotation_depths):
            rotation_key[depth].append(r_id)

        # At each depth,
        for d in range(d_max):
            # For each of the rotations at this depth,
            for t in rotation_key[d + 1]:
                cands_in_t = [a for a, b in rotation_poset[t]]
                reviewers_in_t = [b for a, b in rotation_poset[t]]
                # Inspect the rotations at the preceding depth
                for s in range(min(rotation_key[d+1])):

                    if (s, t) in edges:
                        if verbose:
                            print("Already have an edge from {} to {}".format(s, t))
                        break

                    cands_in_s = {a for a, _ in rotation_poset[s]}

                    if cands_in_s.intersection(cands_in_t):
                        if verbose:
                            print("New edge from {} to {} (rule 1)".format(s, t))
                        edges.add((s, t))

                    # A surprisingly simple implementation of rule 2.
                    # If one of the members of the present rotation was involved
                    # in a removal in the previous, then this rotation depends on it.
                    # No need to check the ranks because if they were ranked lower,
                    # they would've been eliminated in elim_rotation() without adding
                    # to removal list.
                    else:
                        for c, r in rotation_removals[s]:
                            if c in cands_in_t:  # and r in cands[c]:
                                if verbose:
                                    print("New edge from {} to {} (rule 2)".format(s, t))
                                edges.add((s, t))
                                break

        return edges, rotation_poset, rotation_weights, rotation_depths, rotation_key

    def draw_rotation_digraph(self, augment=False, opt=True, reverse=False,
                              method=default_method, verbose=False, kwargs={}):
        """
        Arguments:

            augment=False       Whether to augment the digraph with sink and source nodes
                                so that it can be visually inspected for the minimal cut.

            opt=True            Whether to highlight the nodes included in the optimal assignment.

            reverse=False       Whether to run proposal algorithm in the reverse direction.

            method              'hone' to use my optimization algorithm, 'simplex'
                                to use generic simplex.

            verbose=False       Self-explanatory.

            kwargs={}           To be passed to matplotlib.

        Returns:

            graph               A NetworkX directed graph describing the dependencies among
                                the rotations.

            (assn)              The optimal pairings, if opt=True.

        Uses matplotlib and NetworkX to draw the rotation digraph, which can be visually
        inspected for the minimal cut corresponding to the optimal solution.
        """

        assn, r_in_opt, edges, rotation_poset, rotation_weights, rotation_depths, rotation_key = \
            self.osa(reverse, method, verbose, heavy=True)

        if rotation_depths:
            n = len(rotation_weights)

            pos = [[depth, i % (n**0.5)] for i, depth in enumerate(rotation_depths)]

            labels = {i: i for i in range(n)}

            if augment:
                # Source and sink
                labels[n] = 's'
                labels[n + 1] = 't'
                pos.append((-1.5, -.33 * n**0.5))
                pos.append((max(rotation_depths) + 1.5, -.67 * n**0.5))

                # Initial edge capacities are inf
                edges_capacities = [1e16] * len(edges)

                edges_st = []
                edges_st_capacities = []

                for i, w in enumerate(rotation_weights):
                    if w < 0:
                        edges_st.append((n, i))
                        edges_st_capacities.append(-w)
                    elif w > 0:
                        edges_st.append((i, n + 1))
                        edges_st_capacities.append(w)

            pos = np.array(pos)

            plt.figure(**kwargs)
            graph = nx.DiGraph()
            graph.add_nodes_from(labels)
            graph.add_edges_from(edges)
            nx.draw_networkx_nodes(graph,
                                   pos,
                                   nodelist=range(n),
                                   node_color='black')
            nx.draw_networkx_edges(graph,
                                   pos,
                                   edgelist=edges,
                                   edge_color='seagreen')
            # Rotation indices
            nx.draw_networkx_labels(graph, pos, labels, font_color='white')
            # Rotation weights
            if not augment:
                nx.draw_networkx_labels(graph,
                                        pos + [0, 0.06 * n**0.5],
                                        {i: "({})".format(w) for i, w in enumerate(rotation_weights)},
                                        font_color='black')

            if opt:
                nx.draw_networkx_nodes(graph,
                                       pos,
                                       nodelist=np.arange(n)[r_in_opt],
                                       node_color='dodgerblue')

            if augment:
                nx.draw_networkx_nodes(graph,
                                       pos,
                                       nodelist=[n, n + 1],
                                       node_color='slategray')

                nx.draw_networkx_edges(graph,
                                       pos,
                                       edgelist=edges_st,
                                       edge_color='darkorchid',
                                       label=edges_capacities)

                props = dict(boxstyle='square', lw=0, fc='white', alpha=0.5)
#                 nx.draw_networkx_edge_labels(graph, pos,
#                                              edge_labels={e: r'$\infty$' for e in edges},
#                                              bbox=props)
                nx.draw_networkx_edge_labels(graph, pos,
                                             edge_labels={e: l for e, l in
                                                          zip(edges_st, edges_st_capacities)},
                                             bbox=props)

            x0, x1 = plt.ylim()
            plt.ylim(x0, x1 + 0.06 * n**0.5)

        else:
            print("No rotations to graph")
            graph = None

        if opt:
            return graph, assn
        else:
            return graph

    def osa(self, reverse=False, method=default_method, verbose=False, heavy=False):
        """
        Arguments:

            reverse=False       Runs proposal algorithm in the reverse direction.

            method=default_method   'hone' to use my optimization algorithm, 'simplex'
                                    to use generic simplex. Config above.

            verbose=False       Self-explanatory.

            heavy=False         Whether to pass through the rotation digraph data from
                                self.rotation_digraph(); used by self.draw_rotation_digraph().

        Returns:

            out           A Scipy OptimizeResult object.

            r_in_opt      Boolean index of which rotations are included in the optimal assignment.

            (5 others)    Passes through results of self.rotation_digraph() if heavy was enabled.

        Uses Scipy's solver to find the optimal stable assignment from the rotation digraph
        data. Returns a Scipy OptimizeResult object; note that the reported function value
        negative weight of the maximal rotation poset.
        """

        edges, rotation_poset, rotation_weights, rotation_depths, rotation_key = \
            self.rotation_digraph(reverse, verbose)

        G, H = deepcopy(self.shortlists_internal)

        if rotation_depths:
            if verbose:
                print("\nOptimizing over rotation set")

            r_in_opt, rotation_members = \
                osa_from_rotation_digraph(edges, rotation_poset, rotation_weights,
                                          rotation_depths, rotation_key,
                                          method, verbose)

            # Now we will eliminate the rotations; G and H are the original shortlists
            # stored when we first rotated.
            for i in rotation_members:
                elim_rotation(G, H, i)

        else:
            r_in_opt = []

        assn = [(i, g[0]) for i, g in enumerate(G) if len(g) >= 1]

        self.shortlists_internal = None

        if heavy:
            return assn, r_in_opt, \
                   edges, rotation_poset, rotation_weights, rotation_depths, rotation_key
        else:
            return assn, r_in_opt
        