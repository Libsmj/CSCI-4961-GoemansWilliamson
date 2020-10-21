"""
Functional library with maximum cut algorithms.

"""
#    This file is part of maxcutpy.
#
#    maxcutpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    maxcutpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with max_cut.  If not, see <http://www.gnu.org/licenses/>.

#    Andrea Casini <andreacasini88@gmail.com>
#	 Nicola Rebagliati <nicola.rebagliati@gmail.com>

__author__ = "\n".join(['Andrea Casini (<andreacasini88@gmail.com>)',
                        'Nicola Rebagliati (<nicola.rebagliati@gmail.com>)'])

import networkx as nx
import numpy as np
import signal

from multiprocessing import Process, Queue, cpu_count
from scipy import integrate

import gcut as gc

__all__ = ['brute_force_max_cut',
           'greedy_cut',
           'consistent_max_cut',
           'marked_consistent_max_cut',
           'recursive_spectral_cut',
           'soto_function',
           'trevisan_function']

#==============================================================================
# Time out decorator
#
# Return execution time in seconds.
#==============================================================================


class TimedOutExc(Exception):
    pass


def timeout(timeout):
    def decorate(f):
        def handler():
            raise TimedOutExc()

        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)

            try:
                result = f(*args, **kwargs)
            except TimedOutExc:
                result = None
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
            return result

        new_f.func_name = f.func_name
        return new_f

    return decorate


#==============================================================================
# Theoretic Approximation Functions
#==============================================================================


def _f1(x, eps):
    return ((-1.0 + np.sqrt(4.0 * (eps / x) ** 2 - 8.0 * eps / x + 5.0)) /
            (2.0 * (1.0 - eps / x)))


def _f2(x, eps):
    return 1.0 / (1.0 + 2.0 * np.sqrt((1.0 - eps / x) * eps / x))


def soto_function(eps):
    """Implementation of Jose' Soto's theory"""

    # Unique solution
    eps0 = 0.2280155

    if eps >= 1.0 / 3.0:
        return 1.0 / 2.0

    if eps0 <= eps <= 1.0 / 3.0:
        return ((integrate.quad(lambda x: 1.0 / 2.0, 0, 3 * eps)[0] +
                 integrate.quad(_f1, 3.0 * eps, 1.0, eps)[0]
            ))

    if not eps:
        return 1.0

    if eps <= eps0:
        return ((integrate.quad(lambda x: 1.0 / 2.0, 0, 3 * eps)[0] +
                 integrate.quad(_f1, 3.0 * eps, eps / eps0, eps)[0] +
                 integrate.quad(_f2, eps / eps0, 1.0, eps)[0]
            ))


def trevisan_function(eps):
    """Implementation of Luca Trevisan's theory"""
    if eps <= 1.0 / 16.0:
        return 1.0 - 4.0 * np.sqrt(eps) + 8.0 * eps
    return 1.0 / 2.0


#==============================================================================
# Greedy Cut
#
# Approximation ratio: 0.5
# Complexity: O(n^2)
#==============================================================================


def _greedy_choice(G, candidate, blue_nodes, black_nodes, visited):
    """Helper function to greedy cut"""

    G.node[candidate][gc.PARTITION] = gc.BLUE
    blue_cut_val = gc.cut_edges(nx.subgraph(G, visited))

    G.node[candidate][gc.PARTITION] = gc.BLACK
    black_cut_val = gc.cut_edges(nx.subgraph(G, visited))

    if blue_cut_val > black_cut_val:
        G.node[candidate][gc.PARTITION] = gc.BLUE
        blue_nodes.add(candidate)
    else:
        black_nodes.add(candidate)

    return blue_nodes, black_nodes


def greedy_cut(G, nbunch=None, visited=None):
    """Return a good cut of a graph G.

    Good cut: a cut is good if it cuts at least half of the number of
    edges of the graph.

    """
    if nbunch is None:
        nbunch = G.nodes()  # set of nbunch

    if visited is None:
        visited = set()

    gc.init_cut(G, nbunch)

    blue_nodes = set()
    black_nodes = set()

    candidate = nbunch.pop()
    visited.add(candidate)

    _greedy_choice(G, candidate, blue_nodes, black_nodes, visited)

    while nbunch:
        candidate = nbunch.pop()
        visited.add(candidate)

        _greedy_choice(G, candidate, blue_nodes, black_nodes, visited)

    gc.set_partitions(G, blue_nodes, black_nodes)
    return blue_nodes, black_nodes


#==============================================================================
# 2TSC Approximation Algorithm by Luca Trevisan
#
# Approximation ratio: 0.531
# Complexity: O(n^2)
#==============================================================================


def _first_lemma(G, y):
    """Compute first lemma."""
    numerator = 0.0
    for i in y:
        for j in y:
            if G.has_edge(i, j):
                numerator += abs(y[i] + y[j])

    denominator = 0.0
    for i in y:
        denominator += G.degree(i) * abs(y[i])

    # Handle float division by zero
    if not denominator:
        return None
    return numerator / denominator


def _second_lemma(G):
    """Compute second lemma."""
    cut_edges = gc.cut_edges(G)
    uncut_edges = G.number_of_edges() - cut_edges

    numerator = float(uncut_edges - cut_edges)
    denominator = float(G.number_of_edges())

    return numerator / denominator


def _largest_eigenvector(G):
    """Return the largest eigenvector of Laplacian of graph G."""
    L = nx.normalized_laplacian(G)
    eigenvalues, eigenvectors = np.linalg.eig(L)
    ind = np.argmax(eigenvalues)    # highest eigenvalue index and
    largest = eigenvectors[:, ind]  # its corresponding eigenvector
    return dict(zip(G, largest))


def _two_threshold_spectral_cut(G):
    """Return an indicator vector of a cut computed using the largest
    eigenvector.

    """
    x = _largest_eigenvector(G)

    smallest = gc.sign_norm(dict(x))  # all 1 and -1 vector
    min_ratio = _first_lemma(G, smallest)

    y = dict.fromkeys(x, 0)

    for k in x:
        for i in x:
            if x[i] < -abs(x[k]):
                y[i] = -1
            elif x[i] > abs(x[k]):
                y[i] = 1
            elif abs(x[i]) <= abs(x[k]):
                y[i] = 0

        # Compute first lemma
        ratio = _first_lemma(G, y)

        if ratio is not None:
            if min_ratio is None:
                min_ratio, smallest = ratio, dict(y)
            elif ratio < min_ratio:
                min_ratio, smallest = ratio, dict(y)

    # Smallest ratio's vector
    return smallest


def _aux_recursive_spectral_cut(G):
    """Return an approximate solution to the max cut problem.

    Use the two_threshold_spectral_cut and recursively cut
    the undecided nodes.

    """

    if G is None or G.number_of_nodes() == 0:
        return set(), set()

    smallest = _two_threshold_spectral_cut(G)

    R = set()
    L = set()
    V = set()

    for i in smallest:
        if smallest[i] == gc.BLUE:
            R.add(i)
        elif smallest[i] == gc.UNDECIDED:
            V.add(i)
        elif smallest[i] == gc.BLACK:
            L.add(i)

    G1 = nx.Graph(nx.subgraph(G, V))

    M = G.number_of_edges() - G1.number_of_edges()
    C = gc.edges_between(G, L, R)  # cut edges
    X = gc.edges_between(G, L, V) + gc.edges_between(G, R, V)

    if C + 0.5 * X <= 0.5 * M or not V:
        if gc.edges_between(G, L, R) < G.number_of_edges() / 2:
            return greedy_cut(G)
        return L, R

    if C + 0.5 * X > 0.5 * M:
        # SPECIAL CASE: all undecided nodes (V) are isolate deg = 0
        if gc.is_all_isolate(G1):
            gc.set_partitions(G, L, R)
            visited = (L | R) - V
            B, K = greedy_cut(G, V, visited)
            return L | B, R | K

        V1, V2 = recursive_spectral_cut(G1)

        if (gc.edges_between(G, V1 | L, V2 | R) >
            gc.edges_between(G, V1 | R, V2 | L)):
            return  V1 | L, V2 | R
        return V1 | R, V2 | L


def recursive_spectral_cut(G):
    B, K = _aux_recursive_spectral_cut(G)
    # Set blue and black nodes in graph G
    gc.set_partitions(G, B, K)
    return gc.cut_edges(G)


#==============================================================================
# Enumerative Methods for Maximum Cut's Exact Solutions by Andrea Casini
#
# Complexity: O(2^n)
#==============================================================================


def brute_force_max_cut(G):
    """Compute maximum cut of a graph considering all the possible cuts."""

    max_cut_value = 0
    max_cut_ind = 0

    n = G.number_of_nodes()

    for i in range(1, 2 ** (n - 1)):
        cut_graph = nx.Graph(G)

        gc.binary_cut(cut_graph, i)
        value = gc.cut_edges(cut_graph)

        if value > max_cut_value:
            max_cut_value = value
            max_cut_ind = i

    gc.binary_cut(G, max_cut_ind)
    return gc.partition_dictionary(G), max_cut_value


#==============================================================================
# Consistent Max Cut
#==============================================================================


def _choose_new_candidate(partition_dict, Q):
    """Return the first candidate node.

    Candidate node: first undecided at top of the stack.

    """
    candidate = Q.pop()
    n_color = partition_dict[candidate]

    # Choose the first marked or undecided node
    while n_color != gc.UNDECIDED and n_color != gc.MARKED and Q:
        candidate = Q.pop()
        n_color = partition_dict[candidate]

    # Q is empty and the last candidate is been yet colored
    if n_color is not gc.UNDECIDED and n_color is not gc.MARKED:
        return None
    return candidate


def _aux_consistent_max_cut(G,
                            partition_dict,
                            Q,
                            candidate,
                            color,
                            B_nn,
                            K_nn):
    """Helper function to 'Consistent Max Cut Algorithm'.

    Compute and returns the max cut dictionary and cut value together
    with the blue and black neighbors dictionaries.

    """

    # Color the candidate and update its neighbors
    partition_dict[candidate] = color
    neighbors = gc.update_neighbors_labels(G,
        candidate,
        color,
        B_nn,
        K_nn)

    # Check consistency condition on candidate's neighbors
    if not gc.is_cut_consistent(G, partition_dict, neighbors, B_nn, K_nn):
        return None, 0, B_nn, K_nn

    # All nodes are being colored
    if not Q:
        return partition_dict, gc.cut_edges(G, partition_dict), B_nn, K_nn

    is_colored = True

    while is_colored:
        is_colored, is_consistent = gc.could_be_cut(G,
            partition_dict,
            Q,
            B_nn,
            K_nn)

        if not is_consistent:
            return None, 0, B_nn, K_nn

    # Pick a new candidate
    new_candidate = _choose_new_candidate(partition_dict, Q)

    # Stop if there is no valid candidate
    if new_candidate is None:
        return partition_dict, gc.cut_edges(G, partition_dict), B_nn, K_nn

    # Recursively blue and black coloring the candidate
    B_cut_dict, B_cut_val, BB_nn, BK_nn = _aux_consistent_max_cut(G,
        dict(partition_dict),
        list(Q),
        new_candidate,
        gc.BLUE,
        dict(B_nn),
        dict(K_nn))

    K_cut_dict, K_cut_val, KB_nn, KK_nn = _aux_consistent_max_cut(G,
        dict(partition_dict),
        list(Q),
        new_candidate,
        gc.BLACK,
        dict(B_nn),
        dict(K_nn))

    if B_cut_dict is None and K_cut_dict is None:
        return None, 0, None, None

    # Choose best cut according to the effective cut value
    if B_cut_val > K_cut_val:
        return B_cut_dict, B_cut_val, BB_nn, BK_nn
    return K_cut_dict, K_cut_val, KB_nn, KK_nn


def consistent_max_cut(G, lowest=False):
    """Compute maximum cut of a graph taking advantage of
    the consistency property.

    """

    gc.init_cut(G)

    # Initialize dictionaries
    partition_dict = dict.fromkeys(G, gc.UNDECIDED)

    B_nn = dict.fromkeys(G, 0)
    K_nn = dict.fromkeys(G, 0)

    # Visit graph according to the node's degree
    deg_node_seq = gc.degree_nodes_sequence(G, reverse=lowest)

    # Pick first candidate
    candidate = _choose_new_candidate(partition_dict, deg_node_seq)

    max_cut_dict, max_cut_val, B_nn, K_nn = _aux_consistent_max_cut(G,
        partition_dict,
        deg_node_seq,
        candidate,
        gc.BLUE,
        B_nn,
        K_nn)

    # Use dictionary to cut the graph
    gc.cut(G, max_cut_dict)
    return max_cut_dict, max_cut_val


#==============================================================================
# Marked Consistent Max Cut
#==============================================================================


def _compute_estimated_cut(G, partition_dict, B_nn, K_nn):
    """Compute overestimated cut value on a partial marked graph"""

    buffer_dict = dict(partition_dict)

    BK = set()  # decided nodes before marked coloring
    M = set()   # marked nodes

    for node in partition_dict:
        if partition_dict[node] == gc.MARKED:
            M.add(node)
        else:
            BK.add(node)

    # Try to color the marked nodes remained
    gc.marked_nodes_could_be_cut(G,
        buffer_dict,
        M,
        B_nn,
        K_nn)

    B1K1 = set()    # decided nodes after marked coloring

    for node in M:
        if buffer_dict[node] != gc.MARKED:
            B1K1.add(node)

    nx.set_node_attributes(G, gc.PARTITION, buffer_dict)

    result = (gc.number_of_edges(G, M) +
              gc.cut_edges(G) -
              gc.subgraph_cut_edges(G, B1K1))

    return result


def _aux_marked_consistent_max_cut(G,
                                   partition_dict,
                                   Q,
                                   candidate,
                                   color,
                                   consistent_cuts,
                                   B_nn,
                                   K_nn):
    """Helper function.
    Computes an estimated maximum cut based on an overestimated
    cut value.

    """

    # Color the candidate and update its neighbors
    partition_dict[candidate] = color
    neighbors = gc.update_neighbors_labels(G,
        candidate,
        color,
        B_nn,
        K_nn)

    # Check consistency condition on candidate's neighbors
    if not gc.is_cut_consistent(G, partition_dict, neighbors, B_nn, K_nn):
        return None, 0, B_nn, K_nn

    # All nodes are been colored
    if not Q:
        est_cut_val = _compute_estimated_cut(G, partition_dict, B_nn, K_nn)
        consistent_cuts.append((partition_dict, est_cut_val, B_nn, K_nn))
        return partition_dict, est_cut_val, B_nn, K_nn

    is_colored = True

    while is_colored:
        is_colored, is_consistent = gc.could_be_cut(G,
            partition_dict,
            None,
            B_nn,
            K_nn)

        if not is_consistent:
            return None, 0, B_nn, K_nn

    # pick a new candidate
    new_candidate = _choose_new_candidate(partition_dict, Q)

    # stop if all nodes are been colored
    if new_candidate is None:
        est_cut_val = _compute_estimated_cut(G, partition_dict, B_nn, K_nn)

        consistent_cuts.append((partition_dict, est_cut_val, B_nn, K_nn))
        return partition_dict, est_cut_val, B_nn, K_nn

    # recursively blue and black coloring the candidate
    B_cut_dict, B_cut_val, BB_nn, BK_nn = _aux_marked_consistent_max_cut(G,
        dict(partition_dict),
        list(Q),
        new_candidate,
        gc.BLUE,
        consistent_cuts,
        dict(B_nn),
        dict(K_nn))

    K_cut_dict, K_cut_val, KB_nn, KK_nn = _aux_marked_consistent_max_cut(G,
        dict(partition_dict),
        list(Q),
        new_candidate,
        gc.BLACK,
        consistent_cuts,
        dict(B_nn),
        dict(K_nn))

    if B_cut_dict is None and K_cut_dict is None:
        return None, 0, None, None

    # Choose best cut according to the effective cut value
    if B_cut_val > K_cut_val:
        return B_cut_dict, B_cut_val, BB_nn, BK_nn
    return K_cut_dict, K_cut_val, KB_nn, KK_nn


def _complete_cut(G, partition_dict, Q, B_nn, K_nn):
    """Compute the maximum cut from a partial partitioned graph."""

    if not Q:
        return partition_dict, gc.cut_edges(G, partition_dict)

    candidate = _choose_new_candidate(partition_dict, Q)

    if candidate is None:
        return partition_dict, gc.cut_edges(G, partition_dict)

    # Complete cut
    B_cut_dict, B_cut_val = _aux_consistent_max_cut(G,
        dict(partition_dict),
        list(Q),
        candidate,
        gc.BLUE,
        dict(B_nn),
        dict(K_nn))[0:2]

    K_cut_dict, K_cut_val = _aux_consistent_max_cut(G,
        dict(partition_dict),
        list(Q),
        candidate,
        gc.BLACK,
        dict(B_nn),
        dict(K_nn))[0:2]

    # Choose best cut according to the effective cut value
    if B_cut_val > K_cut_val:
        return B_cut_dict, B_cut_val
    return K_cut_dict, K_cut_val


def _do_work(G,
             work_queue,
             max_cut_dict,
             max_cut_value,
             max_cuts_queue,
             Q):
    """Work function dedicated for multiprocessing only."""
    consistent_cuts = work_queue.get()

    for C in consistent_cuts:
        a_cut_dict, a_cut_value = _complete_cut(G,
            C[0],
            list(Q),
            C[2],
            C[3])

        if a_cut_value > max_cut_value:
            max_cut_value = a_cut_value
            max_cut_dict = a_cut_dict

    max_cuts_queue.put((max_cut_dict, max_cut_value))


def marked_consistent_max_cut(G, strategy=0, parallel=False):
    """Compute the maximum cut of a graph using the consistency property
    and a marking nodes strategy (0:best).

    """

    gc.init_cut(G)

    marked_nodes = set()

    # Choose the marked nodes according to the chosen marking strategy
    if strategy == 0:
        marked_nodes = gc.lowest_degree_nodes(G, G.number_of_nodes() / 2)

    elif strategy == 1:
        marked_nodes = gc.two_maximal_independent_set(G)

    elif strategy == 2:
        marked_nodes = gc.highest_degree_nodes(G, G.number_of_nodes() / 2)

    # initialization
    partition_dict = dict.fromkeys(G, gc.UNDECIDED)
    consistent_cuts = list()

    B_nn = dict.fromkeys(G, 0)
    K_nn = dict.fromkeys(G, 0)

    for n in marked_nodes:
        partition_dict[n] = gc.MARKED

    # unmarked nodes stack sorted by degree
    unmarked_deg_seq = gc.degree_nodes_sequence(G,
        set(G.nodes()) - marked_nodes)

    # Choose candidate from the unmarked nodes
    candidate = unmarked_deg_seq.pop()

    # Compute highest overestimated cut value over the unmarked nodes
    estimated_cut, estimated_val, B_nn, K_nn = _aux_marked_consistent_max_cut(
        G,
        dict(partition_dict),
        unmarked_deg_seq,
        candidate,
        gc.BLUE,
        consistent_cuts,
        B_nn,
        K_nn)

    # marked nodes stack sorted by degree
    marked_deg_seq = gc.degree_nodes_sequence(G, marked_nodes)

    # Compute the real cut based on the estimated one
    real_cut_dict, real_cut_val = _complete_cut(G,
        estimated_cut,
        list(marked_deg_seq),
        B_nn,
        K_nn)

    filtered_cons_cuts = []

    # filter the consistent cuts
    for C in consistent_cuts:
        if C[1] > real_cut_val:
            filtered_cons_cuts.append(C)

    # you find the max cut
    if not len(filtered_cons_cuts):
        gc.cut(G, real_cut_dict)
        return real_cut_dict, real_cut_val

    # find maximum cut
    max_cut_dict = dict()
    max_cut_value = 0

    if len(filtered_cons_cuts) < 50:
        parallel = False

    # use or not multiprocessing in order to find max cut
    if not parallel:
        for C in filtered_cons_cuts:
            a_cut_dict, a_cut_value = _complete_cut(G,
                C[0],
                list(marked_deg_seq),
                C[2],
                C[3])

            if a_cut_value > max_cut_value:
                max_cut_value = a_cut_value
                max_cut_dict = a_cut_dict

    else:
        cpus = cpu_count()  # Number of cores
        size = len(filtered_cons_cuts) / cpus + 1  # Work load for each process

        work_queue = Queue()  # Each element is a work for a single process
        max_cuts_queue = Queue()  # where to put results

        # Distribute the work in a queue
        for i in range(cpus):
            j = size * i
            work_queue.put(filtered_cons_cuts[j: j + size])

        # Start a number of process equivalent to the number of core installed
        processes = [Process(target=_do_work,
            args=(G,
                  work_queue,
                  max_cut_dict,
                  max_cut_value,
                  max_cuts_queue,
                  marked_deg_seq))

                     for i in range(cpus)]

        # handle the time out exception
        try:
            for p in processes:
                p.start()

            for p in processes:
                p.join()

        except TimedOutExc:  # kill processes

            for p in processes:
                p.terminate()
            return None

        # Iterate over the queue and get the maximum cut
        while not max_cuts_queue.empty():
            a_cut_dict, a_cut_value = max_cuts_queue.get()

            if a_cut_value > max_cut_value:
                max_cut_value = a_cut_value
                max_cut_dict = a_cut_dict

    gc.cut(G, max_cut_dict)
    return max_cut_dict, max_cut_value

