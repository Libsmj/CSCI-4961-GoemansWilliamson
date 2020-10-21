"""
Functional library with graph cut methods and assorted utilities.

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
#    Nicola Rebagliati <nicola.rebagliati@gmail.com>

__author__ = "\n".join(['Andrea Casini (<andreacasini88@gmail.com>)',
                        'Nicola Rebagliati (<nicola.rebagliati@gmail.com>)'])

import networkx as nx
import numpy as np
import random

__all__ = ['all_possible_cuts',
           'are_undecided_nodes',
           'binary_cut',
           'compute_epsilon',
           'could_be_cut',
           'cut',
           'cut_edges',
           'degree_nodes_sequence',
           'get_partitions',
           'highest_degree_nodes',
           'init_cut',
           'is_all_isolate',
           'is_cut_consistent',
           'lowest_degree_nodes',
           'marked_nodes_could_be_cut',
           'minority_color',
           'number_of_edges',
           'partition_dictionary',
           'pick_random_nodes',
           'remove_isolates_nodes',
           'set_partitions',
           'sign_norm',
           'strong_minority_color',
           'subgraph_cut_edges',
           'two_maximal_independent_set',
           'update_neighbors_labels']

PARTITION = 'partition'
DEGREE = 'degree'

BLUE = 1
BLACK = -1

UNDECIDED = 0   # magenta
MARKED = 2      # red


#==============================================================================
# Graph's partitions
#==============================================================================


def partition_dictionary(G):
    """Return a dictionary representation of a cut"""
    return nx.get_node_attributes(G, PARTITION)


def set_partitions(G, blue_nodes, black_nodes):
    """Set node's blue class and black class"""
    init_cut(G)
    cut(G, dict.fromkeys(blue_nodes, BLUE))
    cut(G, dict.fromkeys(black_nodes, BLACK))


def get_partitions(G, nbunch=None):
    """Return all partitions of a graph G as different sets"""
    if nbunch is None:
        nbunch = G.nodes()

    blue_nodes = set()
    black_nodes = set()
    undecided_nodes = set()
    marked_nodes = set()

    for i in nbunch:

        if G.node[i][PARTITION] is BLUE:
            blue_nodes.add(i)

        elif G.node[i][PARTITION] is BLACK:
            black_nodes.add(i)

        elif G.node[i][PARTITION] is MARKED:
            marked_nodes.add(i)

        else:
            undecided_nodes.add(i)

    return blue_nodes, black_nodes, undecided_nodes, marked_nodes


def all_possible_cuts(G):
    """Return all possible cut graphs.

    Warning: demonstration porpuses only

    """
    cuts_list = []
    n = G.number_of_nodes()

    for i in range(1, 2 ** (n - 1)):
        cut_graph = nx.Graph(G)
        binary_cut(cut_graph, i)
        cuts_list.append(cut_graph)

    return cuts_list


#==============================================================================
# Cut Indices
#==============================================================================


def are_undecided_nodes(G):
    """Check the existence of undecided nodes"""
    for v in G.nodes():
        if G.node[v][PARTITION] == UNDECIDED:
            return True
    return False


def edges_between(G, a, b):
    """Return the number of edges between two sets of nodes

    Note: a and b should have no element in common.

    """
    return len(nx.edge_boundary(G, a, b))


def cut_edges(G, partition_dict=None):
    """Return the value of the cut.
    Cut edges: the number of cross edges between blue and black nodes.

    """
    nbunch = None

    if partition_dict is not None:
        cut(G, partition_dict)

    blue_nodes, black_nodes = get_partitions(G, nbunch)[0:2]
    return edges_between(G, blue_nodes, black_nodes)


def subgraph_cut_edges(G, nbunch):
    """Return cut edges in a subgraph of G induced by nbunch nodes"""
    edges = 0

    for (u, v) in G.edges_iter(nbunch):

        if v in nbunch:

            v_color = G.node[v][PARTITION]
            u_color = G.node[u][PARTITION]

            if (v_color is BLUE and u_color is BLACK or
                u_color is BLUE and v_color is BLACK):

                edges += 1

    return edges


def number_of_edges(G, nbunch):
    """Return the number of edges in a bunch of nodes."""

    edges = 0

    for (u, v) in G.edges_iter(nbunch):
        if v in nbunch:
            edges += 1

    return edges


def compute_epsilon(G):
    """Compute epsilon value of a cut graph.

    Epsilon := 1 - X / |E| where X is the number of cut edges

    """
    return round(1.0 - float(cut_edges(G)) / float(G.number_of_edges()), 3)


#==============================================================================
# Consistency Condition
#==============================================================================


def update_neighbors_labels(G, v, color, B_nn, K_nn):
    """Update neighbors colors dictionaries."""

    neighbors = G.neighbors(v)

    if color is BLUE:
        for nn in neighbors:
            B_nn[nn] += 1

    elif color is BLACK:
        for nn in neighbors:
            K_nn[nn] += 1

    return neighbors


def minority_color(G, v, B_nn, K_nn):
    """Compute the minority color of a node v.

    Minority color: tells the class of which a node should belong
    in order to maximize the cut.

    """

    degree = G.node[v][DEGREE]

    if B_nn[v] > degree / 2:
        return BLACK

    if K_nn[v] > degree / 2:
        return BLUE

    return UNDECIDED


def strong_minority_color(v, B_nn, K_nn):
    """Compute strong minority class of a node v."""

    if B_nn[v] == 0 and K_nn[v] == 0:
        return MARKED

    if B_nn[v] > K_nn[v]:
        return BLACK
    return BLUE


def is_cut_consistent(G, partition_dict, nodes_to_check, B_nn, K_nn):
    """Check consistency condition on a bunch of nodes"""

    for n in nodes_to_check:
        n_color = partition_dict[n]

        if n_color is BLUE or n_color is BLACK:
            min_clr = minority_color(G, n, B_nn, K_nn)

            if min_clr != UNDECIDED and min_clr != n_color:
                return False

    return True


def could_be_cut(G, partition_dict, nodes, B_nn, K_nn):
    """Return true if the graph is being cut according
    to the node's minority class.

    """

    if nodes is None:
        nodes = partition_dict.keys()

    is_colored = False
    is_consistent = True

    for n in nodes:
        n_color = partition_dict[n]

        if n_color is not BLUE and n_color is not BLACK:
            min_clr = minority_color(G, n, B_nn, K_nn)

            if min_clr is not UNDECIDED:

                partition_dict[n] = min_clr
                neighbors = update_neighbors_labels(G,
                    n,
                    min_clr,
                    B_nn,
                    K_nn)

                is_colored = True
                is_consistent = is_cut_consistent(G,
                    partition_dict,
                    neighbors,
                    B_nn,
                    K_nn)

                if not is_consistent:
                    return is_colored, is_consistent

    return is_colored, is_consistent


def marked_nodes_could_be_cut(G, partition_dict, marked_nodes, B_nn, K_nn):
    """Try to cut the marked nodes using the strong minority class
    for each node

    """

    for n in marked_nodes:
        min_color = strong_minority_color(n, B_nn, K_nn)
        if min_color is not MARKED:
            partition_dict[n] = min_color


#==============================================================================
# Cut Methods
#==============================================================================


def init_cut(G, nbunch=None):
    """Initialize cut: set all nodes or a bunch of nodes as undecided"""
    if nbunch is None:
        nbunch = G.nodes()

    nx.set_node_attributes(G, PARTITION, dict.fromkeys(nbunch, UNDECIDED))
    nx.set_node_attributes(G, DEGREE, G.degree(nbunch))


def _integer_to_binary(i, n):
    """Convert an integer to binary."""
    rep = bin(i)[2:]
    return ('0' * (n - len(rep))) + rep


def cut(G, partition_dict):
    """Use a partition dictionary to cut a graph"""
    nx.set_node_attributes(G, PARTITION, partition_dict)


def binary_cut(G, int_cut, bin_cut=None):
    """Cut a graph G using a binary operation."""
    if bin_cut is None:
        bin_cut = _integer_to_binary(int_cut, G.number_of_nodes())

    for i, node in enumerate(G.nodes()):
        if bin_cut[i] is '0':
            G.node[node][PARTITION] = BLACK
        else:
            G.node[node][PARTITION] = BLUE

    return nx.get_node_attributes(G, PARTITION)


#==============================================================================
# Marking Strategies Methods
#==============================================================================


def degree_nodes_sequence(G, nbunch=None, reverse=False):
    """Return a list of nodes in ascending order according to
    their degree value.

    """
    degrees_dict = G.degree(nbunch)
    return sorted(degrees_dict,
        key=lambda key: degrees_dict[key],
        reverse=reverse)


def lowest_degree_nodes(G, n_nodes):
    """Return the n nodes with lowest degree"""
    deg_node_seq = degree_nodes_sequence(G, reverse=False)
    return set(deg_node_seq[0:n_nodes])


def highest_degree_nodes(G, n_nodes):
    """Return the n nodes with highest degree"""
    deg_node_seq = degree_nodes_sequence(G, reverse=True)
    return set(deg_node_seq[0:n_nodes])


def two_maximal_independent_set(G):
    """Return a set of nodes from a bipartite subgraph"""
    i0 = nx.maximal_independent_set(G)
    i1 = nx.maximal_independent_set(G, i0)

    b = set(i0) | set(i1)

    return b


def pick_random_nodes(G, n_nodes):
    """Return a random set of n nodes from the graph"""
    return random.sample(G.nodes(), n_nodes)


#==============================================================================
# Others
#==============================================================================


def is_all_isolate(G):
    """Return true if all the nodes in the graph G are isolates"""

    nodes = G.nodes()

    if not nodes:
        return False

    for v in nodes:
        if nx.degree(G, v):
            return False
    return True


def remove_isolates_nodes(G):
    """Remove isolated nodes from a graph G.

    Isolated node: node with degree equal to zero.

    """
    isolate = False
    if not nx.is_connected(G):
        for node in G.nodes():
            if not G.degree(node):
                G.remove_node(node)
                isolate = True
    return isolate


def sign_norm(d):
    """Normalize dictionary keys according to sign function"""
    sign_d = {}
    for i in d:
        sign_d[i] = np.sign(d[i])
    return sign_d

