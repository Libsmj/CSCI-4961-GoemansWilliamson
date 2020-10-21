
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
import pylab as pl

import gdraw as gd
import maxcut as mc
import gcut as gc

from timeit import Timer
from functools import partial


def execution_time(function, n_times=1, *args, **kwargs):
    """Return the execution time of a function in seconds."""
    return round(Timer(partial(function, *args, **kwargs)).timeit(n_times), 3)


def main():

    # Graph's parameters
    number_of_nodes = 20
    edge_density = 0.3

    # Random graph
    G1 = nx.erdos_renyi_graph(number_of_nodes, edge_density)

    # Some other cool graphs (check NetworkX library)
    G2 = nx.star_graph(number_of_nodes)
    G3 = nx.path_graph(number_of_nodes)
    G5 = nx.dodecahedral_graph()

    # Choose a graph to cut
    G = G1

    # Check graph connectivity
    if not nx.is_connected(G):
        raise ValueError('Graph is not connected.')

    # Plot initial graph
    pl.figure()
    gd.draw_custom(G, title='Initial Graph')

    # Consistent max cut
    t1 = execution_time(mc.consistent_max_cut, 1, G)
    print('Time Consistent Max Cut: ' + str(t1))
    print(str(gc.cut_edges(G)) + ' edges has been cut.\n')

    # Marked consistent max cut
    t2 = execution_time(mc.marked_consistent_max_cut, 1, G)
    print('Time Marked Consistent Max Cut: ' + str(t2))
    print(str(gc.cut_edges(G)) + ' edges has been cut.\n')

    # Plot exact cut solution graph
    pl.figure()
    gd.draw_cut_graph(G, title='Exact Max cut solution')

    # Recursive Spectral Cut (Trevisan approach)
    t3 = execution_time(mc.recursive_spectral_cut, 1, G)
    print('Time Recursive Spectral Cut: ' + str(t3))
    print(str(gc.cut_edges(G)) + ' edges has been cut.\n')

    # Plot approximated cut solution graph
    pl.figure()
    gd.draw_cut_graph(G, title='Approximated Max cut solution')

    # Show figures
    pl.show()


if __name__ == '__main__':

    main()

    pass
