"""
Graphic library with pretty drawing methods for graphs using matplotlib.

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
import pylab as pl

import gcut as gc


__all__ = ['draw_custom', 'draw_cut_graph']


def draw_custom(G, pos=None,
                node_size=1000,
                edge_width=3,
                font_size=12,
                node_color='white',
                color_map='Blues',
                edge_label=None,
                title=''):
    """Draw the graph G using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.

    """
    if pos is None:
        pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos,
        node_color=node_color,
        node_size=node_size
    )

    nx.draw_networkx_edges(G, pos,
        width=edge_width,
        edge_color=range(nx.number_of_edges(G)),
        edge_cmap=pl.get_cmap(color_map),
        edge_vmin= -10,
        edge_vmax=10)

    nx.draw_networkx_labels(G,
        pos,
        font_size=font_size)

    nx.draw_networkx_edge_labels(G, pos,
        edge_labels=nx.get_edge_attributes(G, edge_label))

    pl.title(title)
    pl.axis('off')


def draw_cut_graph(G,
                   partition_dict=None,
                   pos=None,
                   node_size=1000,
                   edge_width=3,
                   font_size=12,
                   labels=None,
                   title=''):
    """Draw a cut graph G using Matplotlib."""

    if partition_dict != None:
        nx.set_node_attributes(G, gc.PARTITION, partition_dict)

    if pos == None:
        pos = nx.circular_layout(G, scale=20)

    blue_nodes, black_nodes, undecided_nodes, marked_nodes = gc.get_partitions(G)

    # Draw nodes and edges of the first partition
    nx.draw_networkx_nodes(G, pos,
        blue_nodes,
        node_size=node_size,
        node_color='blue')

    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, blue_nodes, blue_nodes),
        width=edge_width,
        edge_color='blue')

    # Draw nodes and edges of the second partition
    nx.draw_networkx_nodes(G, pos,
        black_nodes,
        node_size=node_size,
        node_color='black')

    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, black_nodes, black_nodes),
        width=edge_width,
        edge_color='black')

    # Draw undecided nodes and edges
    nx.draw_networkx_nodes(G, pos,
        undecided_nodes,
        node_size=node_size,
        node_color='magenta')

    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, undecided_nodes, undecided_nodes),
        width=edge_width,
        edge_color='magenta')

    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, blue_nodes, undecided_nodes),
        width=edge_width,
        style='dotted',
        edge_color='magenta')

    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, undecided_nodes, black_nodes),
        width=edge_width,
        style='dotted',
        edge_color='magenta')

    # Draw marked nodes and edges
    nx.draw_networkx_nodes(G, pos,
        marked_nodes,
        node_size=node_size,
        node_color='red')

    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, marked_nodes, marked_nodes),
        width=edge_width,
        edge_color='red')

    #Draw edges beetween marked and unmarked
    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, marked_nodes, blue_nodes),
        width=edge_width,
        edge_color='orange')

    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, marked_nodes, black_nodes),
        width=edge_width,
        edge_color='orange')

    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, marked_nodes, undecided_nodes),
        width=edge_width,
        edge_color='orange')

    # Draw cut edges
    nx.draw_networkx_edges(G, pos,
        nx.edge_boundary(G, blue_nodes, black_nodes),
        width=edge_width,
        style='dashed',
        edge_color='gray')

    nx.draw_networkx_labels(G,
        pos,
        labels=labels,
        font_color='white',
        font_size=font_size,
        font_weight='bold')

    pl.title(title)
    pl.axis('off')


