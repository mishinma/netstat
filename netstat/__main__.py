""" Main script """
import argparse
from netstat.parallel import (exact_distance_distribution_parallel,
                              sample_random_pairs_parallel,
                              sample_random_sources_parallel)
from netstat.util import (largest_connected_component, load_graph_data,
                          build_graph)
from netstat.statistics import compute_statistics
from netstat.approx import anf0

EXACT = 'e'
APPROX = 'a'

RAN_PAIR = 'ran-pairs'
RAN_SOURCES = 'ran-soures'
ANF = 'anf'


# python -m netstat <dsetname> --approx --sample-pairs --directed

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("fname", help="Filename to read")
# parser.add_argument("-m", "--mode", choices=[EXACT, APPROX],
#                     help="e: exact computation | a: approximat computation")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("-e", "--exact", action="store_const",
                  help="Exact computation", dest='mode', const='e')
mode.add_argument("-a", "--approx", choices=[RAN_PAIR, RAN_SOURCES, ANF],
                  dest='mode',
                  help="Approximation options: \n\t" +
                  "ran-pairs: sample random pairs\n\t" +
                  "ran-sources: sample random sources\n\t" +
                  "anf: approximate using ANF0")
graph_prop = parser.add_mutually_exclusive_group()
graph_prop.add_argument("-dir", "--directed",
                        help="Treat graph as directed graph",
                        action="store_true")
graph_prop.add_argument("-undir", "--undirected",
                        help="Treat graph as undirected graph",
                        action="store_true")
parser.set_defaults(mode='anf')
# Set some default to allow omitting the flag
parser.set_defaults(undirected=False)
args = parser.parse_args()
num_nodes, num_edges, graph_data = load_graph_data(args.fname)
graph = build_graph(num_nodes, num_edges, graph_data)
if args.directed:
    lcc = largest_connected_component(graph)
    directed = True
elif args.undirected:
    lcc = largest_connected_component(graph, connection='weak')
    directed = False
else:
    lcc = largest_connected_component(graph)
    directed = True
if args.mode == EXACT:
    # ToDo: get exact distribution
        dist_distr = exact_distance_distribution_parallel(graph,
                                                          directed=directed)
else:
    # TODO get k and num_dist - default of from commandline?
    if mode == RAN_PAIR:
        dist_distr = sample_random_pairs_parallel(graph, k, directed=directed)
    elif mode == RAN_SOURCES:
        dist_distr = sample_random_sources_parallel(graph, k, directed=directed)
    else:
        # TODO what about k and r?
        dist_distr = anf0(graph, k=5, r=0, num_dist=num_dist, directed=directed)

mean, median, diam, eff_diam = compute_statistics(dist_distr)
# ToDo: Print statistics
