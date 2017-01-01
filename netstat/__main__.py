""" Main script """

import argparse
import time

from netstat.util import largest_connected_component, \
    load_graph_data, build_graph, print_statistics
from netstat.statistics import compute_statistics
from netstat.approx import anf0
from netstat.parallel import exact_distance_distribution_parallel, \
    sample_random_pairs_parallel, sample_random_sources_parallel

EXACT = 'exact'
RAND_PAIRS = 'randp'
RAND_SOURCES = 'rands'
ANF = 'anf'


def run_exact(graph, directed=True):
    dist_distr = exact_distance_distribution_parallel(graph,
                                         directed=directed)
    return dist_distr


def run_random_pairs(graph, k, directed=True):
    dist_distr = sample_random_pairs_parallel(graph, k, directed=directed)
    return dist_distr


def run_random_sources(graph, k, directed=True):
    dist_distr = sample_random_sources_parallel(graph, k, directed=directed)
    return dist_distr


def run_anf(graph, k, r, num_dist, directed=True):
    dist_distr = anf0(graph, k=k, r=r, num_dist=num_dist, directed=directed)
    return dist_distr


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     prog='netstat')

    parser.add_argument("fname", help="Filename to read")
    parser.add_argument("-undir", "--undirected", default=False,
                              help="Treat graph as undirected", action="store_true")

    subparsers = parser.add_subparsers(help='sub-command help')

    # Exact parser
    parser_exact = subparsers.add_parser('exact', help='Exact help')
    parser_exact.set_defaults(mode=EXACT)

    # Random Pairs parser
    parser_randp = subparsers.add_parser('randp', help='Random Pairs help')
    parser_randp.add_argument('-k', type=int, nargs=1, default=2000,
                              help='Sample size')
    parser_randp.set_defaults(mode=RAND_PAIRS)

    # Random Sources parser
    parser_rands = subparsers.add_parser('rands', help='Random Sources help')
    parser_rands.add_argument('-k', type=int, nargs=1, default=2000,
                              help='Sample size')
    parser_rands.set_defaults(mode=RAND_SOURCES)

    # ANF parser
    parser_anf = subparsers.add_parser('anf', help='ANF0 help')
    parser_anf.add_argument('-r', type=int, nargs=1, default=0,
                            help="Num bits to add to bitstring of length log(n)")
    parser_anf.add_argument('-k', type=int, nargs=1, default=0,
                            help="Num parallel approximations")
    parser_anf.add_argument('-num-dist', type=int, nargs=1, default=20,
                            help="Num distances to approximate")
    parser_anf.set_defaults(mode=ANF)

    args = parser.parse_args()
    print args  # For debugging purposes

    start_time = time.time()

    print "Loading the graph data..."
    num_nodes, num_edges, graph_data = load_graph_data(args.fname)

    print "Building the adjacency matrix..."
    graph = build_graph(num_nodes, num_edges, graph_data)

    if not args.undirected:
        connection, directed = 'strong', True
    else:
        connection, directed = 'weak', False

    print "Finding the largest connected component..."
    lcc = largest_connected_component(graph, connection=connection)

    print "Computing the distance distribution..."
    if args.mode == EXACT:
        dist_distr = run_exact(lcc, directed=directed)

    elif args.mode == RAND_PAIRS:
        try:
            k = args.k[0]
        except TypeError:
            k = args.k

        dist_distr = run_random_pairs(lcc, k, directed=directed)

    elif args.mode == RAND_SOURCES:
        try:
            k = args.k[0]
        except TypeError:
            k = args.k

        dist_distr = run_random_sources(lcc, k, directed=directed)

    else:
        try:
            r = args.r[0]
        except TypeError:
            r = args.r
        try:
            k = args.k[0]
        except TypeError:
            k = args.k
        try:
            num_dist = args.num_dist[0]
        except TypeError:
            num_dist = args.num_dist

        dist_distr = run_anf(graph, k=k, r=r, num_dist=num_dist, directed=directed)

    stats = compute_statistics(dist_distr)
    print_statistics(*stats)

    elapsed = (time.time() - start_time)
    print "--- {} m ---".format(elapsed / 60)


if __name__ == '__main__':
    main()
