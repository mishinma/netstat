""" Main script """

import argparse
import os
import time

from netstat.clean import clean_file
from netstat.util import largest_connected_component, \
    load_graph_data, build_graph, print_statistics
from netstat.statistics import compute_statistics
from netstat.exact import exact_distance_distribution
from netstat.approx import anf0, sample_random_pairs, sample_random_sources
from netstat.parallel import exact_distance_distribution_parallel, \
    sample_random_pairs_parallel, sample_random_sources_parallel

EXACT = 'exact'
RAND_PAIRS = 'randp'
RAND_SOURCES = 'rands'
ANF = 'anf'


def run_exact(graph, directed=True, parallel=True):
    if parallel:
        dist_distr = exact_distance_distribution_parallel(graph,
                                             directed=directed)
    else:
        dist_distr = exact_distance_distribution(graph,
                                                 directed=directed)
    return dist_distr


def run_random_pairs(graph, k, directed=True, parallel=True):
    if parallel:
        dist_distr = sample_random_pairs_parallel(graph, k, directed=directed)
    else:
        dist_distr = sample_random_pairs(graph, k, directed=directed)
    return dist_distr


def run_random_sources(graph, k, directed=True, parallel=True):
    if parallel:
        dist_distr = sample_random_sources_parallel(graph, k, directed=directed)
    else:
        dist_distr = sample_random_sources(graph, k, directed=directed)
    return dist_distr


def run_anf(graph, k, r, num_dist, directed=True, parallel=False):
    if parallel:
        print "Parallel version not implemented - running single core..."
    dist_distr = anf0(graph, k=k, r=r, num_dist=num_dist, directed=directed)
    return dist_distr


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     prog='netstat')

    parser.add_argument("fname", help="Filename to read")
    parser.add_argument("-c", "--clean", default=False, action="store_true",
                        help="Clean file")
    parser.add_argument("-s", "--single", default=False, action="store_true",
                        help="Run single core version")
    parser.add_argument("-u", "--undirected", default=False, action="store_true",
                        help="Treat graph as undirected")

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
    parser_anf.add_argument('-d', type=int, nargs=1, default=20,
                            help="Num distances to approximate")
    parser_anf.set_defaults(mode=ANF)

    args = parser.parse_args()
    print args  # For debugging purposes

    parallel = not args.single

    start_time = time.time()

    fname = args.fname

    if args.clean:
        print "Cleaning the file"
        root, ext = os.path.splitext(fname)
        fname_new = root + '-clean' + ext
        clean_file(fname, fname_new)
        fname = fname_new

    print "Loading the graph data..."
    num_nodes, num_edges, graph_data = load_graph_data(fname)

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
        dist_distr = run_exact(lcc, directed=directed, parallel=parallel)

    elif args.mode == RAND_PAIRS:
        try:
            k = args.k[0]
        except TypeError:
            k = args.k

        dist_distr = run_random_pairs(lcc, k,
                                      directed=directed, parallel=parallel)

    elif args.mode == RAND_SOURCES:
        try:
            k = args.k[0]
        except TypeError:
            k = args.k

        dist_distr = run_random_sources(lcc, k,
                                        directed=directed, parallel=parallel)

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
            num_dist = args.d[0]
        except TypeError:
            num_dist = args.d

        dist_distr = run_anf(graph, k=k, r=r, num_dist=num_dist,
                             directed=directed, parallel=parallel)

    stats = compute_statistics(dist_distr)
    print_statistics(*stats)

    elapsed = (time.time() - start_time)
    print "--- {} m ---".format(elapsed / 60)


if __name__ == '__main__':
    main()
