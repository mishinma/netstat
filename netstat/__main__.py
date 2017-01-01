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


def run_netstat(fname, mode, clean=False, parallel=True, directed=False,
                connection='strong', **kwargs):

    if clean:
        print "Cleaning the file"
        root, ext = os.path.splitext(fname)
        fname_new = root + '-clean' + ext
        clean_file(fname, fname_new)
        fname = fname_new

    print "Loading the graph data..."
    num_nodes, num_edges, graph_data = load_graph_data(fname)

    print "Building the adjacency matrix..."
    graph = build_graph(num_nodes, num_edges, graph_data)

    print "Finding the largest connected component..."
    lcc = largest_connected_component(graph, connection=connection)

    print "Computing the distance distribution..."

    start_time = time.time()

    if mode == EXACT:
        dist_distr = run_exact(lcc, directed=directed, parallel=parallel)

    elif mode == RAND_PAIRS or mode == RAND_SOURCES:
        try:
            k = kwargs['k']
            if k > lcc.shape[0]:
                print "Sample size is bigger than amount of available nodes. "\
                      "Running with complete graph."
                k = lcc.shape[0]
        except KeyError:
            p = kwargs['p']
            if p > 100:
                print "Provided percentage is larger than 100%. "\
                      "Running with complete graph."
                p = 100
            k = int(float(p)/100*lcc.shape[0])

        print "Sample size: %d (~%d%%)" % (k, int(float(k)/lcc.shape[0]*100))
        if mode == RAND_PAIRS:
            dist_distr = run_random_pairs(lcc, k,
                                          directed=directed, parallel=parallel)
        else:
            dist_distr = run_random_sources(lcc, k,
                                            directed=directed, parallel=parallel)

    else:
        k, r, num_dist = kwargs['k'], kwargs['r'], kwargs['num_dist']
        dist_distr = run_anf(graph, k=k, r=r, num_dist=num_dist,
                             directed=directed, parallel=parallel)

    stats = compute_statistics(dist_distr)
    print_statistics(*stats)

    elapsed = (time.time() - start_time)
    print "--- {} m ---".format(elapsed / 60)


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
    randp_arg = parser_randp.add_mutually_exclusive_group()
    randp_arg.add_argument('-k', type=int, nargs=1,
                           help='Sample size (number of nodes)')
    randp_arg.add_argument('-p', type=int, nargs=1, default=10,
                           help='Sample size (percentage of total nodes)')
    parser_randp.set_defaults(mode=RAND_PAIRS)

    # Random Sources parser
    parser_rands = subparsers.add_parser('rands', help='Random Sources help')
    rands_arg = parser_rands.add_mutually_exclusive_group()
    rands_arg.add_argument('-k', type=int, nargs=1,
                           help='Sample size (number of nodes)')
    rands_arg.add_argument('-p', type=int, nargs=1, default=10,
                           help='Sample size (percentage of total nodes)')
    parser_rands.set_defaults(mode=RAND_SOURCES)

    # ANF parser
    parser_anf = subparsers.add_parser('anf', help='ANF0 help')
    parser_anf.add_argument('-r', type=int, nargs=1, default=0,
                            help="Num bits to add to bitstring of length log(n)")
    parser_anf.add_argument('-k', type=int, nargs=1, default=3,
                            help="Num parallel approximations")
    parser_anf.add_argument('-d', type=int, nargs=1, default=20,
                            help="Num distances to approximate")
    parser_anf.set_defaults(mode=ANF)

    args = parser.parse_args()
    # print args

    parallel = not args.single

    if not args.undirected:
        connection, directed = 'strong', True
    else:
        connection, directed = 'weak', False

    mode = args.mode

    kwargs = dict()

    if mode == EXACT:
        pass

    elif mode == RAND_PAIRS or mode == RAND_SOURCES:
        if args.k:
            try:
                k = args.k[0]
            except TypeError:
                k = args.k

            kwargs['k'] = k
        else:
            try:
                p = args.p[0]
            except TypeError:
                p = args.p

            kwargs['p'] = p

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

        kwargs['r'] = r
        kwargs['k'] = k
        kwargs['num_dist'] = num_dist

    run_netstat(fname=args.fname, mode=args.mode, clean=args.clean, connection=connection,
         directed=directed, parallel=parallel, **kwargs)


if __name__ == "__main__":
    main()
