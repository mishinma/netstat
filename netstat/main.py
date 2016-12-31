import multiprocessing
import time
import sys
import h5py
import numpy as np
import polygon as pol

from netstat.statistics import compute_statistics
from netstat.graph import breadth_first_search


def source_worker(graph, chunk, directed, output):
    dist_distr = pol.get_distance_distribution(graph, chunk, directed)
    output.put(dist_distr)


def pair_worker(graph, chunk, directed, output):
    dist_distr = np.zeros(shape=pol.MAX_NUM_DIST, dtype=np.int64)
    for pair in chunk:
        source_dist = breadth_first_search(graph, i_start=pair[0], directed=directed)
        dist = source_dist[1][pair[1]]
        if dist > dist_distr.size:
            dist_distr.resize(dist+1)
        dist_distr[dist] += 1
    output.put(dist_distr)


def write_to_file(fname, data, directed, time_elapsed):
    f = h5py.File('distance_distributions.h5', 'a')
    network_name = fname.split('.')[0]

    if network_name.endswith('-clean'):
        network_name = network_name[:-6]
    if directed:
        dname = '{}-dir:{}'.format(network_name, time_elapsed)
    else:
        dname = '{}-undir:{}'.format(network_name, time_elapsed)

    f.create_dataset(dname, data=data)
    f.flush()
    f.close()


def merge_distributions(queue, directed):
    dist_distr = np.zeros(shape=pol.MAX_NUM_DIST, dtype=np.int64)

    while not queue.empty():
        chunk_dist = queue.get()
        if chunk_dist.size > dist_distr.size:
            dist_distr.resize(chunk_dist.shape)
        dist_distr[:chunk_dist.size] += chunk_dist

    return dist_distr


def get_distance_distribution_parallel(graph, nodes=None, directed=True):
    if nodes is None:
        nodes = np.arange(graph.shape[0])
    # Shared queue for output
    distributions = multiprocessing.Queue()
    cores = multiprocessing.cpu_count()
    processes = []
    # Split evenly into chunks for each available cpu
    chunks = np.array_split(nodes, cores)

    for i in range(cores):
        proc = multiprocessing.Process(target=source_worker, args=(graph, chunks[i], directed, distributions))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    return merge_distributions(distributions, directed)


def sample_random_pairs(graph, k, directed):
    samples = np.random.choice(np.arange(graph.shape[0]), (k, 2))
    dist_distr = np.zeros(shape=pol.MAX_NUM_DIST, dtype=np.int64)
    for pair in samples:
        source_dist = breadth_first_search(graph, i_start=pair[0], directed=directed)
        dist = source_dist[1][pair[1]]
        if dist > dist_distr.size:
            dist_distr.resize(dist+1)
        dist_distr[dist] += 1
    return dist_distr


def sample_random_pairs_parallel(graph, k, directed):
    distributions = multiprocessing.Queue()
    cores = multiprocessing.cpu_count()
    processes = []
    samples = np.random.choice(np.arange(graph.shape[0]), (k, 2))
    # Split evenly into chunks for each available cpu
    chunks = np.array_split(samples, cores)

    for i in range(cores):
        proc = multiprocessing.Process(target=pair_worker, args=(graph, chunks[i], directed, distributions))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    return merge_distributions(distributions, directed)


def sample_random_sources(graph, k, directed):
    samples = np.random.choice(np.arange(graph.shape[0]), k)
    return pol.get_distance_distribution(graph, samples, directed)


def sample_random_sources_parallel(graph, k, directed):
    samples = np.random.choice(np.arange(graph.shape[0]), k)
    return get_distance_distribution_parallel(graph, samples, directed)


def print_statistics(dist_distr):
    mn, mdn, diam, eff_diam = compute_statistics(dist_distr)
    print "Mean {}".format(mn)
    print "Median {}".format(mdn)
    print "Diameter {}".format(diam)
    print "Eff diameter {}".format(eff_diam)


def read_datasets():
    f = h5py.File("distance_distributions.h5", "r")
    for name in f:
        print '---- {} ----'.format(name)
        print_statistics(f[name][:])


def compute_distribution(args):
    fname = args[1]
    directed = False
    if len(args) > 2:
        directed = args[2] == 'dir'
    num_nodes, num_edges, graph_data = pol.load_graph_data(fname)
    graph = pol.build_graph(num_nodes, num_edges, graph_data)
    if directed:
        lcc = pol.get_lcc(graph, connection='strong')
    else:
        lcc = pol.get_lcc(graph, connection='weak')
    start_time = time.time()
    dist_distr = get_distance_distribution_parallel(lcc, directed=directed)
    # dist_distr = sample_random_pairs_parallel(lcc, 100, directed)
    # dist_distr = sample_random_sources_parallel(lcc, 100, directed)

    elapsed = (time.time() - start_time)
    print '--- {} m ---'.format(elapsed / 60)
    write_to_file(fname, dist_distr, directed, elapsed)


if __name__ == '__main__':
    compute_distribution(sys.argv)
    # read_datasets()
    print 'Done'
