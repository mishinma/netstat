import multiprocessing
import time
import sys
import h5py
import numpy as np
import polygon as pol

from scipy.sparse import csgraph


def worker(graph, chunk, directed, output):
    dist_distr = pol.get_distance_distribution(graph, chunk, directed, True)
    output.put(dist_distr)


def write_to_file(fname, data, directed=True):
    f = h5py.File('distance_distributions', 'w')
    network_name = fname.split('.')[0]

    if network_name.endswith('-clean'):
        network_name = network_name[:-6]
    if directed:
        dname = network_name + '-dir'
    else:
        dname = network_name + '-undir'

    f.create_dataset(dname, data=data)
    f.flush()
    f.close()


def merge_distributions(queue, directed):
    dist_distr = np.zeros(shape=100, dtype=np.int32)

    while not queue.empty():
        chunk_dist = queue.get()
        if chunk_dist.size > dist_distr.size:
            dist_distr.resize(chunk_dist.shape)
        dist_distr[:chunk_dist.size] += chunk_dist

    dist_distr_trimmed = np.trim_zeros(dist_distr, 'b')
    if not directed:
        dist_distr_trimmed /= 2  # because dist(u,v) = dist(v,u)
    return dist_distr_trimmed


def get_distance_distribution_parallel(graph, directed=True):
    nodes = np.arange(graph.shape[0])
    # Shared queue for output
    distributions = multiprocessing.Queue()
    cores = multiprocessing.cpu_count()
    processes = []
    # Split evenly into chunks for each available cpu
    chunks = np.array_split(nodes, cores)

    for i in range(cores):
        proc = multiprocessing.Process(target=worker, args=(graph, chunks[i], directed, distributions))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    return merge_distributions(distributions, directed)


def sample_random_pairs(graph, k, directed):
    # Sample k pairs
    samples = np.random.choice(np.arange(graph.shape[0]), (k, 2))
    dist_distr = np.zeros(shape=100, dtype=np.int32)
    for i in range(len(samples)):
        source_dist = csgraph.shortest_path(graph, method='auto', directed=directed, unweighted=True, indices=samples[i][0])
        dist = int(source_dist[samples[i][1]])
        if dist > dist_distr.size:
            dist_distr.resize((dist+1,))
        dist_distr[dist] += 1
    dist_distr_trimmed = np.trim_zeros(dist_distr, 'b')
    return dist_distr_trimmed


def sample_random_sources(graph, k, directed):
    samples = np.random.choice(np.arange(graph.shape[0]), k)
    dist_distr = pol.get_distance_distribution(graph, samples, directed)
    dist_distr_trimmed = np.trim_zeros(dist_distr, 'b')
    return dist_distr_trimmed


def get_distance_of_index(index, dist_distr):
    i = 0
    upper_bound = dist_distr[i]
    while index > upper_bound:
        i += 1
        upper_bound += dist_distr[i]
    # Index i represents distance i+1
    return i+1


def compute_median(dist_distr):
    median_index = np.median(np.arange(np.sum(dist_distr)))
    is_integer = median_index == int(median_index)
    dist = get_distance_of_index(int(median_index), dist_distr)
    if is_integer:
        return dist
    else:
        return (2*dist+1)/2.0


def compute_eff_diameter(dist_distr):
    eff_diameter_index = np.percentile(np.arange(np.sum(dist_distr)), 90, interpolation='linear')
    dist = get_distance_of_index(int(eff_diameter_index), dist_distr)
    return dist


def compute_statistics(dist_distr):
    mean = np.sum([(i+1)*dist_distr[i] for i in xrange(dist_distr.size)])/float(np.sum(dist_distr))
    median = compute_median(dist_distr)
    diameter = dist_distr.size
    eff_diameter = compute_eff_diameter(dist_distr)
    print 'Mean: %d' % mean
    print 'Median: %d' % median
    print 'Diameter: %d' % diameter
    print 'Effective Diameter: %d' % eff_diameter


if __name__ == '__main__':
    fname = sys.argv[1]
    directed = False
    if len(sys.argv) > 2:
        directed = sys.argv[2] == 'dir'
    num_nodes, num_edges, graph_data = pol.load_graph_data(fname)
    graph = pol.build_graph(num_nodes, num_edges, graph_data)
    if directed:
        lcc = pol.get_lcc(graph)
    else:
        lcc = pol.get_lcc(graph, connection='weak')
    start_time = time.time()
    dist_distr = get_distance_distribution_parallel(lcc, directed)
    # dist_distr = sample_random_pairs(lcc, 100, directed)
    # dist_distr = sample_random_sources(lcc, 100, directed)
    print dist_distr
    compute_statistics(dist_distr)
    elapsed = (time.time() - start_time)
    print '--- {} m ---'.format(elapsed / 60)
    write_to_file(fname, dist_distr, directed)

    print 'Done'
