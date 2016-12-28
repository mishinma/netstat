import multiprocessing
import time
import sys
import h5py
import numpy as np
import polygon as pol


def worker(graph, chunk, directed, output):
    dist_distr = pol.get_distance_distribution(graph, chunk, directed)
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


def merge_distributions(queue):
    dist_distr = np.zeros(shape=100, dtype=np.int32)

    while not queue.empty():
        chunk_dist = queue.get_nowait()
        if chunk_dist.size > dist_distr.size:
            dist_distr.resize(chunk_dist.shape)
        dist_distr[:chunk_dist.size] += chunk_dist

    dist_distr_trimmed = np.trim_zeros(dist_distr, 'b')
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

    return distributions


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
    chunked_dist_distr = get_distance_distribution_parallel(lcc, directed)
    dist_distr = merge_distributions(chunked_dist_distr)
    elapsed = (time.time() - start_time)
    print '--- {} m ---'.format(elapsed / 60)
    write_to_file(fname, dist_distr, directed)

    print 'Done'
