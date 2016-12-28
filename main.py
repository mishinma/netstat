import multiprocessing
import time
import sys
import h5py
import numpy as np

from netstat import polygon as pol


def worker(graph, chunk, directed, output):
    dist_distr = pol.get_distance_distribution(graph, chunk, directed)
    output.put(dist_distr)


if __name__ == '__main__':
    fname = sys.argv[1]
    num_nodes, num_edges, graph_data = pol.load_graph_data(fname)
    graph = pol.build_graph(num_nodes, num_edges, graph_data)
    lcc = pol.get_lcc(graph, connection='weak')
    nodes = np.arange(lcc.shape[0])
    # Shared queue for output
    distributions = multiprocessing.Queue()
    cores = multiprocessing.cpu_count()
    processes = []
    # Split evenly into chunks for each available cpu
    chunks = np.array_split(nodes, cores)
    start_time = time.time()

    for i in range(cores):
        proc = multiprocessing.Process(target=worker, args=(lcc, chunks[i], False, distributions))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()
    print "Finished calculation in %s m" % str((time.time()-start_time)/60)
    print "merging outputs..."
    dist_distr = np.zeros(shape=100, dtype=np.int32)

    while not distributions.empty():
        chunk_dist = distributions.get_nowait()
        if chunk_dist.size > dist_distr.size:
            dist_distr.resize(chunk_dist.shape)
        dist_distr[:chunk_dist.size] += chunk_dist

    elapsed = (time.time() - start_time)
    print "--- {} m ---".format(elapsed / 60)
    dist_distr_trimmed = np.trim_zeros(dist_distr, 'b')
    print dist_distr_trimmed

    f = h5py.File('distance_distributions', 'w')
    network_name = fname.split('.')[0]
    if network_name.endswith('-clean'):
        network_name = network_name[:-6]
    dname = network_name + '-undir'
    f.create_dataset(dname, data=dist_distr)
    f.flush()
    f.close()

    print "Done"
