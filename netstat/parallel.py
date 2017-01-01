import multiprocessing
import numpy as np

from netstat.graph import breadth_first_search
from netstat.exact import exact_distance_distribution, MAX_NUM_DIST


def source_worker(graph, chunk, directed, output):
    dist_distr = exact_distance_distribution(graph, chunk, directed)
    output.put(dist_distr)


def pair_worker(graph, chunk, directed, output):
    dist_distr = np.zeros(shape=MAX_NUM_DIST, dtype=np.int64)
    for pair in chunk:
        source_dist = breadth_first_search(graph, i_start=pair[0], directed=directed)
        dist = source_dist[1][pair[1]]
        if dist > dist_distr.size:
            dist_distr.resize(dist+1)
        dist_distr[dist] += 1
    output.put(dist_distr)


def merge_distributions(queue):
    dist_distr = np.zeros(shape=MAX_NUM_DIST, dtype=np.int64)

    while not queue.empty():
        chunk_dist = queue.get()
        if chunk_dist.size > dist_distr.size:
            dist_distr.resize(chunk_dist.shape)
        dist_distr[:chunk_dist.size] += chunk_dist

    return dist_distr


def exact_distance_distribution_parallel(graph, nodes=None, directed=True):
    if nodes is None:
        nodes = np.arange(graph.shape[0])
    # Shared queue for output
    distributions = multiprocessing.Queue()
    cores = multiprocessing.cpu_count()
    processes = []
    # Split evenly into chunks for each available cpu
    chunks = np.array_split(nodes, cores)

    for i in range(cores):
        proc = multiprocessing.Process(target=source_worker,
                                       args=(graph, chunks[i], directed, distributions))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    return merge_distributions(distributions)


def sample_random_pairs_parallel(graph, k, directed):
    distributions = multiprocessing.Queue()
    cores = multiprocessing.cpu_count()
    processes = []
    samples = np.random.choice(np.arange(graph.shape[0]), (k, 2))
    # Split evenly into chunks for each available cpu
    chunks = np.array_split(samples, cores)

    for i in range(cores):
        proc = multiprocessing.Process(target=pair_worker,
                                       args=(graph, chunks[i], directed, distributions))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    return merge_distributions(distributions)


def sample_random_sources_parallel(graph, k, directed):
    samples = np.random.choice(np.arange(graph.shape[0]), k)
    return exact_distance_distribution_parallel(graph, samples, directed)
