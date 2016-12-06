import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph


dirname = os.path.expanduser("~/docs/netstat/data")
fname = os.path.join(dirname, "wiki-Vote_clean.txt")

# Nodes: 7115 Edges: 103689
num_nodes = 7115
num_edges = 103689

graph_data = pd.read_csv(fname, sep=',', names=['FromNodeId', 'ToNodeId'],
                 dtype={'FromNodeId': np.int64, 'ToNodeId': np.int64}, skiprows=4)

graph_data = graph_data.values

# indptr = np.hstack([0, np.cumsum(np.bincount(graph_data[:,0]))])
row_ind = graph_data[:, 0]
col_ind = graph_data[:, 1]
data = np.full(shape=num_edges, fill_value=1, dtype=np.uint8)

graph = csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, num_nodes))

dist_matrix = csgraph.shortest_path(graph, method='D', directed=False, unweighted=True)
import pdb; pdb.set_trace()
#
# num_components, labels = csgraph.connected_components(graph, directed=False)
# foo = np.bincount(labels, minlength=num_components)

#
#
# wcc = csgraph.connected_components(graph)
# _, counts = np.unique(wcc[1], return_counts=True)
#
# print "Num edges LWCC: {}".format(max(counts))
#
# scc = csgraph.connected_components(graph, connection='strong')
# _, counts = np.unique(scc[1], return_counts=True)
# print "Num edges LSCC: {}".format(max(counts))


