import numpy as np
cimport numpy as np

from scipy.sparse.csgraph._validation import validate_graph

cimport cython
from libc cimport stdlib

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

# NULL_IDX is the index used in predecessor matrices to store a non-path
DEF NULL_IDX = -9999

cpdef breadth_first_search(csgraph, i_start, directed=True):
    """ Return a breadth-first ordering starting with specified node """

    csgraph = validate_graph(csgraph, directed, dense_output=False)
    cdef int N = csgraph.shape[0]

    cdef np.ndarray node_list = np.empty(N, dtype=ITYPE)
    cdef np.ndarray distances = np.empty(N, dtype=ITYPE)
    node_list.fill(NULL_IDX)
    distances.fill(NULL_IDX)

    if directed:
        length = _breadth_first_directed(i_start,
                                csgraph.indices, csgraph.indptr,
                                node_list, distances)
    else:
        csgraph_T = csgraph.T.tocsr()
        length = _breadth_first_undirected(i_start,
                                           csgraph.indices, csgraph.indptr,
                                           csgraph_T.indices, csgraph_T.indptr,
                                           node_list, distances)

    return node_list[:length], distances


cdef unsigned int _breadth_first_directed(
                           unsigned int head_node,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indices,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] distances):
    # Inputs:
    #  head_node: (input) index of the node from which traversal starts
    #  indices: (input) CSR indices of graph
    #  indptr:  (input) CSR indptr of graph
    #  node_list: (output) breadth-first list of nodes
    #  distances: (output) list of distances from `head_node`.
    #   Should be initialized to NULL_IDX
    # Returns:
    #  n_nodes: the number of nodes in the breadth-first tree
    cdef unsigned int i, pnode, cnode
    cdef unsigned int i_nl, i_nl_end
    cdef unsigned int N = node_list.shape[0]

    node_list[0] = head_node
    distances[head_node] = 0
    i_nl = 0
    i_nl_end = 1

    while i_nl < i_nl_end:
        pnode = node_list[i_nl]

        for i in range(indptr[pnode], indptr[pnode + 1]):
            cnode = indices[i]
            if (cnode == head_node):
                continue
            elif (distances[cnode] == NULL_IDX):
                node_list[i_nl_end] = cnode
                distances[cnode] = distances[pnode] + 1
                i_nl_end += 1

        i_nl += 1

    return i_nl


cdef unsigned int _breadth_first_undirected(
                           unsigned int head_node,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indices1,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr1,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indices2,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr2,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] distances):
    # Inputs:
    #  head_node: (input) index of the node from which traversal starts
    #  indices1: (input) CSR indices of graph
    #  indptr1:  (input) CSR indptr of graph
    #  indices2: (input) CSR indices of transposed graph
    #  indptr2:  (input) CSR indptr of transposed graph
    #  node_list: (output) breadth-first list of nodes
    #  distances: (output) list of distances from `head_node`.t
    #                tree.  Should be initialized to NULL_IDX
    # Returns:
    #  n_nodes: the number of nodes in the breadth-first tree
    cdef unsigned int i, pnode, cnode
    cdef unsigned int i_nl, i_nl_end
    cdef unsigned int N = node_list.shape[0]

    node_list[0] = head_node
    distances[head_node] = 0
    i_nl = 0
    i_nl_end = 1

    while i_nl < i_nl_end:
        pnode = node_list[i_nl]

        for i in range(indptr1[pnode], indptr1[pnode + 1]):
            cnode = indices1[i]
            if (cnode == head_node):
                continue
            elif (distances[cnode] == NULL_IDX):
                node_list[i_nl_end] = cnode
                distances[cnode] = distances[pnode] + 1
                i_nl_end += 1

        for i in range(indptr2[pnode], indptr2[pnode + 1]):
            cnode = indices2[i]
            if (cnode == head_node):
                continue
            elif (distances[cnode] == NULL_IDX):
                node_list[i_nl_end] = cnode
                distances[cnode] = distances[pnode] + 1
                i_nl_end += 1

        i_nl += 1

    return i_nl
