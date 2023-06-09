# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-03 14:21:40
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-04 21:33:56
# %%
import numpy as np
import networkx as nx
import math
from scipy import (
    sparse,
)  # needed for TREXPIC, Isomap (singular value decomposition) and Laplacian Eigenmaps (eigendecomposition)


# A function for creating a weighted version of a graph based on its d-dimensional Euclidean embedding generated by the exponentialized version of Isomap.
# G is the (weighted) NetworkX Graph to be embedded (If link weights are inputted, these must mean distances, i.e. larger value:=less similarity or weaker connection!)
# d is the number of dimensions of the space to which the network will be embedded
# q>0 is the multiplying factor in the exponent in the matrix to be reduced (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal matrix elements towards 0). If None, its default setting will be used.
# The function returns the weighted NetworkX Graph G_embWeighted and the dictionary Coord that assigns to each node name a NumPy array of d elements containing the Cartesian coordinates of the given network node in the d-dimensional Euclidean space.
# Example for function call:
#   [G_w,positionDict]=embedding.expISO(G,d)
def expISO(G, d, q=None):
    N = len(G)  # the number of nodes in graph G
    if N < d:
        print(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.ISO can not be larger than the number of nodes, i.e. "
            + str(N)
            + ".\n\n"
        )

    # create the matrix to be reduced
    shortestPathLengthsDict = dict(nx.shortest_path_length(G, weight="weight"))
    # shortestPathLengthDict[source][target]=length of the shortest path from the source node to the target node

    if q == None:  # use the default setting of the multiplying factor
        maxSPL = 0  # initialization of the largest occurring value of the shortest path lengths
        for sour in shortestPathLengthsDict.keys():
            for targ in shortestPathLengthsDict[sour].keys():
                if shortestPathLengthsDict[sour][targ] > maxSPL:
                    maxSPL = shortestPathLengthsDict[sour][targ]
        qmin = math.log(1.0 / 0.9999) * maxSPL
        qmax = math.log(10) * maxSPL
        q = math.exp((math.log(qmin) + math.log(qmax)) / 2)
        print("The default q multiplying factor is " + str(q) + ".")

    # create the matrix of expected Euclidean distances
    listOfNodes = list(G.nodes())
    D = np.zeros(
        (N, N)
    )  # D[i][j] is the expected Euclidean distance calculated from the length of the shortest path from the ith node to the jth node, where the nodes are ordered according to listOfNodes
    for rowID in range(1, N):
        for colID in range(rowID):  # iterate over the node pairs
            # since the network to be embedded is undirected, the matrix D must be symmetric
            # the distance of each node from itself is 0 -> the diagonal elements of D must remain 0
            try:
                expDistance = math.exp(
                    -q / shortestPathLengthsDict[listOfNodes[rowID]][listOfNodes[colID]]
                )
            except (
                KeyError
            ):  # there is no path between the given two nodes -> set the distance to the possible highest value (SPL=infinity -> exp(-q/SPL)=exp(0)=1)
                expDistance = 1.0
            except (
                ZeroDivisionError
            ):  # two different nodes are connected with a link/links of weight 0
                expDistance = 0.0
            D[rowID, colID] = expDistance
            D[colID, rowID] = D[rowID, colID]
    # in the original (non-exponential) algorithm: D=np.array([[shortestPathLengthsDict[s][t] for t in listOfNodes] for s in listOfNodes])

    # centering, i.e. the creation of the matrix of expected inner products
    H = np.identity(N) - np.ones((N, N)) / N  # centering matrix
    IP = (
        -np.matmul(np.matmul(H, np.multiply(D, D)), H) / 2
    )  # multiply=element-wise product

    # dimension reduction
    if d == N:
        U, S, VT = np.linalg.svd(
            IP
        )  # find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
        # note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else:  # d<N: only the first d largest singular values (and the corresponding singular vectors) are retained
        U, S, VT = sparse.linalg.svds(
            IP, d, which="LM", solver="arpack"
        )  # the singular values are ordered from the smallest to the largest in S (increasing order)
        # reverse the order of the singular values to obtain a decreasing order, and reverse the order of singular vectors accordingly:
        S = S[::-1]
        U = U[:, ::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues < d:
        print(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.expISO can not be larger than the number of positive singular values of the inner product matrix, i.e. "
            + str(numOfPositiveSingularValues)
            + ".\n\n"
        )
    Ssqrt = np.sqrt(S)

    # create the dictionary of node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node in the Euclidean space
    Coord = (
        {}
    )  # initialize a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space
    nodeIndex = 0
    for nodeName in listOfNodes:
        # calculate the position of the given node
        Uarray = U[nodeIndex, :]
        Coord[nodeName] = np.multiply(
            Uarray, Ssqrt
        )  # the jth element is the jth Cartesian coordinate of the node named nodeName in the reduced space
        # we could also use: Varray = VT[:,nodeIndex] and then Coord[nodeName] = np.multiply(Varray,Ssqrt)
        nodeIndex = nodeIndex + 1

    # create the graph with embedding-based link weights
    G_embWeighted = nx.Graph()
    G_embWeighted.add_nodes_from(G.nodes)  # keep the node order of the original graph
    for i, j in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
        # assign a weight to the i-j edge:
        w = 1.0 - (
            np.inner(Coord[i], Coord[j])
            / (np.linalg.norm(Coord[i]) * np.linalg.norm(Coord[j]))
        )  # weight=cosine distance=1-cos(dTheta)
        if w < 0:  # a numerical error has occurred
            w = 0.0
        if w > 2:  # a numerical error has occurred
            w = 2.0
        G_embWeighted.add_edge(i, j, weight=w)

    return [G_embWeighted, Coord]


# A function for creating a weighted version of a graph based on its d-dimensional Euclidean embedding generated by the exponentialized version of Isomap.
# G is the (weighted) NetworkX Graph to be embedded (If link weights are inputted, these must mean distances, i.e. larger value:=less similarity or weaker connection!)
# d is the number of dimensions of the space to which the network will be embedded
# q>0 is the multiplying factor in the exponent in the matrix to be reduced (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal matrix elements towards 0). If None, its default setting will be used.
# The function returns the weighted NetworkX Graph G_embWeighted and the dictionary Coord that assigns to each node name a NumPy array of d elements containing the Cartesian coordinates of the given network node in the d-dimensional Euclidean space.
# Example for function call:
#   [G_w,positionDict]=embedding.expISO(G,d)
def expISOv2(G, d, q=None, verbose=False):
    A = to_scipy_matrix(G, return_node_labels=False)
    N = A.shape[0]  # the number of nodes in graph G
    if N < d:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.ISO can not be larger than the number of nodes, i.e. "
            + str(N)
            + ".\n\n"
        )

    # create the matrix to be reduced
    shortestPathLengthMatrix = sparse.csgraph.shortest_path(A, directed=False)

    if q == None:  # use the default setting of the multiplying factor
        maxSPL = np.ma.masked_invalid(shortestPathLengthMatrix).max()
        qmin = np.log(1.0 / 0.9999) * maxSPL
        qmax = np.log(10) * maxSPL
        q = np.exp((np.log(qmin) + np.log(qmax)) / 2)
        if verbose:
            print("The default q multiplying factor is " + str(q) + ".")

    # create the matrix of expected Euclidean distances
    D = np.zeros((N, N))
    # create the matrix of expected Lorentz products
    D = np.exp(-q / shortestPathLengthMatrix)
    # in the original (non-exponential) algorithm: D=np.array([[shortestPathLengthsDict[s][t] for t in listOfNodes] for s in listOfNodes])

    # centering, i.e. the creation of the matrix of expected inner products
    H = np.identity(N) - np.ones((N, N)) / N  # centering matrix
    IP = (
        -np.matmul(np.matmul(H, np.multiply(D, D)), H) / 2
    )  # multiply=element-wise product

    # dimension reduction
    if d == N:
        U, S, VT = np.linalg.svd(
            IP
        )  # find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
        # note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else:  # d<N: only the first d largest singular values (and the corresponding singular vectors) are retained
        U, S, VT = sparse.linalg.svds(
            IP, d, which="LM", solver="arpack"
        )  # the singular values are ordered from the smallest to the largest in S (increasing order)
        # reverse the order of the singular values to obtain a decreasing order, and reverse the order of singular vectors accordingly:
        S = S[::-1]
        U = U[:, ::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues < d:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.expISO can not be larger than the number of positive singular values of the inner product matrix, i.e. "
            + str(numOfPositiveSingularValues)
            + ".\n\n"
        )
    Ssqrt = np.sqrt(S)

    # create the dictionary of node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node in the Euclidean space
    Coord = np.einsum("ij,j->ij", U, Ssqrt)

    src, trg, weights = sparse.find(A)
    nCoord = np.einsum(
        "ij,i->ij", Coord, 1 / np.linalg.norm(Coord, axis=1)
    )  # row-normalized array
    new_weights = 1.0 - np.array(
        np.sum(nCoord[src, :] * nCoord[trg, :], axis=1)
    ).reshape(-1)

    new_weights = np.clip(new_weights, 0.0, 2.0)
    Anew = sparse.csr_matrix((new_weights, (src, trg)), shape=A.shape)
    # I remove the conversion of the results to networkx.Graph since it is inefficient.
    return Anew, Coord


# utils
def to_scipy_matrix(net, return_node_labels=False):
    """
    Converts a networkx graph, a SciPy sparse matrix or a numpy ndarray to a SciPy sparse matrix in CSR format.

    Parameters
    ----------
    net : nx.Graph or scipy.sparse.spmatrix or numpy.ndarray
        The input graph or matrix to convert.
    return_node_labels : bool, optional (default=False)
        Whether to return node labels or not. If True and the input is a networkx graph,
        then a list of node labels will be returned along with the sparse matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        The converted sparse matrix in CSR format.
    list, optional
        If `return_node_labels` is True and the input is a networkx graph,
        a list of node labels corresponding to the rows/columns of the sparse matrix.

    Raises
    ------
    ValueError
        If the input is not of type nx.Graph, scipy.sparse.spmatrix or numpy.ndarray.
    """
    _node_labels = None
    if isinstance(net, nx.Graph):
        _net = nx.to_scipy_sparse_array(
            net, weight="weight", format="csr"
        )  # adjacency matrix as a SciPy sparse matrix

        if return_node_labels:
            _node_labels = list(net.nodes())
    elif sparse.issparse(net):
        _net = net
        if return_node_labels:
            _node_labels = np.arange(net.shape[0], dtype=np.int64)
    elif isinstance(net, np.ndarray):
        _net = sparse.csr_matrix(net)
        if return_node_labels:
            _node_labels = np.arange(net.shape[0], dtype=np.int64)
    else:
        ValueError("Unexpected data type {} for the adjacency matrix".format(type(net)))

    if return_node_labels:
        return _net, _node_labels
    else:
        return _net

import graph_tool.all as gt
g = gt.collection.ns["polblogs"]
A = gt.adjacency(g).T
G = nx.from_scipy_sparse_array(A + A.T)

# %%
%%time
retvals = expISOv2(G, d=12)
# %%
%%time
retvals2 = expISO(G, d=12)
# %%
W = nx.adjacency_matrix(retvals2[0])
dB = retvals[0] - W
np.max(np.abs(dB.data))
