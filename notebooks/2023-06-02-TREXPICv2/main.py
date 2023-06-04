# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-02 17:30:02
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-03 21:46:50
# %%
import numpy as np
import networkx as nx
import math
from scipy import sparse

#
# Bianka's impl.
#
def TREXPIC(G, d, q=None, K=-1):
    N = len(G)  # the number of nodes in graph G
    if N < d + 1:
        print(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of nodes-1, i.e. "
            + str(N - 1)
            + ".\n\n"
        )

    zeta = math.sqrt(-K)

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

    # create the matrix of expected Lorentz products
    listOfNodes = list(G.nodes())
    L = np.ones(
        (N, N)
    )  # L[i][j] is the expected Lorentz product calculated from the length of the shortest path from the ith node to the jth node, where the nodes are ordered according to listOfNodes
    for rowID in range(1, N):
        for colID in range(rowID):  # iterate over the node pairs
            # since the network to be embedded is undirected, the matrix L must be symmetric
            # the distance of each node from itself is 0 and cosh(0)=1 -> the diagonal elements of L must remain 1
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
            L[rowID, colID] = math.cosh(zeta * expDistance)
            L[colID, rowID] = L[rowID, colID]

    # dimension reduction
    if d == N - 1:
        U, S, VT = np.linalg.svd(
            L
        )  # find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
        # note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else:  # d<N-1: only the first d+1 largest singular values (and the corresponding singular vectors) are retained
        U, S, VT = sparse.linalg.svds(
            L, d + 1, which="LM", solver="arpack"
        )  # the singular values are ordered from the smallest to the largest in S (increasing order)
        # reverse the order of the singular values to obtain a decreasing order, and reverse the order of singular vectors accordingly:
        S = S[::-1]
        U = U[:, ::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues < d + 1:
        print(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of positive singular values of the Lorentz matrix-1, i.e. "
            + str(numOfPositiveSingularValues - 1)
            + ".\n\n"
        )
    Ssqrt = np.sqrt(
        S[1:]
    )  # d number of singular values are used for determining the directions of the position vectors: from the second to the d+1th one

    # create the dictionary of node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node in the native ball
    Coord = (
        {}
    )  # initialize a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the native representation of the d-dimensional hyperbolic space
    nodeIndex = 0
    numOfErrors = 0
    for nodeName in listOfNodes:
        # calculate the position of the given node
        Uarray = U[nodeIndex, :]
        firstCoordOnHyperboloid = math.fabs(
            math.sqrt(S[0]) * Uarray[0]
        )  # to select the upper sheet of the two-sheet hyperboloid, firstCoordOnHyperboloid has to be positive
        # we could also use: Varray = VT[:,nodeIndex] and then firstCoordOnHyperboloid = math.fabs(math.sqrt(S[0])*Varray[0])
        if firstCoordOnHyperboloid < 1:  # a numerical error has occurred
            r_native = 0
            numOfErrors = numOfErrors + 1
        else:
            r_native = (1 / zeta) * math.acosh(firstCoordOnHyperboloid)
        directionArray = np.multiply(
            Uarray[1:], Ssqrt
        )  # the jth element is the jth coordinate of the node named nodeName in the reduced space
        # we could also use: directionArray = np.multiply(Varray[1:],Ssqrt)
        originalNorm = np.linalg.norm(directionArray)
        Coord[nodeName] = (
            r_native * directionArray / originalNorm
        )  # the Cartesian coordinates of the node named nodeName in the native representation of the d-dimensional hyperbolic space
        nodeIndex = nodeIndex + 1
    if numOfErrors > 0:
        print(
            "TREXPIC placed "
            + str(numOfErrors)
            + " nodes at an invalid position on the hyperboloid. The native radial coordinate of these nodes is set to 0. Consider changing the q parameter of the embedding!"
        )

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


#
# My implementation
#
def TREXPICv2(G, d, q=None, K=-1, verbose=False):
    A = to_scipy_matrix(G, return_node_labels=False)
    N = A.shape[0]  # the number of nodes in graph G
    if N < d + 1:
        raise ValueError(
            f"\n\nERROR: The number d of embedding dimensions in the function embedding. TREXPIC can not be larger than the number of nodes-1, i.e. {N-1}\n\n"
        )
    zeta = math.sqrt(-K)

    # create the matrix to be reduced
    shortestPathLengthMatrix = sparse.csgraph.shortest_path(A, directed=False)
    if q == None:  # use the default setting of the multiplying factor
        maxSPL = np.ma.masked_invalid(shortestPathLengthMatrix).max()
        qmin = math.log(1.0 / 0.9999) * maxSPL
        qmax = math.log(10) * maxSPL
        q = math.exp((math.log(qmin) + math.log(qmax)) / 2)
        if verbose:
            print("The default q multiplying factor is " + str(q) + ".")

    # create the matrix of expected Lorentz products
    L = np.exp(-q / shortestPathLengthMatrix)
    L = np.cosh(zeta * L)

    # dimension reduction
    if d == N - 1:
        U, S, VT = np.linalg.svd(
            L
        )  # find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
        # note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else:  # d<N-1: only the first d+1 largest singular values (and the corresponding singular vectors) are retained
        U, S, VT = sparse.linalg.svds(
            L, d + 1, which="LM", solver="arpack"
        )  # the singular values are ordered from the smallest to the largest in S (increasing order)
        # reverse the order of the singular values to obtain a decreasing order, and reverse the order of singular vectors accordingly:
        S = S[::-1]
        U = U[:, ::-1]
        VT = VT[::-1]

    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues < d + 1:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of positive singular values of the Lorentz matrix-1, i.e. "
            + str(numOfPositiveSingularValues - 1)
            + ".\n\n"
        )
    Ssqrt = np.sqrt(
        S[1:]
    )  # d number of singular values are used for determining the directions of the position vectors: from the second to the d+1th one

    numOfErrors = 0
    firstCoordOnHyperboloid = np.abs(
        np.sqrt(S[0]) * U[:, 0]
    )  # to select the upper sheet of the two-sheet hyperboloid, firstCoordOnHyperboloid has to be positive
    is_numerical_error = firstCoordOnHyperboloid < 1
    numOfErrors = np.sum(is_numerical_error)
    r_negative = (1 / zeta) * np.arccosh(np.maximum(firstCoordOnHyperboloid, 1))
    r_negative[is_numerical_error] = 0

    # the jth element is the jth coordinate of the node named nodeName in the reduced space
    directionArray = np.einsum("ij,j->ij", U[:, 1:], Ssqrt)
    originalNorm = np.array(np.linalg.norm(directionArray, axis=1)).reshape(-1)
    Coord = np.einsum("ij,i->ij", directionArray, r_negative / originalNorm)

    if numOfErrors > 0:
        raise ValueError(
            "TREXPIC placed "
            + str(numOfErrors)
            + " nodes at an invalid position on the hyperboloid. The native radial coordinate of these nodes is set to 0. Consider changing the q parameter of the embedding!"
        )

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



#
# Test script
#
import graph_tool.all as gt
g = gt.collection.ns["polblogs"]
A = gt.adjacency(g).T
G = nx.from_scipy_sparse_array(A + A.T)

# %%
%%time
retvals = TREXPIC(G, d=12)
# %%
%%time
retvals = TREXPICv2(G, d=12)