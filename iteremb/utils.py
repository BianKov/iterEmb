# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-02 16:47:58
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-08 12:22:58
import os
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import scipy.stats  # for testing outliers during the dimension selection of the spectral methods with the Modified Thompson Tau test
from scipy import (
    sparse,
)  # needed for TREXPIC, Isomap (singular value decomposition) and Laplacian Eigenmaps (eigendecomposition)


# A function for loading the undirected edge list of a network to be embedded. Self-loops are removed, multi-edges are converted to single edges, and only the largest connected component is returned as a NetworkX Graph.
# path is a string with the path of the text file containing the edge list to be loaded
# In the edge list each line has to correspond to one connected node pair. The first two columns are assumed to contain the node identifiers, the third column (if exists) is assumed to contain the corresponding link weight, and the other columns will be disregarded. If no link weights are given, each edge will have a weight of 1.0.
# skipRows is the number of lines to be skipped at the beginning of the text file containing the edge list; the default is 0
# delimiter is the string used to separate the columns in the text file to be loaded; the default is "\t"
# Example for function call:
#   G=embedding.loadGraph(os.getcwd()+"/"+directoryName+"/edgeList.txt",1,"\t")
def loadGraph(path, skipRows=0, delimiter="\t"):
    edgeList = (
        []
    )  # initialize the list of the (source node identifier,target node identifier,link weight) edge 3-tuples
    fileHandler = open(path, "r")
    for l in range(skipRows):
        line = fileHandler.readline()
    while True:
        line = fileHandler.readline()  # get the next line from file
        if not line:  # line is empty (end of the file)
            break
        listOfWords = line.split(delimiter)
        linkWeight = None  # initialization
        sourceNodeID = listOfWords[
            0
        ]  # string from the first column as the identifier of the source node
        if (
            listOfWords[1][-1] == "\n"
        ):  # the second column is the last in the currently loaded line
            targetNodeID = listOfWords[1][
                :-1
            ]  # string from the second column without "\n" as the identifier of the target node
        else:  # there are more than two columns in the currently loaded line
            targetNodeID = listOfWords[
                1
            ]  # string from the second column as the identifier of the target node
            if (
                listOfWords[2][-1] == "\n"
            ):  # the third column is the last in the currently loaded line
                linkWeight = float(
                    listOfWords[2][:-1]
                )  # convert the string from the third column without "\n" to a float
            else:  # there are more than three columns in the currently loaded line
                linkWeight = float(listOfWords[2])
        if sourceNodeID != targetNodeID:  # the self-loops are disregarded
            if (
                linkWeight == None
            ):  # set all the link weights to 1.0 (the embedding functions will assume that the links have a weight attribute)
                edgeList.append((sourceNodeID, targetNodeID, 1.0))
            else:  # a link weight value has also been loaded
                edgeList.append((sourceNodeID, targetNodeID, linkWeight))
    fileHandler.close()

    G_total = nx.Graph()
    G_total.add_weighted_edges_from(
        edgeList, weight="weight"
    )  # multi-edges are automatically converted to single edges -> when loading a weighted edge list, please include all the links with only the one required weight, otherwise only the lastly appearing weight value will be stored
    # extract the largest connected component:
    G = max(
        [G_total.subgraph(comp).copy() for comp in nx.connected_components(G_total)],
        key=len,
    )  # .copy(): create a subgraph with its own copy of the edge/node attributes -> changes to attributes in the subgraph are NOT reflected in the original graph; without copy the subgraph is a frozen graph for which edges can not be added or removed

    return G


# At https://www.nature.com/articles/s41467-017-01825-5 different pre-weighting strategies are described for undirected networks that facilitated the estimation of the spatial arrangement of the nodes in dimension reduction techniques that are built on the shortest path lengths. This function performs these link weighting procedures. Larger weight corresponds to less similarity - can be used for facilitating TREXPIC, Isomap or Laplacian Eigenmaps.
# G is an unweighted NetworkX Graph to be pre-weighted. Note that the link weights in G do not change from 1.0, the function returns instead a copy of G that is weighted.
# weightingType is a string corresponding to the name of the pre-weighting rule to be used. Can be set to 'RA1', 'RA2', 'RA3', 'RA4' or 'EBC'.
# Example for function call:
#   G_preweighted = embedding.preWeighting(G,'RA1')
def preWeighting(G, weightingType):
    G_weighted = nx.Graph()
    G_weighted.add_nodes_from(G.nodes)  # keep the node order of the original graph
    if weightingType == "RA1":
        for (
            i,
            j,
        ) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(
                nx.common_neighbors(G, i, j)
            )  # set of the common neighbors' indices
            # set=unordered collection with no duplicate elements,
            # set operations (union, intersect, complement) can be executed (see RAtype==2)
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) + G.degree(j) + G.degree(i) * G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i, j, weight=w)
    elif weightingType == "RA2":
        for (
            i,
            j,
        ) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(
                nx.common_neighbors(G, i, j)
            )  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j

            # ei=the external degree of the node i with respect to node j,
            # i.e. the number of links from i to neither j nor the common neighbors with j,
            # i.e. the number of i's neighbors without node j and the common neighbors with j
            neighborSet_i = {
                n for n in G[i]
            }  # set with the indices of the neighbors of node i
            # G[i]=adjacency dictionary of node i -> iterating over its keys(=neighboring node indices)
            ei = len(neighborSet_i - {j} - CNset)

            # ej=the external degree of the node j with respect to node i,
            # i.e. the number of links from j to neither i nor the common neighbors with i,
            # i.e. the number of j's neighbors without node i and the common neighbors with i
            neighborSet_j = {
                n for n in G[j]
            }  # set with the indices of the neighbors of node j
            # G[j]=adjacency dictionary of node j -> iterating over its keys(=neighboring node indices)
            ej = len(neighborSet_j - {i} - CNset)

            # assign a weight to the i-j edge:
            w = (1 + ei + ej + ei * ej) / (1 + CN)
            G_weighted.add_edge(i, j, weight=w)
    elif weightingType == "RA3":
        for (
            i,
            j,
        ) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(
                nx.common_neighbors(G, i, j)
            )  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) + G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i, j, weight=w)
    elif weightingType == "RA4":
        for (
            i,
            j,
        ) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(
                nx.common_neighbors(G, i, j)
            )  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) * G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i, j, weight=w)
    elif weightingType == "EBC":  # use the edge betweenness centrality
        # create a dictionary, which contains all the shortest paths between all node pairs
        # shortestPathsDict[(source,target)] is the list of shortest paths from node with ID source to node with ID target
        # a path is a list of nodes following each other in the path
        # the graph to be embedded should be connected (all nodes can be reached from any node)
        shortestPathsDict = {}
        nodeList = list(G.nodes)
        N = len(nodeList)
        for u in range(N - 1):  # u=0,1,...,N-2
            for v in range(u + 1, N):  # v=u+1,...,N-1
                # these loops are sufficient only if graph G is undirected (the same number of paths lead from the uth node to the vth node and from the vth node to the uth node) and does not contain any self-loops
                node_u = nodeList[u]
                node_v = nodeList[v]
                shortestPathsDict[(node_u, node_v)] = [
                    p
                    for p in nx.all_shortest_paths(
                        G, source=node_u, target=node_v, weight=None
                    )
                ]  # weight=None: every edge has weight/distance/cost 1 (the possible current weights are disregarded)

        # weight all the edges
        for i, j in G.edges():
            w = 0  # initialize the weight of the i-j edge
            for u in range(N - 1):
                for v in range(u + 1, N):
                    shortestPathsBetween_uv = shortestPathsDict[
                        (nodeList[u], nodeList[v])
                    ]  # list of shortest paths between the uth node and the vth node
                    sigma = len(
                        shortestPathsBetween_uv
                    )  # the total number of shortest paths between the uth node and the vth node
                    # count those paths between node u and node v which contains the i-j edge
                    sigma_ij = 0
                    for (
                        q
                    ) in (
                        shortestPathsBetween_uv
                    ):  # q=list of nodes following each other in a path between the uth node and the vth node
                        if (
                            i in q and j in q
                        ):  # since q is a shortest path, therefore in this case abs(q.index(i)-q.index(j))==1 is already granted
                            sigma_ij = sigma_ij + 1
                    w = w + (sigma_ij / sigma)
            G_weighted.add_edge(i, j, weight=w)  # assign a weight to the i-j edge
    else:
        print("False parameter: weightingType\n")
    return G_weighted


# This function chooses the number of embedding dimensions for the spectral methods based on the spectral gaps of the graph's normalized Laplacian.
# G is an unweighted NetworkX Graph to be embedded
# path is a string with the path of the figure to be saved
# 1>alpha>0 is the significance level: at smaller values of alpha it is harder to become an outlier (less gaps will be labeled as outliers)
# weightingType is a string corresponding to the name of the pre-weighting rule to be used. Can be set to 'RA1', 'RA2', 'RA3', 'RA4' or 'EBC'.
# Example for function call:
#   embeddingDimForTREXPIC = embedding.choosingDim(G,os.getcwd()+'/'+directoryName+'/automaticDimensionSelectionForTREXPIC_graph0.png')
def choosingDim(G, path, alpha=math.pow(10, -6), weightingType="RA1"):
    # pre-weighting
    G_weighted = preWeighting(
        G, weightingType
    )  # larger weight corresponds to less similarity
    # convert the weights: larger weight must correspond to higher similarity
    scalingFactor = 0
    for (
        i,
        j,
    ) in (
        G_weighted.edges()
    ):  # (i,j)=a tuple, which denotes the edge between node u and v
        scalingFactor = scalingFactor + G_weighted[i][j]["weight"]
    scalingFactor = math.pow(scalingFactor / len(G_weighted.edges()), 2)
    for (
        i,
        j,
    ) in (
        G_weighted.edges()
    ):  # (i,j)=a tuple, which denotes the edge between node u and v
        G_weighted[i][j]["weight"] = math.exp(
            -math.pow(G_weighted[i][j]["weight"], 2) / scalingFactor
        )

    # examine the eigenvalues
    N = len(G_weighted)  # number of nodes
    dMax = int(N / 2)  # theoretical maximum: N-1
    # Eigenvalues of the normalized graph Laplacian (the first one is 0; the smaller the more important):
    eigenValsOfL = nx.normalized_laplacian_spectrum(
        G_weighted, weight="weight"
    )  # increasing order of N number of eigenvalues
    gaps = np.zeros(
        dMax - 1
    )  # only N-1 number of eigenvalues are relevant (the one that is always 0 is not)
    for i in range(2, dMax + 1):
        gaps[i - 2] = eigenValsOfL[i] - eigenValsOfL[i - 1]
    maxGapSize = np.amax(gaps)
    meanGapSize = np.mean(gaps)
    stdOfGapSize = np.std(gaps, ddof=1)  # sample standard deviation
    numOfGaps = len(gaps)
    critical_t = scipy.stats.t.ppf(q=1 - alpha / 2, df=numOfGaps - 2)
    rejectionRegion = (critical_t * (numOfGaps - 1)) / (
        math.sqrt(numOfGaps) * math.sqrt(numOfGaps - 2 + math.pow(critical_t, 2))
    )
    delta = (maxGapSize - meanGapSize) / stdOfGapSize
    if delta > rejectionRegion:  # the value of the largest gap is an outlier
        bestDim = np.argmax(gaps) + 1
    else:
        bestDim = dMax
        print(
            "WARNING: Maybe there is no significant community structure in the given network!"
        )
    if bestDim == 1:
        d = 2
    else:
        d = bestDim
    print("The chosen number of dimensions is " + str(d))

    # plot the spectrum
    dMaxPlot = dMax - 1
    fig = plt.figure(figsize=(10, 16))
    # plot the eigenvalues
    ax = plt.subplot(2, 1, 1)
    plt.plot(
        range(1, dMaxPlot + 2), eigenValsOfL[: dMaxPlot + 1], "g-", marker="."
    )  # first eigenvalue=0 -> would need the eigenvalues from the second to the (d+1)th one for an embedding...
    plt.xlabel("Index of eigenvalue")
    plt.ylabel("Eigenvalue of the normalized Laplacian")
    plt.xlim((0.0, dMaxPlot + 2.0))
    plt.title("Best number of dimensions: " + str(bestDim))
    # plot the gaps between the adjacent eigenvalues
    ax = plt.subplot(2, 1, 2)
    plt.plot(
        range(1, dMaxPlot + 1), gaps[:dMaxPlot], "b-", marker="."
    )  # first eigenvalue=0 -> would need the eigenvalues from the second to the (d+1)th one for an embedding...
    plt.xlabel("Number of dimensions")
    plt.ylabel("Gaps between the adjacent Eigenvalues of the normalized Laplacian")
    plt.xlim((0.0, dMaxPlot + 2.0))
    plt.subplots_adjust(
        hspace=0.15
    )  # set the distance between the figures (wspace=vizszintes)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return int(d)


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
        _net = sparse.csr_matrix(
            nx.to_scipy_sparse_array(net, weight="weight", format="csr")
        )  # adjacency matrix as a SciPy sparse matrix
        if return_node_labels:
            _node_labels = list(net.nodes())
    elif sparse.issparse(net):
        _net = sparse.csr_matrix(net)
        if return_node_labels:
            _node_labels = np.arange(net.shape[0], dtype=np.int64)
    else:
        ValueError("Unexpected data type {} for the adjacency matrix".format(type(net)))

    if return_node_labels:
        return _net, _node_labels
    else:
        return _net


def to_networkx(net):
    if isinstance(net, nx.Graph):
        return net
    elif sparse.issparse(net):
        _net = sparse.csr_matrix(net)
    else:
        ValueError("Unexpected data type {} for the adjacency matrix".format(type(net)))
    return nx.from_scipy_sparse_array(_net)
