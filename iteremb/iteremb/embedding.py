#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import scipy.stats  # for testing outliers during the dimension selection of the spectral methods with the Modified Thompson Tau test
from scipy import (
    sparse,
)  # needed for TREXPIC, Isomap (singular value decomposition) and Laplacian Eigenmaps (eigendecomposition)
import fastnode2vec  # a fast version of node2vec


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


# A function for creating a weighted version of a graph based on its d-dimensional hyperbolic embedding generated by the TREXPIC method.
# G is the (weighted) NetworkX Graph to be embedded (If link weights are inputted, these must mean distances, i.e. larger value:=less similarity or weaker connection!)
# d is the number of dimensions of the space to which the network will be embedded
# q>0 is the multiplying factor in the exponent in the matrix to be reduced (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal matrix elements towards 0). If None, its default setting will be used.
# K<0 is the curvature of the hyperbolic space
# The function returns the weighted NetworkX Graph G_embWeighted and the dictionary Coord that assigns to each node name a NumPy array of d elements containing the Cartesian coordinates of the given network node in the native representation of the d-dimensional hyperbolic space.
# Example for function call:
#   [G_w,positionDict]=embedding.TREXPIC(G,d)
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


# A function for creating a weighted version of a graph based on its d-dimensional Euclidean embedding generated by the Isomap method.
# G is the (weighted) NetworkX Graph to be embedded (If link weights are inputted, these must mean distances, i.e. larger value:=less similarity or weaker connection!)
# d is the number of dimensions of the space to which the network will be embedded
# The function returns the weighted NetworkX Graph G_embWeighted and the dictionary Coord that assigns to each node name a NumPy array of d elements containing the Cartesian coordinates of the given network node in the d-dimensional Euclidean space.
# Example for function call:
#   [G_w,positionDict]=embedding.ISO(G,d)
def ISO(G, d):
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

    # create the matrix of expected Euclidean distances
    listOfNodes = list(G.nodes())
    D = np.array(
        [[shortestPathLengthsDict[s][t] for t in listOfNodes] for s in listOfNodes]
    )  # D[i][j] is the expected Euclidean distance given by the length of the shortest path from the ith node to the jth node, where the nodes are ordered according to listOfNodes

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
            "\n\nERROR: The number d of embedding dimensions in the function embedding.ISO can not be larger than the number of positive singular values of the inner product matrix, i.e. "
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


# A function for creating a weighted version of a graph based on its d-dimensional Euclidean embedding generated by the Laplacian eigenmaps method.
# G is the (weighted) NetworkX Graph to be embedded (If link weights are inputted, these must mean distances, i.e. larger value:=less similarity or weaker connection!)
# d is the number of dimensions of the space to which the network will be embedded
# scalingFactor is the tunable parameter in the distance-proximity weight conversion formula. If None, its default setting will be used.
# The function returns the weighted NetworkX Graph G_embWeighted and the dictionary Coord that assigns to each node name a NumPy array of d elements containing the Cartesian coordinates of the given network node in the d-dimensional Euclidean space.
# Example for function call:
#   [G_w,positionDict]=embedding.LE(G,d)
def LE(G, d, scalingFactor=None):
    N = len(G)  # the number of nodes in graph G
    if N < d + 1:
        print(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.LE can not be larger than the number of nodes-1, i.e. "
            + str(N - 1)
            + ".\n\n"
        )

    # create the weighted graph that will be eventually used for the embedding (larger weights must mean higher similarity/stronger connection)
    listOfWeights = [G[i][j]["weight"] for (i, j) in G.edges()]
    if (
        len(set(listOfWeights)) == 1
    ):  # all the link weights are equal, i.e. the network is practically unweighted -> simple-minded Laplacian Eigenmaps
        G_forEmbedding = nx.Graph()
        G_forEmbedding.add_nodes_from(
            G.nodes
        )  # keep the node order of the original graph
        for (
            i,
            j,
        ) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            # assign a weight to the i-j edge:
            w = float(
                G[i][j]["weight"]
            )  # the function that performs the eigendecomposition needs matrixes of floats
            # note that from the viewpoint of the eigenvectors, a constant multiplying factor of all the link weights does not matter
            G_forEmbedding.add_edge(i, j, weight=w)
    else:  # Laplacian Eigenmaps with heat kernel
        if scalingFactor == None:  # use the default setting
            scalingFactor = math.pow(
                np.mean(listOfWeights), 2
            )  # default: the square of the average of all the link weights, as in https://www.nature.com/articles/s41467-017-01825-5
            # another option could be e.g.: scalingFactor = math.pow(np.std(listOfWeights),2) #default: the square of the standard deviation of all the link weights, as in https://arxiv.org/abs/2304.06580
            print("The default scaling factor is " + str(scalingFactor) + ".")
        G_forEmbedding = nx.Graph()
        G_forEmbedding.add_nodes_from(
            G.nodes
        )  # keep the node order of the original graph
        for (
            i,
            j,
        ) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            # assign a weight to the i-j edge:
            w = math.exp(-math.pow(G[i][j]["weight"], 2) / scalingFactor)
            G_forEmbedding.add_edge(i, j, weight=w)

    # calculate the degree matrix and the graph Laplacian
    listOfNodes = list(G.nodes())
    adjMatrix = nx.to_scipy_sparse_matrix(
        G_forEmbedding, nodelist=listOfNodes, weight="weight", format="csr"
    )  # adjacency matrix as a SciPy sparse matrix
    # in newer versions of NetworkX: adjMatrix = nx.to_scipy_sparse_array(G_forEmbedding, nodelist=listOfNodes, weight='weight', format="csr")
    numOfRows, numOfCols = adjMatrix.shape
    degreeMatrix = sparse.spdiags(
        adjMatrix.sum(axis=1).reshape(1, N), 0, numOfCols, numOfRows, format="csr"
    )  # degrees in a SciPy sparse matrix
    # when using newer versions of NetworkX: degreeMatrix = sparse.csr_array(sparse.spdiags(adjMatrix.sum(axis=1), 0, numOfCols, numOfRows, format="csr"))
    graphLaplacian = degreeMatrix - adjMatrix

    # perform eigendecomposition -> use the eigenvectors related to the smallest eigenvalues, discarding the first because it is zero
    eigenvalues, eigenvectors = sparse.linalg.eigsh(
        graphLaplacian,
        k=d + 1,
        M=degreeMatrix,
        which="SM",
        ncv=min(N, max(2 * (2 * (d + 1) + 1), 2 * 20)),
    )  # find k eigenvalues and eigenvectors of the real symmetric square matrix graphLaplacian
    # w: array of k eigenvalues
    # v: array representing the k eigenvectors: the column v[:, i] is the eigenvector corresponding to the eigenvalue w[i]
    # When using the parameter which='SM', eigsh returns the smallest (in magnitude) eigenvalues in an increasing order. However, ARPACK is generally better at finding large values than small values. If small eigenvalues are desired, it should be considered to use shift-invert mode (i.e., the setting sigma=0) for better performance. Nevertheless, in some simple tests the setting which='SM' performed better.
    eigenvectors = eigenvectors[
        :, 1 : d + 1
    ]  # omit the eigenvector of 0 eigenvalue and keep only d number of the following eigenvectors
    # (the eigenvalues belonging to the retained eigenvectors are given by w[1:d+1])

    # create the dictionary of node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node
    Coord = {}
    nodeIndex = 0
    for nodeName in listOfNodes:
        Coord[nodeName] = eigenvectors[nodeIndex, :]
        nodeIndex = nodeIndex + 1

    # create the graph with embedding-based link weights
    G_embWeighted = nx.Graph()
    G_embWeighted.add_nodes_from(
        listOfNodes
    )  # keep the node order of the original graph
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


# A function for creating an exponentially weighted version of a graph based on its angular node arrangement in a d-dimensional Euclidean embedding created by node2vec.
# G is the (weighted) NetworkX Graph to be embedded (If link weights are inputted, these must mean proximities, i.e. larger value:=higher similarity or stronger connection!)
# q>0 is the multiplying factor in the exponent in the weight function: link weight=exp(q*(cos(angular distance)-1))
# d, walkLength, numOfWalks, windowLength, batchWalks, pForEmbedding, qForEmbedding: the usual parameters of the node2vec method
# The function returns the weighted NetworkX Graph G_embWeighted and the dictionary Coord that assigns to each node name a NumPy array of d elements containing the Cartesian coordinates of the given network node in the d-dimensional Euclidean space.
# Example for function call:
#   [G_w,positionDict]=embedding.expNode2vec(G,q)
def expNode2vec(
    G,
    q,
    d=64,
    walkLength=10,
    numOfWalks=80,
    windowLength=10,
    batchWalks=10000,
    pForEmbedding=1.0,
    qForEmbedding=1.0,
):
    # create the embedding
    listOfNodes = list(G.nodes())
    adjMatrix = nx.adjacency_matrix(
        G, nodelist=listOfNodes, weight="weight"
    )  # adjacency matrix as a SciPy Compressed Sparse Row (csr) matrix
    # in newer versions of NetworkX: adjMatrix = sparse.csr_matrix(nx.adjacency_matrix(G,nodelist=listOfNodes,weight='weight'))
    model = fastnode2vec.Node2Vec(
        walk_length=walkLength,
        num_walks=numOfWalks,
        window_length=windowLength,
        batch_walks=batchWalks,
        p=pForEmbedding,
        q=qForEmbedding,
    )
    model.fit(adjMatrix)
    center_vec = model.transform(dim=d)  # d number of columns; 1 row=1 node
    # context_vec = model.out_vec #d number of columns; 1 row=1 node
    Coord = (
        {}
    )  # initialize the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space
    nodeID = 0
    for nodeName in listOfNodes:
        Coord[nodeName] = center_vec[nodeID, :]
        nodeID = nodeID + 1

    # create the graph with embedding-based link weights
    G_embWeighted = nx.Graph()
    G_embWeighted.add_nodes_from(
        listOfNodes
    )  # keep the node order of the original graph
    for i, j in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
        # assign a weight to the i-j edge:
        w_notExp = (
            np.inner(Coord[i], Coord[j])
            / (np.linalg.norm(Coord[i]) * np.linalg.norm(Coord[j]))
        ) - 1  # w_notExp=cos(angular distance)-1: nonpositive, proximity-like weights
        if w_notExp < -2:  # a numerical error has occurred
            w_notExp = -2.0
        if w_notExp > 0:  # a numerical error has occurred
            w_notExp = 0.0
        w = math.exp(
            q * w_notExp
        )  # link weight=exponentialized form of a measure composed from cosine similarity
        G_embWeighted.add_edge(i, j, weight=w)

    return [G_embWeighted, Coord]


# A function for creating a weighted version of a graph based on its angular node arrangement in a d-dimensional Euclidean embedding created by node2vec.
# G is the (weighted) NetworkX Graph to be embedded (If link weights are inputted, these must mean proximities, i.e. larger value:=higher similarity or stronger connection!)
# d, walkLength, numOfWalks, windowLength, batchWalks, pForEmbedding, qForEmbedding: the usual parameters of the node2vec method
# The function returns the weighted NetworkX Graph G_embWeighted and the dictionary Coord that assigns to each node name a NumPy array of d elements containing the Cartesian coordinates of the given network node in the d-dimensional Euclidean space.
# Example for function call:
#   [G_w,positionDict]=embedding.node2vec(G)
def node2vec(
    G,
    d=64,
    walkLength=10,
    numOfWalks=80,
    windowLength=10,
    batchWalks=10000,
    pForEmbedding=1.0,
    qForEmbedding=1.0,
):
    # create the embedding
    listOfNodes = list(G.nodes())
    adjMatrix = nx.adjacency_matrix(
        G, nodelist=listOfNodes, weight="weight"
    )  # adjacency matrix as a SciPy Compressed Sparse Row (csr) matrix
    # in newer versions of NetworkX: adjMatrix = sparse.csr_matrix(nx.adjacency_matrix(G,nodelist=listOfNodes,weight='weight'))
    model = fastnode2vec.Node2Vec(
        walk_length=walkLength,
        num_walks=numOfWalks,
        window_length=windowLength,
        batch_walks=batchWalks,
        p=pForEmbedding,
        q=qForEmbedding,
    )
    model.fit(adjMatrix)
    center_vec = model.transform(dim=d)  # d number of columns; 1 row=1 node
    # context_vec = model.out_vec #d number of columns; 1 row=1 node
    Coord = (
        {}
    )  # initialize the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space
    nodeID = 0
    for nodeName in listOfNodes:
        Coord[nodeName] = center_vec[nodeID, :]
        nodeID = nodeID + 1

    # create the graph with embedding-based link weights
    G_embWeighted = nx.Graph()
    G_embWeighted.add_nodes_from(
        listOfNodes
    )  # keep the node order of the original graph
    for i, j in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
        # assign a weight to the i-j edge:
        w = (
            np.inner(Coord[i], Coord[j])
            / (np.linalg.norm(Coord[i]) * np.linalg.norm(Coord[j]))
        ) + 1  # link weight=cos(angular distance)+1: nonnegative, proximity-like weights
        if w < 0:  # a numerical error has occurred
            w = 0.0
        if w > 2:  # a numerical error has occurred
            w = 2.0
        G_embWeighted.add_edge(i, j, weight=w)

    return [G_embWeighted, Coord]
