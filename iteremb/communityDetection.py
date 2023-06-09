# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-25 16:46:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-08 12:23:37
#!/usr/bin/env python3

import numpy as np
import networkx as nx
import community as cmL  # Louvain
import infomap  # the applied version: 1.0.0b50
from networkx.algorithms.community.label_propagation import (
    asyn_lpa_communities as alabprop,
)
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

from iteremb import utils
from scipy import sparse
import numpy as np
from scipy import sparse


def commDetWithWeightThresholding(
    G, whichWeightToKeep, minRemovalRate=0.05, maxRemovalRate=0.95
):
    """
    Detects the community structure of a graph with weight thresholding and returns a dictionary
    with node names as keys and group identifiers as values.

    Parameters
    ----------
    G : networkx.Graph or scipy.sparse.csr_matrix
        Network
    whichWeightToKeep : str
        The type of weight thresholding to perform.
        'small': retain link weights BELOW a threshold if inputted weights are distance-like measures.
        'large': retain link weights ABOVE a threshold if inputted weights are proximity-like measures.
    minRemovalRate : float, optional
        The allowed smallest fraction of links to be removed by the weight thresholding (default is 0.05).
    maxRemovalRate : float, optional
        The allowed largest fraction of links to be removed by the weight thresholding (default is 0.95).

    Returns
    -------
    numpy.ndarray
        with node names as keys and group identifiers as values.
    """

    # convert G to scipy matrix
    A = utils.to_scipy_matrix(G)

    # reverse sign of weights if "large" thresholding is chosen
    if whichWeightToKeep == "large":
        A.data = -A.data

    # sort weights in descending order
    weightList = np.sort(A.data)[::-1]

    # calculate minimum and maximum thresholds based on removal rates
    min_threshold = np.quantile(weightList, minRemovalRate)
    max_threshold = np.quantile(weightList, maxRemovalRate)

    # filter weights within threshold range
    weightList = weightList[
        (weightList >= min_threshold) * (weightList <= max_threshold)
    ]

    # find the threshold value where there is a significant drop in weight values
    edge_idx = np.argmax(np.abs(np.diff(weightList)))
    weightThreshold = (weightList[edge_idx] + weightList[edge_idx + 1]) / 2

    # set weights above threshold to zero and eliminate zero entries
    A_pruned = A.copy()
    A_pruned.data[A_pruned.data > weightThreshold] = 0
    A_pruned.eliminate_zeros()

    # take absolute values of weights and find connected components
    A_pruned.data = np.abs(A_pruned.data)
    _, partition = sparse.csgraph.connected_components(
        csgraph=A_pruned, directed=False, return_labels=True
    )

    return partition


def Louvain(G):
    """
    Detects the community structure of a graph with the Louvain method and returns a dictionary with node names as keys and group identifiers as values.

    Parameters:
    -----------
    G : NetworkX graph object
        The examined graph.

    Returns:
    --------
    partition : dict
        A dictionary containing obtained partition with key value pairs where key is
        the name of a node and value is its corresponding community identifier. Communities
        are numbered from 0 with integers.
    """
    A = utils.to_networkx(G)
    partition = cmL.best_partition(A)
    partition = np.array(list(partition.values()))
    return partition


def Infomap(G):
    """
    Parameters
    ----------
    G : networkx.Graph or scipy.sparse.csr_matrix
        The graph to be partitioned into communities.
    """
    A = utils.to_scipy_matrix(G)
    r, c, v = sparse.find((A + A.T))
    im = infomap.Infomap("--two-level --directed --zero-based-numbering")
    for i in range(len(r)):
        im.add_link(r[i], c[i], v[i])
    im.run()
    cids = np.zeros(A.shape[0])
    for node in im.tree:
        if node.is_leaf:
            cids[node.node_id] = node.module_id
    return np.unique(cids, return_inverse=True)[1]


def asynLabProp(G):
    """
    Performs the Asynchronous Label Propagation algorithm to detect communities in a graph.

    Parameters
    ----------
    G : networkx.Graph or scipy.sparse.csr_matrix
        The graph to be partitioned into communities.

    Attributes
    ----------
    comms : iterable of sets of node names
        Communities given as sets of node names.
    A : scipy.sparse.csr_matrix
        Sparse adjacency matrix of the graph `G`.
    N : int
        Number of nodes in the graph `G`.
    partition : numpy.ndarray
        Array containing the community ID for each node in `G`.

    Returns
    -------
    partition : numpy.ndarray
        Array containing the community ID for each node in `G` after running the Asynchronous Label Propagation algorithm.
    """
    # iterable of communities given as sets of node names
    comms = alabprop(G, weight="weight")

    # Convert the networkx graph to a scipy sparse matrix
    A = utils.to_scipy_matrix(G)

    # Get the shape of the matrix A
    N = A.shape[0]

    # Create an array partition with zeros for all nodes
    partition = np.zeros(N)

    # Initialize the group ID to zero
    groupID = 0

    # Assign each node to its corresponding community
    for c in comms:  # c is a set of nodes forming a community
        for nodeName in c:
            partition[nodeName] = groupID
        groupID = groupID + 1

    # Filter out edges between nodes in different communities
    src, trg, _ = sparse.find(A)
    retain = partition[src] == partition[trg]
    src, trg = src[retain], trg[retain]

    # Create the adjacency matrix for the pruned graph
    Aprune = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=A.shape)

    # Compute the connected components of the pruned graph
    _, partition = sparse.csgraph.connected_components(
        Aprune, directed=False, return_labels=True
    )

    # Return the partition
    return partition


###SPATIAL CLUSTERING###


def kMeansClustering(Coord, numOfGroups):
    """
    Perform k-means clustering on the given coordinates.

    Parameters
    ----------
    Coord : array-like of shape (n_samples, n_features)
        The input data to cluster.
    numOfGroups : int
        The desired number of clusters.

    Returns
    -------
    partition : ndarray of shape (n_samples,)
        Labels of each point in the input data.

    Notes
    -----
    This function uses the KMeans algorithm from scikit-learn.
    """
    clustering = KMeans(n_clusters=numOfGroups, algorithm="elkan")
    clustering.fit(Coord)
    partition = clustering.labels_
    return partition


###EVALUATION OF DETECTED COMMUNITIES###


def calculateSimilarityMeasures(realComms, detectedComms) -> tuple:
    """
    Evaluates the community detection and returns the adjusted mutual information (AMI),
    the adjusted Rand index (ARI) and the element-centric similarity (elCentSim) between
    the ground truth and the predicted community structure of a network.

    Parameters:
    - realComms: A numpy array representing the ground truth (real) community structure of a network.
                        Key represents a node's name and value represents the identifier of its community.
    - detectedComms: A numpy array representing the detected community structure of a network.
                            Key represents a node's name and value represents the identifier of its community.

    Returns:
    - tuple: A tuple containing the following similarity measures:
            * AMI (float): Adjusted Mutual Information
            * ARI (float): Adjusted Rand Index
            * elCentSim (float): Element-Centric Similarity
              where 0 <= AMI, ARI, elCentSim <= 1.

    Note:
    AMI=0 corresponds to the value expected due to chance alone, and AMI=1 means perfect agreement
    between the two categorizations. ARI is close to 0 for random labeling and ARI is 1 for identical categorizations.
    Use ARI when the size of the groups in the ground truth clustering are nearly equal, and use AMI when
    the ground truth clustering is unbalanced with respect to the group sizes, i.e. strongly different community sizes occur.
    elCentSim is 1 for identical partitions and decreases towards 0 as the similarity declines.
    """
    AMI = adjusted_mutual_info_score(
        realComms, detectedComms, average_method="max"
    )  # the 'max' method results in the smallest AMI value from all the possible normalizations -> calculate the possible worst AMI
    ARI = adjusted_rand_score(realComms, detectedComms)
    elCentSim = calc_esim(realComms, detectedComms)

    return [AMI, ARI, elCentSim]


def calc_esim(y, ypred):
    """
    Element-centric similarity.
    """
    ylab, y = np.unique(y, return_inverse=True)
    ypredlab, ypred = np.unique(ypred, return_inverse=True)

    Ka, Kb = len(ylab), len(ypredlab)

    K = np.maximum(Ka, Kb)
    N = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, K)
    )
    nA = np.array(UA.sum(axis=0)).reshape(-1)
    nB = np.array(UB.sum(axis=0)).reshape(-1)

    nAB = (UA.T @ UB).toarray()

    # Calc element-centric similarity
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    S = np.sum(np.multiply(Q, (nAB**2))) / N
    return S


def calculateModularity(G: nx.Graph, partition: np.ndarray) -> float:
    """
    Calculates modularity of a graph based on the given partition.

    Parameters:
        G (nx.Graph): NetworkX graph object or scipy.sparse.csr_matrix
        partition (np.ndarray): Node partition array.

    Returns:
        float: Modularity value.
    """
    A = utils.to_scipy_matrix(G)  # Convert graph to adjacency matrix
    deg = np.array(A.sum(axis=0))  # Calculate degree of each node
    m = np.sum(deg) / 2  # Calculate sum of all degrees
    # Construct the membership matrix U, where U[i,k]=1 if i belongs to the k-th community, otherwise U[i,k] =0.
    U = sparse.csr_matrix(
        (np.ones_like(partition), (np.arange(len(partition)), partition)),
        shape=(A.shape[0], int(np.max(partition) + 1)),
    )
    D = np.array(deg.reshape((1, -1)) @ U).reshape(
        -1
    )  # Calculate sum of degrees of nodes in each community
    # Calculate modularity value using formula
    Q = (1.0 / (2 * m)) * (np.trace(U.T @ A @ U) - np.sum(D**2) / (2 * m))
    return Q
