#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import networkx as nx
from collections import Counter
import community as cmL  # Louvain
import infomap  # the applied version: 1.0.0b50
from networkx.algorithms.community.label_propagation import (
    asyn_lpa_communities as alabprop,
)
from sklearn.cluster import KMeans
from networkx.algorithms.community.quality import modularity
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

# for element-centric clustering comparison ( https://www.nature.com/articles/s41598-019-44892-y ):
from clusim.clustering import Clustering, print_clustering
import clusim.sim as sim


###COMMUNITY DETECTION###


# This function detects the community structure of a graph with weight thresholding and returns a dictionary with node names as keys and group identifiers as values.
# G is the examined NetworkX graph
# whichWeightToKeep is a string::
# whichWeightToKeep='small' means that the link weights BELOW a threshold will be retained. Use this option if the inputted weights are distance-like measures, i.e. when larger weight=less similarity or weaker connection.
# whichWeightToKeep='large' means that the link weights ABOVE a threshold will be retained. Use this option if the inputted weights are proximity-like measures, i.e. when larger weight=higher similarity or stronger connection.
# minRemovalRate is the allowed smallest fraction of links to be removed by the weight thresholding
# maxRemovalRate is the allowed largest fraction of links to be removed by the weight thresholding
def commDetWithWeightThresholding(
    G, whichWeightToKeep, minRemovalRate=0.05, maxRemovalRate=0.95
):
    weightList = [G[s][t]["weight"] for (s, t) in G.edges()]
    numOfWeights = len(weightList)
    minNumOfLinksToBeRemoved = int(minRemovalRate * numOfWeights)
    maxNumOfLinksToBeRemoved = int(maxRemovalRate * numOfWeights)

    largestDifferenceOfNeighboringWeights = 0  # initialization
    weightThreshold = 0  # initialization
    if whichWeightToKeep == "small":
        weightList.sort(reverse=False)  # increasing order of the weights
        for wID in range(
            numOfWeights - maxNumOfLinksToBeRemoved,
            numOfWeights - minNumOfLinksToBeRemoved - 1,
        ):
            wDiff = weightList[wID + 1] - weightList[wID]
            if wDiff > largestDifferenceOfNeighboringWeights:
                largestDifferenceOfNeighboringWeights = wDiff
                weightThreshold = (weightList[wID + 1] + weightList[wID]) / 2
        G_pruned = nx.Graph()
        G_pruned.add_nodes_from(list(G.nodes()))
        for s, t in G.edges():
            if G[s][t]["weight"] < weightThreshold:
                G_pruned.add_edge(s, t)
    elif whichWeightToKeep == "large":
        weightList.sort(reverse=True)  # decreasing order of the weights
        for wID in range(
            numOfWeights - maxNumOfLinksToBeRemoved,
            numOfWeights - minNumOfLinksToBeRemoved - 1,
        ):
            wDiff = weightList[wID] - weightList[wID + 1]
            if wDiff > largestDifferenceOfNeighboringWeights:
                largestDifferenceOfNeighboringWeights = wDiff
                weightThreshold = (weightList[wID + 1] + weightList[wID]) / 2
        G_pruned = nx.Graph()
        G_pruned.add_nodes_from(list(G.nodes()))
        for s, t in G.edges():
            if G[s][t]["weight"] > weightThreshold:
                G_pruned.add_edge(s, t)
    else:
        print("False parameter: whichWeightToKeep\n")

    componentList = [
        G_pruned.subgraph(c).copy() for c in nx.connected_components(G_pruned)
    ]
    partition = (
        {}
    )  # initialize the dictionary with the obtained partition (key=a node's name,value=its community); the communities are numbered from 0 with integers
    groupID = 0
    for comp in componentList:
        for nodeName in comp.nodes():
            partition[nodeName] = groupID
        groupID = groupID + 1

    return partition


# This function detects the community structure of a graph with the Louvain method and returns a dictionary with node names as keys and group identifiers as values.
# G is the examined NetworkX graph
def Louvain(G):
    # dend = cmL.generate_dendrogram(G)
    # smallestCommunities = cmL.partition_at_level(dend,0)
    partition = cmL.best_partition(
        G
    )  # dictionary with the obtained partition (key=a node's name,value=its community); the communities are numbered from 0 with integers
    return partition


# This function detects the community structure of a graph with the Infomap method and returns a dictionary with node names as keys and group identifiers as values.
# G is the examined NetworkX graph
def Infomap(G):
    nodeList = list(G.nodes())
    nodeNameNodeIDdict = {
        nodeList[i]: i for i in range(len(nodeList))
    }  # key=node name, value=node ID as an unsigned int
    infomInst = infomap.Infomap(
        "--zero-based-numbering --undirected"
    )  # create an Infomap instance for multi-level clustering
    network = infomInst.network()  # default (empty) network
    for i, j in G.edges():
        network.addLink(
            nodeNameNodeIDdict[i],
            nodeNameNodeIDdict[j],
            G.get_edge_data(i, j)["weight"],
        )  # expected argument types: unsigned int,unsigned int,double
    infomInst.run()  # run the Infomap search algorithm to find optimal modules
    partition = (
        {}
    )  # dictionary with the obtained partition (key=a node's name, value=its community); the communities are numbered from 0 with integers
    for node in infomInst.iterTree():
        if node.isLeaf():
            partition[nodeList[node.physicalId]] = node.moduleIndex()
    return partition


# This function detects the community structure of a graph with the asynchronous label propagation method and returns a dictionary with node names as keys and group identifiers as values.
# G is the examined NetworkX graph
def asynLabProp(G):
    comms = alabprop(
        G, weight="weight"
    )  # iterable of communities given as sets of node names
    partition = (
        {}
    )  # dictionary with the obtained partition (key=a node's name,value=its community); the communities are numbered from 0 with integers
    groupID = 0
    for c in comms:  # c is a set of nodes forming a community
        for nodeName in c:
            partition[nodeName] = groupID
        groupID = groupID + 1

    # separate those node groups of the same label that are connected only through nodes of other communities
    numOfOrigGroups = len(set(partition.values()))
    lastGroupID = numOfOrigGroups - 1
    for groupID in range(numOfOrigGroups):
        nodeNamesInGroup = [i for i, x in partition.items() if x == groupID]
        S = G.subgraph(
            nodeNamesInGroup
        )  # the subgraph spanned by the nodes of the group having the identifier groupID
        if (
            nx.is_connected(S) == False
        ):  # the subgraph consists of multiple disconnected parts -> the corresponding group has to be splitted
            compList = [
                comp for comp in nx.connected_components(S)
            ]  # compList[i] is the set of node names in the ith component of the group with the identifier groupID
            numOfNewGroups = (
                len(compList) - 1
            )  # the number of new groups that have to be created (one of the subgroups will keep the original groupID)
            for q in range(numOfNewGroups):
                for nodeName in compList[q]:
                    partition[nodeName] = lastGroupID + 1
                lastGroupID = lastGroupID + 1
            # the identifier of the LAST subgroup is not changed, it remains groupID

    return partition


###SPATIAL CLUSTERING###


# This function detects the community structure of an embedded graph with k-means clustering and returns a dictionary with node names as keys and group identifiers as values.
# CoordDict is a dictionary that assigns to each node name a NumPy array of d elements containing the Cartesian coordinates of the given network node in the d-dimensional embedding space.
# numOfGroups (integer) is the number of clusters to be detected
def kMeansClustering(CoordDict, numOfGroups):
    listOfNodes = list(CoordDict.keys())
    numOfNodes = len(listOfNodes)
    numOfDims = len(CoordDict[listOfNodes[0]])
    CoordMatrix = np.zeros((numOfNodes, numOfDims))
    nodeID = 0
    for nodeName in listOfNodes:
        CoordMatrix[nodeID, :] = CoordDict[nodeName]
        nodeID = nodeID + 1
    clustering = KMeans(n_clusters=numOfGroups, algorithm="elkan")
    clustering.fit(CoordMatrix)
    commStruct_kMeans = (
        clustering.labels_
    )  # an array of integers where the ith element describes the label of the community to which the ith node in listOfNodes belongs
    partition = (
        {}
    )  # initialize the dictionary with the obtained partition (key=a node's name,value=its community); the communities are numbered from 0 with integers
    nodeID = 0
    for nodeName in listOfNodes:
        partition[nodeName] = commStruct_kMeans[nodeID]
        nodeID = nodeID + 1
    return partition


###EVALUATION OF DETECTED COMMUNITIES###


# This function evaluates the community detection and returns the adjusted mutual information (AMI), the adjusted Rand index (ARI) and the element-centric similarity (elCentSim) between the ground truth and the predicted community structure of a network.
# AMI=0 corresponds to the value expected due to chance alone, and AMI=1 means perfect agreement between the two categorization
# ARI is close to 0 for random labeling and ARI is 1 for identical categorizations
# Use ARI when the size of the groups in the ground truth clustering are nearly equal, and use AMI when the ground truth clustering is unbalanced with respect to the group sizes, i.e. strongly different community sizes occur. ( https://jmlr.csail.mit.edu/papers/volume17/15-627/15-627 )
# elCentSim is 1 for identical partitions and decreases towards 0 as the similarity declines
# realComms is the real (ground truth, planted) community structure of a network
# detectedComms is the community structure determined by a community detection algorithm
# each community structure is given by a dictionary, where key=a node's name and value=the identifier of its community
def calculateSimilarityMeasures(realComms, detectedComms):
    # do not consider those ground truth communities that consist of a single node
    # consider only those nodes to be relevant that belong to a group of more than 1 nodes, making it possible for the community detection methods to correctly predict their community label
    commSizeDict = Counter(realComms.values())  # key=groupID, value=number of members
    listOfRelevantNodes = [
        nodeName
        for nodeName in realComms.keys()
        if commSizeDict[realComms[nodeName]] > 1
    ]

    # calculate AMI and ARI
    realCommList_relevant = [
        realComms[nodeName] for nodeName in listOfRelevantNodes
    ]  # list of the real community labels
    detectedCommList_relevant = [
        detectedComms[nodeName] for nodeName in listOfRelevantNodes
    ]  # list of the detected community labels
    AMI = adjusted_mutual_info_score(
        realCommList_relevant, detectedCommList_relevant, average_method="max"
    )  # the 'max' method results in the smallest AMI value from all the possible normalizations -> calculate the possible worst AMI
    ARI = adjusted_rand_score(realCommList_relevant, detectedCommList_relevant)

    # calculate the element-centric similarity
    realCommDict_relevant = {
        nodeName: [realComms[nodeName]] for nodeName in listOfRelevantNodes
    }  # Here non-overlapping communities are assumed!
    detectedCommDict_relevant = {
        nodeName: [detectedComms[nodeName]] for nodeName in listOfRelevantNodes
    }  # Here non-overlapping communities are assumed!
    clustering_real = Clustering(elm2clu_dict=realCommDict_relevant)
    clustering_detected = Clustering(elm2clu_dict=detectedCommDict_relevant)
    elCentSim = sim.element_sim(clustering_real, clustering_detected, alpha=0.9)

    return [AMI, ARI, elCentSim]


# This function returns the modularity of the given division of the network into modules.
# larger modularity means stronger community structure; it is positive if the number of edges within groups exceeds the number expected on the basis of chance.
# G is the examined NetworkX graph
# groupDict is the community structure, given by a dictionary, where key=a node's name and value=the identifier of its community
def calculateModularity(G, groupDict):
    Q = modularity(
        G,
        [
            {node for node in groupDict.keys() if groupDict[node] == group}
            for group in set(groupDict.values())
        ],
        weight=None,
    )
    # weight=None: each edge has weight 1 - we consider only the graph structure, not the geometric information (that can depend on the number of dimensions)
    return Q
