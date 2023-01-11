#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import networkx as nx
import os
import copy
import embedding as em
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import networkx.algorithms.community as nx_comm #for modularity calculation


#This function evaluates the community detection and returns the adjusted mutual information (AMI) and the adjusted Rand index (ARI) between the ground truth and the predicted community structure of a network.
    #AMI=0 means no mutual information and AMI=1 means perfect correlation between the two categorization
    #ARI is close to 0 for random labeling and ARI is 1 for identical categorizations
    #Use ARI when the size of the groups in the ground truth clustering are nearly equal, and use AMI when the ground truth clustering is unbalanced with respect to the group sizes, i.e. strongly different community sizes occur. ( https://jmlr.csail.mit.edu/papers/volume17/15-627/15-627 )
#realComms is the real community structure of a network
#detectedComms is the community structure determined by a community detection algorithm
    #each community structure is given by a list where the ith element describes the label of the community to which the graph's ith node belongs
def AMI_ARI_NMI_calculation(realComms,detectedComms):
    realComms_relevant = []
    detectedComms_relevant = []
    for nodeIndex in range(len(realComms)):
        if realComms.count(realComms[nodeIndex])!=1: #the node with nodeIndex belongs to a group of more than 1 nodes, and thus it is possible for the community detection method to correctly predict its community label
            realComms_relevant.append(realComms[nodeIndex])
            detectedComms_relevant.append(detectedComms[nodeIndex])
    AMI = adjusted_mutual_info_score(realComms_relevant,detectedComms_relevant,average_method='max') #the 'max' method results in the smallest AMI value from all the possible normalizations -> calculate the possible worst AMI
    ARI = adjusted_rand_score(realComms_relevant,detectedComms_relevant)
    NMI = normalized_mutual_info_score(realComms_relevant,detectedComms_relevant)
    return [AMI,ARI,NMI]


def calculateModularity(G,groupDict):
    Q = nx_comm.modularity(G, [{node for node in groupDict if groupDict[node]==group} for group in set(groupDict.values())], weight=None)
        #weight=None: each edge has weight 1 - we consider only the graph structure, not the geometric information (that can depend on the number of dimensions)
    return Q
    









graphID = 1 #https://link.aps.org/accepted/10.1103/PhysRevE.103.022316 : 20 networks at each setting

muList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
dirNameDict = {0.1:"0.1", 0.2:"0.2", 0.3:"0.3", 0.4:"0.4", 0.5:"0.5", 0.6:"0.6", 0.7:"0.7", 0.8:"0.8", 0.9:"0.9"}

minRemovalRate=0.05
maxRemovalRate=0.95
wDiffOfStop = 0.01
maxNumOfEmbeddings = 25

for mu in muList:
    print('mu='+str(mu))
    
    #load data
    directoryName = "LFR_"+dirNameDict[mu]+"/graph"+str(graphID)
    edgeListFileName = "network.dat"
    commStructFileName = "community.dat"
    #load the graph
    G_original=em.loadGraph(os.getcwd()+"/"+directoryName+"/"+edgeListFileName,0,"\t") #the largest connected component, without self-loops, parallel edges and link weights
    print('number of nodes='+str(len(G_original))+', number of edges='+str(len(G_original.edges()))+', average degree='+str(2*len(G_original.edges())/len(G_original)))
    #load the real community structure
    groupDict_real = {} #key=nodeID, value=value (name (string) of the group)
    ReportFileHandler=open(os.getcwd()+"/"+directoryName+"/"+"/"+commStructFileName,'r')
    listOfLines=ReportFileHandler.readlines()
    ReportFileHandler.close()
    for line in listOfLines:
        listOfWords=line.split('\t')
        nodeID=listOfWords[0]
        commID=listOfWords[1]
        groupDict_real[nodeID] = commID
    numOfGroups=len(set(groupDict_real.values()))
    print('number of groups='+str(numOfGroups))
    commStruct_real = [] #a list of integers where the ith element describes the label of the community to which the graph's ith node belongs
    for nodeName in G_original: #iterate over the nodes
        commStruct_real.append(groupDict_real[nodeName])

    #determine the optimal number of dimensions
    d = em.choosingDim_TREXPIC(G_original,os.getcwd()+'/'+directoryName+'/automaticDimensionSelectionForTREXPIC.png')

    #embedding iteration
    #start from the original, unweighted graph
    G_previous = nx.Graph()
    G_previous.add_nodes_from(list(G_original.nodes()))
    for (s,t) in G_original.edges():
        G_previous.add_edge(s,t,weight=1)
    #create the first embedding
    print('TREXPIC - 1')
    G_current=em.TREXPIC(G_previous,d)
    numOfDoneEmbeddings = 1
    #check for convergence: calculate the difference between the current and the previous weights
    avgWeight_previous = np.mean([G_previous[s][t]['weight'] for (s,t) in G_previous.edges()])
    avgWeight_current = np.mean([G_current[s][t]['weight'] for (s,t) in G_current.edges()])
    relChangeInAvgWeight = math.fabs(avgWeight_current-avgWeight_previous)/avgWeight_current
    print("The relative difference of the average link weight at iteration "+str(numOfDoneEmbeddings)+" is "+str(relChangeInAvgWeight))
    #create further embeddings
    while relChangeInAvgWeight>wDiffOfStop and numOfDoneEmbeddings<maxNumOfEmbeddings:
        G_previous = copy.deepcopy(G_current) #eltároljuk a már elvégzett beágyazás kiindulópontját
        print('TREXPIC - '+str(numOfDoneEmbeddings+1))
        G_current=em.TREXPIC(G_previous,d)
        numOfDoneEmbeddings = numOfDoneEmbeddings+1
        #check for convergence: calculate the difference between the current and the previous weights
        avgWeight_previous = np.mean([G_previous[s][t]['weight'] for (s,t) in G_previous.edges()])
        avgWeight_current = np.mean([G_current[s][t]['weight'] for (s,t) in G_current.edges()])
        relChangeInAvgWeight = math.fabs(avgWeight_current-avgWeight_previous)/avgWeight_current
        print("The relative difference of the average link weight at iteration "+str(numOfDoneEmbeddings)+" is "+str(relChangeInAvgWeight))

    #search for the communities after stopping the embedding iteration
    print('The final number of embeddings: '+str(numOfDoneEmbeddings))
    #evaluate the final result using a weight threshold
    weightList = [G_current[s][t]['weight'] for (s,t) in G_current.edges()]
    weightList.sort(reverse=False) #increasing order of the weights=hyperbolic distances
    largestDifferenceOfNeighboringWeights = 0
    weightThreshold = 0
    numOfWeights = len(weightList)
    minNumOfLinksToBeRemoved = int(minRemovalRate*numOfWeights)
    maxNumOfLinksToBeRemoved = int(maxRemovalRate*numOfWeights)
    for wID in range(numOfWeights-maxNumOfLinksToBeRemoved,numOfWeights-minNumOfLinksToBeRemoved-1):
        wDiff = weightList[wID+1]-weightList[wID]
        if wDiff>largestDifferenceOfNeighboringWeights:
            largestDifferenceOfNeighboringWeights = wDiff
            weightThreshold = (weightList[wID+1]+weightList[wID])/2
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(list(G_current.nodes()))
    for (s,t) in G_current.edges():
        if G_current[s][t]['weight']<weightThreshold:
            G_pruned.add_edge(s,t)
    componentList = [G_pruned.subgraph(c).copy() for c in nx.connected_components(G_pruned)]
    groupDict_detected = {} #key=nodeID, value=groupID
    groupID = 0
    for comp in componentList:
        for nodeName in comp.nodes():
            groupDict_detected[nodeName] = groupID
        groupID = groupID+1

    #evaluate the detected community structure
    #calculate the modularity
    Q_detected = calculateModularity(G_original,groupDict_detected)
    #examine the community structure
    numOfComms_detected = len(set(groupDict_detected.values()))
    largestCommSize = 0
    for group in set(groupDict_detected.values()):
        commSize = len({node for node in groupDict_detected if groupDict_detected[node]==group})
        if largestCommSize<commSize:
            largestCommSize = commSize
    largestRelCommSize_detected = largestCommSize/len(G_original)
    #calculate the AMI, ARI and NMI values
    commStruct_detected = [] #a list of integers where the ith element describes the label of the community to which the graph's ith node belongs
    for nodeName in G_original: #iterate over the nodes
        commStruct_detected.append(groupDict_detected[nodeName])
    [AMI_detected,ARI_detected,NMI_detected] = AMI_ARI_NMI_calculation(commStruct_real,commStruct_detected)

    #save all the results
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/TREXPICresults_readable.txt", 'w')
    ReportFileHandler.write('Q_detected = '+str(Q_detected)+'\n')
    ReportFileHandler.write('numOfComms_detected = '+str(numOfComms_detected)+'\n')
    ReportFileHandler.write('largestRelCommSize_detected = '+str(largestRelCommSize_detected)+'\n')
    ReportFileHandler.write('AMI_detected = '+str(AMI_detected)+'\n')
    ReportFileHandler.write('ARI_detected = '+str(ARI_detected)+'\n')
    ReportFileHandler.write('NMI_detected = '+str(NMI_detected)+'\n')
    ReportFileHandler.write('number of done embeddings = '+str(numOfDoneEmbeddings)+'\n')
    ReportFileHandler.write('number of embedding dimensions = '+str(d))
    ReportFileHandler.close()

    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/TREXPICresults.txt", 'w')
    ReportFileHandler.write(str(Q_detected)+'\n')
    ReportFileHandler.write(str(numOfComms_detected)+'\n')
    ReportFileHandler.write(str(largestRelCommSize_detected)+'\n')
    ReportFileHandler.write(str(AMI_detected)+'\n')
    ReportFileHandler.write(str(ARI_detected)+'\n')
    ReportFileHandler.write(str(NMI_detected)+'\n')
    ReportFileHandler.write(str(numOfDoneEmbeddings)+'\n')
    ReportFileHandler.write(str(d))
    ReportFileHandler.close()
