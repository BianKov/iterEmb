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
import communityDetection as cD


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

for mu in muList:
    print('mu='+str(mu))
    
    #load data
    directoryName = "SBM_"+dirNameDict[mu]+"/graph"+str(graphID)
    edgeListFileName = "SBM_edges.txt"
    commStructFileName = "blocks.txt"
    #load the graph
    G_original=em.loadGraph(os.getcwd()+"/"+directoryName+"/"+edgeListFileName,0,"\t") #the largest connected component, without self-loops, parallel edges and link weights
    print('number of nodes='+str(len(G_original))+', number of edges='+str(len(G_original.edges()))+', average degree='+str(2*len(G_original.edges())/len(G_original)))
    for (i,j) in G_original.edges():
        G_original[i][j]['weight']=1.0
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
    #examine the real community structure
    Q_real = calculateModularity(G_original,groupDict_real)
    numOfComms_real = len(set(groupDict_real.values()))
    largestCommSize = 0
    for group in set(groupDict_real.values()):
        commSize = len({node for node in groupDict_real if groupDict_real[node]==group})
        if largestCommSize<commSize:
            largestCommSize = commSize
    largestRelCommSize_real = largestCommSize/len(G_original)

    #save the results
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/realProperties_readable.txt", 'w')
    ReportFileHandler.write('Q_real = '+str(Q_real)+'\n')
    ReportFileHandler.write('numOfComms_real = '+str(numOfComms_real)+'\n')
    ReportFileHandler.write('largestRelCommSize_real = '+str(largestRelCommSize_real)+'\n')
    ReportFileHandler.close()
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/realProperties.txt", 'w')
    ReportFileHandler.write(str(Q_real)+'\n')
    ReportFileHandler.write(str(numOfComms_real)+'\n')
    ReportFileHandler.write(str(largestRelCommSize_real)+'\n')
    ReportFileHandler.close()

    #community detection with traditional methods
    print('Louvain')
    commStruct_Louvain = cD.Louvain(G_original)
    groupDict_Louvain = {} #key=nodeID, value=value (name (string) of the group)
    nodeID=0
    for node in G_original:
        groupDict_Louvain[node] = commStruct_Louvain[nodeID]
        nodeID = nodeID+1
    #calculate the modularity
    Q_Louvain = calculateModularity(G_original,groupDict_Louvain)
    #examine the community structure
    numOfComms_Louvain = len(set(groupDict_Louvain.values()))
    largestCommSize = 0
    for group in set(groupDict_Louvain.values()):
        commSize = len({node for node in groupDict_Louvain if groupDict_Louvain[node]==group})
        if largestCommSize<commSize:
            largestCommSize = commSize
    largestRelCommSize_Louvain = largestCommSize/len(G_original)
    #calculate the AMI, ARI and NMI values
    [AMI_Louvain,ARI_Louvain,NMI_Louvain] = AMI_ARI_NMI_calculation(commStruct_real,commStruct_Louvain)
    #save the results
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/LouvainResults_readable.txt", 'w')
    ReportFileHandler.write('Q_detected = '+str(Q_Louvain)+'\n')
    ReportFileHandler.write('numOfComms_detected = '+str(numOfComms_Louvain)+'\n')
    ReportFileHandler.write('largestRelCommSize_detected = '+str(largestRelCommSize_Louvain)+'\n')
    ReportFileHandler.write('AMI_detected = '+str(AMI_Louvain)+'\n')
    ReportFileHandler.write('ARI_detected = '+str(ARI_Louvain)+'\n')
    ReportFileHandler.write('NMI_detected = '+str(NMI_Louvain)+'\n')
    ReportFileHandler.close()
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/LouvainResults.txt", 'w')
    ReportFileHandler.write(str(Q_Louvain)+'\n')
    ReportFileHandler.write(str(numOfComms_Louvain)+'\n')
    ReportFileHandler.write(str(largestRelCommSize_Louvain)+'\n')
    ReportFileHandler.write(str(AMI_Louvain)+'\n')
    ReportFileHandler.write(str(ARI_Louvain)+'\n')
    ReportFileHandler.write(str(NMI_Louvain)+'\n')
    ReportFileHandler.close()

    print('asynchronous label propagation')
    commStruct_alabprop = cD.asynLabProp(G_original)
    groupDict_alabprop = {} #key=nodeID, value=value (name (string) of the group)
    nodeID=0
    for node in G_original:
        groupDict_alabprop[node] = commStruct_alabprop[nodeID]
        nodeID = nodeID+1
    #calculate the modularity
    Q_alabprop = calculateModularity(G_original,groupDict_alabprop)
    #examine the community structure
    numOfComms_alabprop = len(set(groupDict_alabprop.values()))
    largestCommSize = 0
    for group in set(groupDict_alabprop.values()):
        commSize = len({node for node in groupDict_alabprop if groupDict_alabprop[node]==group})
        if largestCommSize<commSize:
            largestCommSize = commSize
    largestRelCommSize_alabprop = largestCommSize/len(G_original)
    #calculate the AMI, ARI and NMI values
    [AMI_alabprop,ARI_alabprop,NMI_alabprop] = AMI_ARI_NMI_calculation(commStruct_real,commStruct_alabprop)
    #save the results
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/alabpropResults_readable.txt", 'w')
    ReportFileHandler.write('Q_detected = '+str(Q_alabprop)+'\n')
    ReportFileHandler.write('numOfComms_detected = '+str(numOfComms_alabprop)+'\n')
    ReportFileHandler.write('largestRelCommSize_detected = '+str(largestRelCommSize_alabprop)+'\n')
    ReportFileHandler.write('AMI_detected = '+str(AMI_alabprop)+'\n')
    ReportFileHandler.write('ARI_detected = '+str(ARI_alabprop)+'\n')
    ReportFileHandler.write('NMI_detected = '+str(NMI_alabprop)+'\n')
    ReportFileHandler.close()
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/alabpropResults.txt", 'w')
    ReportFileHandler.write(str(Q_alabprop)+'\n')
    ReportFileHandler.write(str(numOfComms_alabprop)+'\n')
    ReportFileHandler.write(str(largestRelCommSize_alabprop)+'\n')
    ReportFileHandler.write(str(AMI_alabprop)+'\n')
    ReportFileHandler.write(str(ARI_alabprop)+'\n')
    ReportFileHandler.write(str(NMI_alabprop)+'\n')
    ReportFileHandler.close()

    print('Infomap')
    commStruct_Infomap = cD.Infomap(G_original)
    groupDict_Infomap = {} #key=nodeID, value=value (name (string) of the group)
    nodeID=0
    for node in G_original:
        groupDict_Infomap[node] = commStruct_Infomap[nodeID]
        nodeID = nodeID+1
    #calculate the modularity
    Q_Infomap = calculateModularity(G_original,groupDict_Infomap)
    #examine the community structure
    numOfComms_Infomap = len(set(groupDict_Infomap.values()))
    largestCommSize = 0
    for group in set(groupDict_Infomap.values()):
        commSize = len({node for node in groupDict_Infomap if groupDict_Infomap[node]==group})
        if largestCommSize<commSize:
            largestCommSize = commSize
    largestRelCommSize_Infomap = largestCommSize/len(G_original)
    #calculate the AMI, ARI and NMI values
    [AMI_Infomap,ARI_Infomap,NMI_Infomap] = AMI_ARI_NMI_calculation(commStruct_real,commStruct_Infomap)
    #save the results
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/InfomapResults_readable.txt", 'w')
    ReportFileHandler.write('Q_detected = '+str(Q_Infomap)+'\n')
    ReportFileHandler.write('numOfComms_detected = '+str(numOfComms_Infomap)+'\n')
    ReportFileHandler.write('largestRelCommSize_detected = '+str(largestRelCommSize_Infomap)+'\n')
    ReportFileHandler.write('AMI_detected = '+str(AMI_Infomap)+'\n')
    ReportFileHandler.write('ARI_detected = '+str(ARI_Infomap)+'\n')
    ReportFileHandler.write('NMI_detected = '+str(NMI_Infomap)+'\n')
    ReportFileHandler.close()
    ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/InfomapResults.txt", 'w')
    ReportFileHandler.write(str(Q_Infomap)+'\n')
    ReportFileHandler.write(str(numOfComms_Infomap)+'\n')
    ReportFileHandler.write(str(largestRelCommSize_Infomap)+'\n')
    ReportFileHandler.write(str(AMI_Infomap)+'\n')
    ReportFileHandler.write(str(ARI_Infomap)+'\n')
    ReportFileHandler.write(str(NMI_Infomap)+'\n')
    ReportFileHandler.close()

