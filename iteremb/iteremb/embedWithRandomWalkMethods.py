#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import networkx as nx
import os
import copy
from collections import Counter
import embedding as em
import communityDetection as cD



wDiffOfStop = 0.01 #the iteration is stopped if the relative change in the average of the link weights is not larger than this number...
maxNumOfEmbeddings = 25 #...or if the number of embeddings reached this number



#load data
directoryName = "exampleNetwork"
edgeListFileName = "edgeList.txt"
commStructFileName = "communityList.txt"
#load the graph
G_original=em.loadGraph(os.getcwd()+"/"+directoryName+"/"+edgeListFileName,0,"\t") #the largest connected component, without self-loops and parallel edges; if no link weights are given, then all the link weights are set to 1.0
nodeList = list(G_original.nodes())
numOfNodes = len(nodeList)
numOfEdges = len(G_original.edges())
print('number of nodes='+str(numOfNodes)+', number of edges='+str(numOfEdges)+', average degree='+str(2*numOfEdges/numOfNodes))
#load the real community structure
groupDict_real = {} #key=node name, value=group name
ReportFileHandler=open(os.getcwd()+"/"+directoryName+"/"+commStructFileName,'r')
listOfLines=ReportFileHandler.readlines()
ReportFileHandler.close()
for line in listOfLines:
    listOfWords=line.split('\t')
    nodeID=listOfWords[0]
    commID=listOfWords[1]
    groupDict_real[nodeID] = commID
numOfComms_real=len(set(groupDict_real.values()))
print('real number of groups='+str(numOfComms_real))
commSizeDict = Counter(groupDict_real.values()) #key=groupID, value=number of members
largestRelCommSize_real = max(commSizeDict.values())/numOfNodes
Q_real = cD.calculateModularity(G_original,groupDict_real)



#search for communities with traditional methods on the original, unweighted graph: an example for using Louvain
groupDict_detected = cD.Louvain(G_original)
#calculate the modularity
Q_Louvain = cD.calculateModularity(G_original,groupDict_detected)
#store the number of detected communities
numOfComms_Louvain = len(set(groupDict_detected.values()))
#examine the community sizes
commSizeDict = Counter(groupDict_detected.values()) #key=groupID, value=number of members
largestRelCommSize_Louvain = max(commSizeDict.values())/numOfNodes
#calculate AMI, ARI and element-centric similarity between the detected and the planted community structures
[AMI_Louvain,ARI_Louvain,elemCentSimilarity_Louvain] = cD.calculateSimilarityMeasures(groupDict_real,groupDict_detected)

#save the results obtained from Louvain
ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/Louvain.txt", 'w')
#properties of the examined network
ReportFileHandler.write('numOfNodes = '+str(numOfNodes)+'\n')
ReportFileHandler.write('numOfEdges = '+str(numOfEdges)+'\n')
ReportFileHandler.write('numOfComms_real = '+str(numOfComms_real)+'\n')
ReportFileHandler.write('largestRelCommSize_real = '+str(largestRelCommSize_real)+'\n')
ReportFileHandler.write('Q_real = '+str(Q_real)+'\n')
#community detection on the original, unweighted network
ReportFileHandler.write('numOfComms_Louvain = '+str(numOfComms_Louvain)+'\n')
ReportFileHandler.write('largestRelCommSize_Louvain = '+str(largestRelCommSize_Louvain)+'\n')
ReportFileHandler.write('Q_Louvain = '+str(Q_Louvain)+'\n')
ReportFileHandler.write('AMI_Louvain = '+str(AMI_Louvain)+'\n')
ReportFileHandler.write('ARI_Louvain = '+str(ARI_Louvain)+'\n')
ReportFileHandler.write('elemCentSimilarity_Louvain = '+str(elemCentSimilarity_Louvain)+'\n')
ReportFileHandler.close()




d = 64 #the higher number of embedding dimensions seems to be better in the case of node2vec; we always use here d=64

#set the q multiplying factor that will be used in all the embedding iterations for mapping the angular distances to exponential link weights
degreeList = [G_original.degree(nbunch=nodeName,weight=None) for nodeName in G_original.nodes()] #Consider here the number of connections, not the sum of weights! (This does not change during the iterations, only the link weights.)
degreeFrequencyDict = Counter(degreeList) #key=degree, value=number of nodes having the given degree
highestDegreeFrequency = max(degreeFrequencyDict.values())
mostFrequentDegrees = [deg for deg in degreeFrequencyDict.keys() if degreeFrequencyDict[deg]==highestDegreeFrequency]
degreeMode=min(mostFrequentDegrees) #the smallest degree value from those that occurrs the most frequently
print('mode of node degrees='+str(degreeMode))
degreeAvg=np.mean(degreeList) #the average value of the node degrees
print('average of node degrees='+str(degreeAvg))
qFactor=10*(degreeAvg/degreeMode)
print('The default q multiplying factor is '+str(qFactor)+'.')



###embedding iteration###

#embed the original graph, create the first embedding
print('node2vec with exp weight - 1')
[G_current,Coord_current]=em.expNode2vec(G_original,qFactor,d) #embed the network with node2vec and weight the link according to an exponential formula
numOfDoneEmbeddings = 1

#check for convergence: calculate the difference between the current and the previous weights
avgWeight_previous = np.mean([G_original[s][t]['weight'] for (s,t) in G_original.edges()])
avgWeight_current = np.mean([G_current[s][t]['weight'] for (s,t) in G_current.edges()])
relChangeInAvgWeight = math.fabs(avgWeight_current-avgWeight_previous)/avgWeight_current
print("The relative difference of the average link weight at iteration "+str(numOfDoneEmbeddings)+" is "+str(relChangeInAvgWeight))

#Measure the angular separation between the nodes after each embedding, just to see what happens during the iterations. (This measure is not needed by the iteration, it just provides some more information in addition to the community detection performances.)
separationMeasureList = []
numeratorList = []
denominatorList = []
distanceWithinComms = 0
numOfNodePairsWithinComms = 0
distanceBetweenComms = 0
numOfNodePairsBetweenComms = 0
for node1ID in range(1,numOfNodes):
    nodeName1 = nodeList[node1ID]
    for node2ID in range(node1ID):
        nodeName2 = nodeList[node2ID]
        argOfAcos = np.inner(Coord_current[nodeName1],Coord_current[nodeName2]) / (np.linalg.norm(Coord_current[nodeName1])*np.linalg.norm(Coord_current[nodeName2])) #cos(angular distance)
        if argOfAcos>1.0: #a numerical error has occurred
            contribution = 0.0
        elif argOfAcos<-1.0: #a numerical error has occurred
            contribution = math.pi
        else:
            contribution = math.acos(argOfAcos)
        if groupDict_real[nodeName1]==groupDict_real[nodeName2]: #nodes of the same group
            distanceWithinComms = distanceWithinComms+contribution
            numOfNodePairsWithinComms = numOfNodePairsWithinComms+1
        else: #nodes of different groups
            distanceBetweenComms = distanceBetweenComms+contribution
            numOfNodePairsBetweenComms = numOfNodePairsBetweenComms+1
numeratorList.append(distanceBetweenComms/numOfNodePairsBetweenComms)
denominatorList.append(distanceWithinComms/numOfNodePairsWithinComms)
try:
    separationMeasureList.append(numeratorList[numOfDoneEmbeddings-1]/denominatorList[numOfDoneEmbeddings-1])
except ZeroDivisionError:
    separationMeasureList.append(math.inf)

#evaluate the first embedding: measure the community detection performance in the non-iterated case
#community detection with a really simple weight thresholding:
groupDict_detected = cD.commDetWithWeightThresholding(G_current,'large') #the link weights mean proximities -> keep the links having a weight above a threshold
#calculate the modularity
Q_wThreshold_1 = cD.calculateModularity(G_original,groupDict_detected)
#store the number of detected communities
numOfComms_wThreshold_1 = len(set(groupDict_detected.values()))
#examine the community sizes
commSizeDict = Counter(groupDict_detected.values()) #key=groupID, value=number of members
largestRelCommSize_wThreshold_1 = max(commSizeDict.values())/numOfNodes
#calculate AMI, ARI and element-centric similarity between the detected and the planted community structures
[AMI_wThreshold_1,ARI_wThreshold_1,elemCentSimilarity_wThreshold_1] = cD.calculateSimilarityMeasures(groupDict_real,groupDict_detected)

#community detection after the first embedding with traditional methods
#the link weights mean proximities -> there is no need for a conversion here
#search for communities e.g. with Louvain:
groupDict_detected = cD.Louvain(G_current)
#To use Infomap: groupDict_detected = cD.Infomap(G_current)
#To use asynchronous label propagation: groupDict_detected = cD.asynLabProp(G_current)
#calculate the modularity
Q_Louvain_1 = cD.calculateModularity(G_original,groupDict_detected)
#store the number of detected communities
numOfComms_Louvain_1 = len(set(groupDict_detected.values()))
#examine the community sizes
commSizeDict = Counter(groupDict_detected.values()) #key=groupID, value=number of members
largestRelCommSize_Louvain_1 = max(commSizeDict.values())/numOfNodes
#calculate AMI, ARI and element-centric similarity between the detected and the planted community structures
[AMI_Louvain_1,ARI_Louvain_1,elemCentSimilarity_Louvain_1] = cD.calculateSimilarityMeasures(groupDict_real,groupDict_detected)

#an example for using k-means clustering on the first embedding:
groupDict_detected = cD.kMeansClustering(Coord_current,numOfComms_real)
#calculate the modularity
Q_kMeans_1 = cD.calculateModularity(G_original,groupDict_detected)
#store the number of detected communities
numOfComms_kMeans_1 = len(set(groupDict_detected.values()))
#examine the community sizes
commSizeDict = Counter(groupDict_detected.values()) #key=groupID, value=number of members
largestRelCommSize_kMeans_1 = max(commSizeDict.values())/numOfNodes
#calculate AMI, ARI and element-centric similarity between the detected and the planted community structures
[AMI_kMeans_1,ARI_kMeans_1,elemCentSimilarity_kMeans_1] = cD.calculateSimilarityMeasures(groupDict_real,groupDict_detected)



#create further embeddings
while relChangeInAvgWeight>wDiffOfStop and numOfDoneEmbeddings<maxNumOfEmbeddings:
    G_previous = copy.deepcopy(G_current) #store the graph obtained from the previous embedding
    print('node2vec with exp weight - '+str(numOfDoneEmbeddings+1))
    [G_current,Coord_current]=em.expNode2vec(G_previous,qFactor,d) #embed the network with node2vec and weight the link according to an exponential formula
    numOfDoneEmbeddings = numOfDoneEmbeddings+1
    
    #check for convergence: calculate the difference between the current and the previous weights
    avgWeight_previous = np.mean([G_previous[s][t]['weight'] for (s,t) in G_previous.edges()])
    avgWeight_current = np.mean([G_current[s][t]['weight'] for (s,t) in G_current.edges()])
    relChangeInAvgWeight = math.fabs(avgWeight_current-avgWeight_previous)/avgWeight_current
    print("The relative difference of the average link weight at iteration "+str(numOfDoneEmbeddings)+" is "+str(relChangeInAvgWeight))
    
    #Measure the angular separation between the nodes after each embedding, just to see what happens during the iterations. (This measure is not needed by the iteration, it just provides some more information in addition to the community detection performances.)
    distanceWithinComms = 0
    distanceBetweenComms = 0
    for node1ID in range(1,numOfNodes):
        nodeName1 = nodeList[node1ID]
        for node2ID in range(node1ID):
            nodeName2 = nodeList[node2ID]
            argOfAcos = np.inner(Coord_current[nodeName1],Coord_current[nodeName2]) / (np.linalg.norm(Coord_current[nodeName1])*np.linalg.norm(Coord_current[nodeName2])) #cos(angular distance)
            if argOfAcos>1.0: #a numerical error has occurred
                contribution = 0.0
            elif argOfAcos<-1.0: #a numerical error has occurred
                contribution = math.pi
            else:
                contribution = math.acos(argOfAcos)
            if groupDict_real[nodeName1]==groupDict_real[nodeName2]: #nodes of the same group
                distanceWithinComms = distanceWithinComms+contribution
            else: #nodes of different groups
                distanceBetweenComms = distanceBetweenComms+contribution
    numeratorList.append(distanceBetweenComms/numOfNodePairsBetweenComms)
    denominatorList.append(distanceWithinComms/numOfNodePairsWithinComms)
    try:
        separationMeasureList.append(numeratorList[numOfDoneEmbeddings-1]/denominatorList[numOfDoneEmbeddings-1])
    except ZeroDivisionError:
        separationMeasureList.append(math.inf)

#the state of "convergence" (or the maximum number of iterations) has been reached -> search for communities (measure the community detection performance in the iterated case)
numOfEmbsAtConv = numOfDoneEmbeddings
print('The number of embeddings until convergence: '+str(numOfEmbsAtConv))

#community detection with a really simple weight thresholding:
groupDict_detected = cD.commDetWithWeightThresholding(G_current,'large') #the link weights mean proximities -> keep the links having a weight above a threshold
#calculate the modularity
Q_wThreshold_conv = cD.calculateModularity(G_original,groupDict_detected)
#store the number of detected communities
numOfComms_wThreshold_conv = len(set(groupDict_detected.values()))
#examine the community sizes
commSizeDict = Counter(groupDict_detected.values()) #key=groupID, value=number of members
largestRelCommSize_wThreshold_conv = max(commSizeDict.values())/numOfNodes
#calculate AMI, ARI and element-centric similarity between the detected and the planted community structures
[AMI_wThreshold_conv,ARI_wThreshold_conv,elemCentSimilarity_wThreshold_conv] = cD.calculateSimilarityMeasures(groupDict_real,groupDict_detected)

#community detection after the embedding "convergence" with traditional methods
#the link weights mean proximities -> there is no need for a conversion here
#search for communities e.g. with Louvain:
groupDict_detected = cD.Louvain(G_current)
#To use Infomap: groupDict_detected = cD.Infomap(G_current)
#To use asynchronous label propagation: groupDict_detected = cD.asynLabProp(G_current)
#calculate the modularity
Q_Louvain_conv = cD.calculateModularity(G_original,groupDict_detected)
#store the number of detected communities
numOfComms_Louvain_conv = len(set(groupDict_detected.values()))
#examine the community sizes
commSizeDict = Counter(groupDict_detected.values()) #key=groupID, value=number of members
largestRelCommSize_Louvain_conv = max(commSizeDict.values())/numOfNodes
#calculate AMI, ARI and element-centric similarity between the detected and the planted community structures
[AMI_Louvain_conv,ARI_Louvain_conv,elemCentSimilarity_Louvain_conv] = cD.calculateSimilarityMeasures(groupDict_real,groupDict_detected)

#an example for using k-means clustering on the final embedding:
groupDict_detected = cD.kMeansClustering(Coord_current,numOfComms_real)
#calculate the modularity
Q_kMeans_conv = cD.calculateModularity(G_original,groupDict_detected)
#store the number of detected communities
numOfComms_kMeans_conv = len(set(groupDict_detected.values()))
#examine the community sizes
commSizeDict = Counter(groupDict_detected.values()) #key=groupID, value=number of members
largestRelCommSize_kMeans_conv = max(commSizeDict.values())/numOfNodes
#calculate AMI, ARI and element-centric similarity between the detected and the planted community structures
[AMI_kMeans_conv,ARI_kMeans_conv,elemCentSimilarity_kMeans_conv] = cD.calculateSimilarityMeasures(groupDict_real,groupDict_detected)



###save the results###

#save the separation rate at each iteration
np.savetxt(os.getcwd() + "/" + directoryName + "/expNode2vec_angSeparation.txt", separationMeasureList)
np.savetxt(os.getcwd() + "/" + directoryName + "/expNode2vec_angSepNumerator.txt", numeratorList)
np.savetxt(os.getcwd() + "/" + directoryName + "/expNode2vec_angSepDenominator.txt", denominatorList)

#save the community detection performances in the case of the simple weight thresholding
ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/expNode2vec_commDetWithThresholding.txt", 'w')
#properties of the examined network
ReportFileHandler.write('numOfNodes = '+str(numOfNodes)+'\n')
ReportFileHandler.write('numOfEdges = '+str(numOfEdges)+'\n')
ReportFileHandler.write('numOfComms_real = '+str(numOfComms_real)+'\n')
ReportFileHandler.write('largestRelCommSize_real = '+str(largestRelCommSize_real)+'\n')
ReportFileHandler.write('Q_real = '+str(Q_real)+'\n')
#community detection after the first embedding
ReportFileHandler.write('numOfComms_wThreshold_1 = '+str(numOfComms_wThreshold_1)+'\n')
ReportFileHandler.write('largestRelCommSize_wThreshold_1 = '+str(largestRelCommSize_wThreshold_1)+'\n')
ReportFileHandler.write('Q_wThreshold_1 = '+str(Q_wThreshold_1)+'\n')
ReportFileHandler.write('AMI_wThreshold_1 = '+str(AMI_wThreshold_1)+'\n')
ReportFileHandler.write('ARI_wThreshold_1 = '+str(ARI_wThreshold_1)+'\n')
ReportFileHandler.write('elemCentSimilarity_wThreshold_1 = '+str(elemCentSimilarity_wThreshold_1)+'\n')
#community detection after the "convergence" of the embedding iteration
ReportFileHandler.write('numOfComms_wThreshold_conv = '+str(numOfComms_wThreshold_conv)+'\n')
ReportFileHandler.write('largestRelCommSize_wThreshold_conv = '+str(largestRelCommSize_wThreshold_conv)+'\n')
ReportFileHandler.write('Q_wThreshold_conv = '+str(Q_wThreshold_conv)+'\n')
ReportFileHandler.write('AMI_wThreshold_conv = '+str(AMI_wThreshold_conv)+'\n')
ReportFileHandler.write('ARI_wThreshold_conv = '+str(ARI_wThreshold_conv)+'\n')
ReportFileHandler.write('elemCentSimilarity_wThreshold_conv = '+str(elemCentSimilarity_wThreshold_conv)+'\n')
#properties of embedding
ReportFileHandler.write('number of embeddings at convergence = '+str(numOfEmbsAtConv)+'\n')
ReportFileHandler.write('number of embedding dimensions = '+str(d))
ReportFileHandler.close()


#save the community detection performances in the case of using Louvain after Laplacian Eigenmaps
ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/expNode2vec_Louvain.txt", 'w')
#properties of the examined network
ReportFileHandler.write('numOfNodes = '+str(numOfNodes)+'\n')
ReportFileHandler.write('numOfEdges = '+str(numOfEdges)+'\n')
ReportFileHandler.write('numOfComms_real = '+str(numOfComms_real)+'\n')
ReportFileHandler.write('largestRelCommSize_real = '+str(largestRelCommSize_real)+'\n')
ReportFileHandler.write('Q_real = '+str(Q_real)+'\n')
#community detection after the first embedding
ReportFileHandler.write('numOfComms_Louvain_1 = '+str(numOfComms_Louvain_1)+'\n')
ReportFileHandler.write('largestRelCommSize_Louvain_1 = '+str(largestRelCommSize_Louvain_1)+'\n')
ReportFileHandler.write('Q_Louvain_1 = '+str(Q_Louvain_1)+'\n')
ReportFileHandler.write('AMI_Louvain_1 = '+str(AMI_Louvain_1)+'\n')
ReportFileHandler.write('ARI_Louvain_1 = '+str(ARI_Louvain_1)+'\n')
ReportFileHandler.write('elemCentSimilarity_Louvain_1 = '+str(elemCentSimilarity_Louvain_1)+'\n')
#community detection after the "convergence" of the embedding iteration
ReportFileHandler.write('numOfComms_Louvain_conv = '+str(numOfComms_Louvain_conv)+'\n')
ReportFileHandler.write('largestRelCommSize_Louvain_conv = '+str(largestRelCommSize_Louvain_conv)+'\n')
ReportFileHandler.write('Q_Louvain_conv = '+str(Q_Louvain_conv)+'\n')
ReportFileHandler.write('AMI_Louvain_conv = '+str(AMI_Louvain_conv)+'\n')
ReportFileHandler.write('ARI_Louvain_conv = '+str(ARI_Louvain_conv)+'\n')
ReportFileHandler.write('elemCentSimilarity_Louvain_conv = '+str(elemCentSimilarity_Louvain_conv)+'\n')
#properties of embedding
ReportFileHandler.write('number of embeddings at convergence = '+str(numOfEmbsAtConv)+'\n')
ReportFileHandler.write('number of embedding dimensions = '+str(d))
ReportFileHandler.close()


#save the community detection performances in the case of using k-means clustering after Laplacian Eigenmaps
ReportFileHandler = open(os.getcwd() + "/" + directoryName + "/expNode2vec_kMeans.txt", 'w')
#properties of the examined network
ReportFileHandler.write('numOfNodes = '+str(numOfNodes)+'\n')
ReportFileHandler.write('numOfEdges = '+str(numOfEdges)+'\n')
ReportFileHandler.write('numOfComms_real = '+str(numOfComms_real)+'\n')
ReportFileHandler.write('largestRelCommSize_real = '+str(largestRelCommSize_real)+'\n')
ReportFileHandler.write('Q_real = '+str(Q_real)+'\n')
#community detection after the first embedding
ReportFileHandler.write('numOfComms_kMeans_1 = '+str(numOfComms_kMeans_1)+'\n')
ReportFileHandler.write('largestRelCommSize_kMeans_1 = '+str(largestRelCommSize_kMeans_1)+'\n')
ReportFileHandler.write('Q_kMeans_1 = '+str(Q_kMeans_1)+'\n')
ReportFileHandler.write('AMI_kMeans_1 = '+str(AMI_kMeans_1)+'\n')
ReportFileHandler.write('ARI_kMeans_1 = '+str(ARI_kMeans_1)+'\n')
ReportFileHandler.write('elemCentSimilarity_kMeans_1 = '+str(elemCentSimilarity_kMeans_1)+'\n')
#community detection after the "convergence" of the embedding iteration
ReportFileHandler.write('numOfComms_kMeans_conv = '+str(numOfComms_kMeans_conv)+'\n')
ReportFileHandler.write('largestRelCommSize_kMeans_conv = '+str(largestRelCommSize_kMeans_conv)+'\n')
ReportFileHandler.write('Q_kMeans_conv = '+str(Q_kMeans_conv)+'\n')
ReportFileHandler.write('AMI_kMeans_conv = '+str(AMI_kMeans_conv)+'\n')
ReportFileHandler.write('ARI_kMeans_conv = '+str(ARI_kMeans_conv)+'\n')
ReportFileHandler.write('elemCentSimilarity_kMeans_conv = '+str(elemCentSimilarity_kMeans_conv)+'\n')
#properties of embedding
ReportFileHandler.write('number of embeddings at convergence = '+str(numOfEmbsAtConv)+'\n')
ReportFileHandler.write('number of embedding dimensions = '+str(d))
ReportFileHandler.close()
