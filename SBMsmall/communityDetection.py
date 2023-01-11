#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import networkx as nx
import subprocess as sp
import community as cmL #Louvain
import infomap
from networkx.algorithms.community.label_propagation import asyn_lpa_communities as alabprop
from sklearn.metrics.cluster import adjusted_mutual_info_score
from networkx.algorithms.community.quality import modularity


#This function returns the hyperbolic distance between two nodes with polar coordinates [r1,phi1] and [r2,phi2].
def hypDist_2D(r1,phi1,r2,phi2):
    cos_dphi = math.cos(math.pi-math.fabs(math.pi-math.fabs(phi1-phi2))) #cosine of the angular distance between the two nodes
    if cos_dphi==1: #in this case the hyperbolic distance between the two nodes is acosh(cosh(r1-r2))=|r1-r2|
        h = math.fabs(r1-r2)
    else:
        argument_of_acosh = math.cosh(r1)*math.cosh(r2)-math.sinh(r1)*math.sinh(r2)*cos_dphi
        if argument_of_acosh<1: #a rounding error occurred, because the hyperbolic distance h is close to zero
            print("The argument of acosh is "+str(argument_of_acosh)+", less than 1.\nr1="+str(r1)+"\nr2="+str(r2)+"\nphi1="+str(phi1)+"\nphi2="+str(phi2))
            h = 0 #acosh(1)=0
        else:
            h = math.acosh(argument_of_acosh)
    return h


#This function creates the input for the edge weighting function in the case of two-dimensional embeddings: it returns a dictionary with edges (tuples) as keys and hyperbolic distances between the concerned nodes as values.
#G is a NetworkX graph
#r is a NumPy array, its ith element is the radial coordinate of the examined graph's ith node on the Poincaré disk
#phi is a NumPy array, its ith element is the angular coordinate of the examined graph's ith node on the Poincaré disk
def hypDistDict_2D(G,r,phi):
    hypDistDict={}
    for (i,j) in G.edges():
        hypDistDict[(i,j)]=hypDist_2D(r[i],phi[i],r[j],phi[j])
    return hypDistDict


#This function returns the hyperbolic distance between two nodes with spherical coordinates [r1,phi1,Theta1] and [r2,phi2,Theta2].
    #the angle phi is measured from the x axis in the x-y plane, counterclockwise; 0<=phi<2pi
    #the angle Theta is measured from the z axis; 0<=Theta<=pi
def hypDist_3D(r1,phi1,Theta1,r2,phi2,Theta2):
    dphi=math.pi-math.fabs(math.pi-math.fabs(phi1-phi2)) #0<=dphi<=pi  
    if Theta1<=math.pi/2: #upper hemisphere
        lat1=(math.pi/2)-Theta1
    else: #lower hemisphere
        lat1=-(Theta1-(math.pi/2))
    if Theta2<=math.pi/2: #upper hemisphere
        lat2=(math.pi/2)-Theta2
    else: #lower hemisphere
        lat2=-(Theta2-(math.pi/2))
    cos_dsigma = math.cos(math.atan(math.sqrt(math.pow(math.cos(lat2)*math.sin(dphi),2)+math.pow(math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dphi),2))/(math.sin(lat1)*math.sin(lat2)+math.cos(lat1)*math.cos(lat2)*math.cos(dphi)))) #cosine of the angular distance between the two nodes
    if cos_dsigma==1: #in this case the hyperbolic distance between the two nodes is h=acosh(cosh(r1-r2))=|r1-r2|
        h = math.fabs(r1-r2)
    else:
        argument_of_acosh = math.cosh(r1)*math.cosh(r2)-math.sinh(r1)*math.sinh(r2)*cos_dsigma
        if argument_of_acosh<1:  #a rounding error occurred, because the hyperbolic distance h is close to zero
            print("The argument of acosh is "+str(argument_of_acosh)+", less than 1.\nr1="+str(r1)+"\nr2="+str(r2)+"\nphi1="+str(phi1)+"\nphi2="+str(phi2)+"\nTheta1="+str(Theta1)+"\nTheta2="+str(Theta2))
            h = 0 #acosh(1)=0
        else:
            h = math.acosh(argument_of_acosh)
    return h


#This function creates the input for the edge weighting function in the case of three-dimensional embeddings: it returns a dictionary with edges (tuples) as keys and hyperbolic distances between the concerned nodes as values.
#G is a NetworkX graph
#r is a NumPy array, its ith element is the radial coordinate of the examined graph's ith node on the Poincaré ball
#phi is a NumPy array, its ith element is the azimuthal angle of the examined graph's ith node on the Poincaré ball
#Theta is a NumPy array, its ith element is the polar angle of the examined graph's ith node on the Poincaré ball
def hypDistDict_3D(G,r,phi,Theta):
    hypDistDict={}
    for (i,j) in G.edges():
        hypDistDict[(i,j)]=hypDist_3D(r[i],phi[i],Theta[i],r[j],phi[j],Theta[j])
    return hypDistDict


#This function weights the edges of a graph according to the hyperbolic distances between its nodes.
#G is the NetworkX graph to be weighted
#hypDistDict is a dictionary: the keys are the edges (tuples) and the values are the hyperbolic distances between the connected nodes
def weighting(G,hypDistDict):
    for (i,j) in G.edges():
        G[i][j]['weight'] = 1/(1+hypDistDict[(i,j)]) #hypDistDict[(i,j)] is the hyperbolic distance between node i and j


#This function determines the community structure of a graph with the Louvain method and returns a list of integers where the ith element describes the label of the community to which the graph's ith node belongs.
#G is the examined NetworkX graph
def Louvain(G):
    #dend = cmL.generate_dendrogram(G)
    #smallestComms = cmL.partition_at_level(dend,0)
    partition = cmL.best_partition(G) #dictionary with the obtained partition (key=a node's ID,value=its community); the communities are numbered from 0 with integers
    commStructList = [partition[nodeID] for nodeID in G] #commStructList[i]=the label of the community in which G's ith node belongs to
    return commStructList


#This function determines the community structure of a graph with the Infomap method in python and returns a list of integers where the ith element describes the label of the community to which the graph's ith node belongs.
#G is the examined NetworkX graph
def Infomap(G):
    infomInst = infomap.Infomap("--zero-based-numbering --undirected") #create an Infomap instance for multi-level clustering
    network = infomInst.network() #defalult (empty) network
    for (i,j) in G.edges():
        network.addLink(int(i),int(j),G.get_edge_data(i,j)['weight'])
    infomInst.run() #run the Infomap search algorithm to find optimal modules
    partition={} #dictionary with the obtained partition (key=a node's ID,value=its community)
    for node in infomInst.iterTree():
        if node.isLeaf():
            partition[node.physicalId] = node.moduleIndex()
    commStructList = [partition[int(nodeID)] for nodeID in G]
    return commStructList


#This function executes the C++ code of Infomap which creates and saves the hierarchy describing the community structure of the graph G, loads it and converts it to a list of integers where the ith element describes the label of the community to which the graph's ith node belongs.
#pathForExecutableFile=path for the executable file (e.g. pathForExecutableFile = os.getcwd()+"/Infomap")
#directoryName=the name of the directory containing the edge list; the resulted tree will be saved here
#edgeListName=the name of the edge list's text file without ".txt"
def InfomapC(G,pathForExecutableFile,directoryName,edgeListName):
    iPath = directoryName+"/"+edgeListName+".txt"
    oPath = directoryName+"/"
    args = [pathForExecutableFile, iPath, oPath, "--input-format link-list", "--tree", "--zero-based-numbering", "--undirected"] #,"--preferred-number-of-modules 20"
    print("Executing ", pathForExecutableFile, "...")
    proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE, shell=False)
    print(args)
    (out, err) = proc.communicate(input = None)
    print("Output from the program: ", out,err)
    if (err != None and len(err) > 0):
        print("Error while executing the program! Error code: ", proc.returncode, "Output: ", err)

    fileHandler = open(os.getcwd()+"/"+oPath+edgeListName+".tree","r")
    delimiter=" "
    partition={} #dictionary with the obtained partition (key=a node's ID,value=its community); the communities are numbered from 0 with integers
    groupID=-1
    while True:
        #Get next line from file
        line = fileHandler.readline()
        if not line: #line is empty (end of the file)
            break;
        if line[0] != "#": #the current line is not a comment
            listOfWords = line.split(delimiter)
            nodeID = int(listOfWords[-2][1:-1]) #the node identifier (without "", converted to int)
            IDwithinGroup = listOfWords[0][-2:]
            if IDwithinGroup == ":1": #the current node belongs to a new group
                groupID = groupID+1
            partition[nodeID] = groupID
    fileHandler.close()
    commStructList = [partition[nodeID] for nodeID in G] #commStructList[i]=the label of the community in which G's ith node belongs to
    return commStructList    


#community detection with asynchronous label propagation
def asynLabProp(G):
    comms=alabprop(G,weight='weight') #iterable of communities given as sets of nodes
    partition={} #dictionary with the obtained partition (key=a node's ID,value=its community); the communities are numbered from 0 with integers
    groupID=0
    for c in comms: #c is a set of nodes forming a community
        for nodeID in c:
            partition[nodeID] = groupID
        groupID=groupID+1
    commStructList = [partition[nodeID] for nodeID in G] #commStructList[i]=the label of the community in which G's ith node belongs to
    return commStructList    


#This function evaluates the community detection and returns the adjusted mutual information between the true and the predicted community structure of a network.
	#0 means no mutual information and 1 means perfect correlation between the two categorization
#realComms is the real community structure of a network
#detectedComms is the community structure determined by a community detection algorithm
	#each community structure is given by a list of integers where the ith element describes the label of the community to which the graph's ith node belongs
def adjMutInf(realComms,detectedComms):
    AMI = adjusted_mutual_info_score(realComms,detectedComms,average_method='max') #the 'max' method results in the smallest AMI value from all the possible normalizations -> calculate the possible worst AMI
    return AMI


#This function evaluates the community detection and returns the modularity of the detected community structure.
        #larger modularity means stronger division of the network into modules; it is positive if the number of edges within groups exceeds the number expected on the basis of chance
#G is the examined NetworkX graph
#N is the number of nodes in the network
#detectedComms is the community structure determined by a community detection algorithm; the community structure is given by a list of integers where the ith element describes the label of the community to which the graph's ith node belongs
def resultedModularity(G,N,detectedComms):
    numberOfCommunities = len(set(detectedComms))
    nodelistPerCommunity = []
    for i in range(numberOfCommunities):
        nodelistPerCommunity.append([]) #create a list for each community
                                        #the communities are labeled with 0,1,...,numberOfCommunities-1 integers; the ith list in nodelistPerCommunity contains the index of those nodes, which belong to the ith community
    for i in range(N):
        nodelistPerCommunity[detectedComms[i]].append(i) #detectedComms[i] is the label of the community to which the graph's ith node belongs according to the detected community structure
    nodesetPerCommunity = [] #list of set of nodes
    for i in range(numberOfCommunities):
        nodesetPerCommunity.append(set(nodelistPerCommunity[i]))
    mod = modularity(G,nodesetPerCommunity,weight='weight')
    return mod
