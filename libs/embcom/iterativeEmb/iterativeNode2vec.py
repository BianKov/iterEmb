#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
import math
import fastnode2vec #the fast version of node2vec
import statistics #for estimating the multiplying factor in the weight - angular distance function used for node2vec



#A function for creating a d-dimensional Euclidean embedding of a graph using iterated node2vec.
#G is the (weighted) NetworkX Graph to be embedded (If link weights are inputted, these must mean proximities, i.e. larger value:=higher similarity or stronger connection!)
#numOfIters is the total number of embedding iterations
#d is the number of dimensions of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the weight function: link weight=exp(q*(1+cos(angular distance)))-1
#walkLength, numOfWalks, windowLength, batchWalks, pForEmbedding, qForEmbedding: the usual parameters of the node2vec method
#The function returns the dictionary Coord where key=node name and value=NumPy array containing the Cartesian coordinates of the given node's position vector
def iteratedExpNode2vec(G,numOfIters,d=64,q=None,walkLength=10,numOfWalks=80,windowLength=10,batchWalks=10000,pForEmbedding=1.0,qForEmbedding=1.0):
    if q==None: #set the q multiplying factor that will be used for mapping the angular distances to link weights to the default value
        degreeList = [G.degree(nbunch=nodeName,weight=None) for nodeName in G.nodes()]
        try:
            degreeMode=statistics.mode(degreeList) #the degree value that occurrs the most frequently
            print('mode of node degrees='+str(degreeMode))
        except statistics.StatisticsError: #there is more than 1 modes - to be fair, we should find all the modes here and then choose e.g. the smallest, but yet, we just simply choose the smallest degree
            degreeMode=min(degreeList)
            print('min of node degrees='+str(degreeMode))
        degreeAvg=np.mean(degreeList) #the average value of the node degrees
        print('average of node degrees='+str(degreeAvg))
        qFactor=10*(degreeAvg/degreeMode)
        print('The default q multiplying factor is '+str(qFactor)+'.')
	
    #embedding iteration
	numOfDoneIters = 0
    listOfNodes = list(G.nodes())
	G_embWeighted = copy.deepcopy(G) #initialization of the network's weighted version
    while numOfDoneIters<numOfIters:
	    #create an embedding
        A = nx.adjacency_matrix(G_embWeighted,nodelist=listOfNodes,weight='weight') #adjacency matrix as a SciPy Compressed Sparse Row matrix
        model = fastnode2vec.Node2Vec(walk_length=walkLength, num_walks=numOfWalks, window_length=windowLength, batch_walks=batchWalks, p=pForEmbedding, q=qForEmbedding)
        model.fit(A)
        center_vec = model.transform(dim=d) #d number of columns; 1 row=1 node
        #context_vec = model.out_vec #d number of columns; 1 row=1 node
        Coord = {} #initialize the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space
        nodeID = 0
        for nodeName in listOfNodes:
            Coord[nodeName] = center_vec[nodeID,:]
            nodeID = nodeID+1

        #create the graph with embedding-based link weights
        G_embWeighted = nx.Graph()
        G_embWeighted.add_nodes_from(listOfNodes) #keep the node order of the original graph
        for (i,j) in G.edges(): #(i,j)=a tuple, which denotes the edge between node u and v
            # assign a weight to the i-j edge:
            w_notExp = 1 + (np.inner(Coord[i],Coord[j]) / (np.linalg.norm(Coord[i])*np.linalg.norm(Coord[j])))
            if w_notExp<0: #a numerical error has occurred
                w_notExp = 0.0
            if w_notExp>2: #a numerical error has occurred
                w_notExp = 2.0
            w = math.exp(q*w_notExp)-1 #weight=exponentialized form of a cosine proximity-like measure
            G_embWeighted.add_edge(i,j,weight=w)

        numOfDoneIters = numOfDoneIters+1
    
    return Coord
