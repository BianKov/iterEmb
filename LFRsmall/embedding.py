#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import fastnode2vec #the fast version of node2vec
import scipy.stats #for testing outliers during the dimension selection of TREXPIC with the Modified Thompson Tau test



#A function for loading the undirected edge list of the network to be embedded. Edge weights are disregarded, self-loops are removed, multi-edges are converted to single edges, and only the largest connected component is returned as a NetworkX Graph.
#path is a string with the path of the text file containing the edge list to be loaded
    #In the edge list each line has to correspond to one connected node pair. Columns after the first two (that contain the node identifiers) will be disregarded, each edge will have weight 1.
#skipRows is the number of lines to be skipped at the beginning of the text file containing the edge list; the default is 0
#delimiter is the string used to separate the columns in the text file to be loaded; the default is "\t"
#Example for function call:
#   G=embedding.loadGraph(os.getcwd()+"/"+directoryName+"/edgeList.txt",1,"\t")
def loadGraph(path,skipRows=0,delimiter="\t"):
    edgeList = [] #initialize the list of the (source node identifier,target node identifier) edge tuples
    fileHandler = open(path,"r")
    for l in range(skipRows):
        line = fileHandler.readline()
    while True:
        line = fileHandler.readline() #get the next line from file
        if not line: #line is empty (end of the file)
            break;
        listOfWords = line.split(delimiter)
        sourceNodeID = listOfWords[0] #string from the first column as the identifier of the source node
        if listOfWords[1][-1]=="\n": #the second column is the last in the currently loaded line
            targetNodeID = listOfWords[1][:-1] #string from the second column without "\n" as the identifier of the target node
        else: #there are more than two columns in the currently loaded line
            targetNodeID = listOfWords[1] #string from the second column as the identifier of the target node
        if sourceNodeID != targetNodeID: #the self-loops are disregarded
            edgeList.append((sourceNodeID,targetNodeID))
    fileHandler.close()

    G_total = nx.Graph()
    G_total.add_edges_from(edgeList) #multi-edges are automatically converted to single edges
    #extract the largest connected component:
    G=max([G_total.subgraph(comp).copy() for comp in nx.connected_components(G_total)],key=len) #.copy(): create a subgraph with its own copy of the edge/node attributes -> changes to attributes in the subgraph are NOT reflected in the original graph; without copy the subgraph is a frozen graph for which edges can not be added or removed

    return G



#At https://www.nature.com/articles/s41467-017-01825-5 different pre-weighting strategies are described for undirected networks that facilitated the estimation of the spatial arrangement of the nodes in dimension reduction techniques that are built on the shortest path lengths. This function performs these link weighting procedures. Larger weight corresponds to less similarity.
#G is an unweighted NetworkX Graph to be pre-weighted. Note that the link weights in G do not change from 1, the function returns a weighted copy of G.
#weightingType is a string corresponding to the name of the pre-weighting rule to be used. Can be set to 'RA1', 'RA2', 'RA3', 'RA4' or 'EBC'.
#Example for function call:
#   G_preweighted = embedding.preWeighting(G,'RA1')
def preWeighting(G,weightingType):
    G_weighted = nx.Graph()
    G_weighted.add_nodes_from(G.nodes) #keep the node order of the original graph
    if weightingType=='RA1':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            # set=unordered collection with no duplicate elements,
            # set operations (union, intersect, complement) can be executed (see RAtype==2)
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) + G.degree(j) + G.degree(i) * G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA2':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j

            # ei=the external degree of the node i with respect to node j,
            # i.e. the number of links from i to neither j nor the common neighbors with j,
            # i.e. the number of i's neighbors without node j and the common neighbors with j
            neighborSet_i = {n for n in G[i]}  # set with the indices of the neighbors of node i
            # G[i]=adjacency dictionary of node i -> iterating over its keys(=neighboring node indices)
            ei = len(neighborSet_i - {j} - CNset)

            # ej=the external degree of the node j with respect to node i,
            # i.e. the number of links from j to neither i nor the common neighbors with i,
            # i.e. the number of j's neighbors without node i and the common neighbors with i
            neighborSet_j = {n for n in G[j]}  # set with the indices of the neighbors of node j
            # G[j]=adjacency dictionary of node j -> iterating over its keys(=neighboring node indices)
            ej = len(neighborSet_j - {i} - CNset)

            # assign a weight to the i-j edge:
            w = (1 + ei + ej + ei * ej) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA3':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) + G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA4':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) * G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='EBC': #use the edge betweenness centrality
        #create a dictionary, which contains all the shortest paths between all node pairs
            #shortestPathsDict[(source,target)] is the list of shortest paths from node with ID source to node with ID target
                #a path is a list of nodes following each other in the path
            #the graph to be embedded should be connected (all nodes can be reached from any node)
        shortestPathsDict = {}
        nodeList=list(G.nodes)
        N = len(nodeList)
        for u in range(N-1): #u=0,1,...,N-2
            for v in range(u+1,N): #v=u+1,...,N-1
            #these loops are sufficient only if graph G is undirected (the same number of paths lead from the uth node to the vth node and from the vth node to the uth node) and does not contain any self-loops
                node_u = nodeList[u]
                node_v = nodeList[v]
                shortestPathsDict[(node_u,node_v)]=[p for p in nx.all_shortest_paths(G,source=node_u,target=node_v,weight=None)] #weight=None: every edge has weight/distance/cost 1 (the possible current weights are disregarded)

        #weight all the edges
        for (i,j) in G.edges():
            w=0 #initialize the weight of the i-j edge
            for u in range(N-1):
                for v in range(u+1,N):
                    shortestPathsBetween_uv = shortestPathsDict[(nodeList[u],nodeList[v])] #list of shortest paths between the uth node and the vth node
                    sigma = len(shortestPathsBetween_uv) #the total number of shortest paths between the uth node and the vth node
                    #count those paths between node u and node v which contains the i-j edge
                    sigma_ij = 0
                    for q in shortestPathsBetween_uv: #q=list of nodes following each other in a path between the uth node and the vth node
                        if i in q and j in q: #since q is a shortest path, therefore in this case abs(q.index(i)-q.index(j))==1 is already granted
                            sigma_ij = sigma_ij+1
                    w=w+(sigma_ij/sigma)
            G_weighted.add_edge(i,j,weight=w) #assign a weight to the i-j edge
    else:
        print('False parameter: weightingType\n')
    return G_weighted



#This function chooses the number of embedding dimensions for TREXPIC based on the spectral gaps of the graph's normalized Laplacian.
#G is an unweighted NetworkX Graph to be embedded
#path is a string with the path of the figure to be saved
#1>alpha>0 is the significance level: at smaller values of alpha it is harder to become an outlier (less gaps will be labeled as outliers)
#weightingType is a string corresponding to the name of the pre-weighting rule to be used. Can be set to 'RA1', 'RA2', 'RA3', 'RA4' or 'EBC'.
#Example for function call:
#   embeddingDimForTREXPIC = embedding.choosingDim_TREXPIC(G,os.getcwd()+'/'+directoryName+'/automaticDimensionSelectionForTREXPIC_graph0.png')
def choosingDim_TREXPIC(G,path,alpha=math.pow(10,-6),weightingType='RA1'):
    #pre-weighting
    G_weighted = preWeighting(G,weightingType) #larger weight corresponds to less similarity
    #convert the weights: larger weight must correspond to higher similarity
    scalingFactor = 0
    for (i, j) in G_weighted.edges(): #(i,j)=a tuple, which denotes the edge between node u and v
        scalingFactor = scalingFactor+G_weighted[i][j]['weight']
    scalingFactor = math.pow(scalingFactor/len(G_weighted.edges()),2)
    for (i, j) in G_weighted.edges(): #(i,j)=a tuple, which denotes the edge between node u and v
        G_weighted[i][j]['weight'] = math.exp(-math.pow(G_weighted[i][j]['weight'],2)/scalingFactor)

    #examine the eigenvalues
    N = len(G_weighted) #number of nodes
    dMax = int(N/2) #theoretical maximum: N-1
    #Eigenvalues of the normalized graph Laplacian (the first one is 0; the smaller the more important):
    eigenValsOfL = nx.normalized_laplacian_spectrum(G_weighted,weight='weight') #increasing order of N number of eigenvalues
    gaps = np.zeros(dMax-1) #only N-1 number of eigenvalues are relevant (the one that is always 0 is not)
    for i in range(2,dMax+1):
        gaps[i-2] = eigenValsOfL[i]-eigenValsOfL[i-1]
    maxGapSize = np.amax(gaps)
    meanGapSize = np.mean(gaps)
    stdOfGapSize = np.std(gaps,ddof=1) #sample standard deviation
    numOfGaps = len(gaps)
    critical_t = scipy.stats.t.ppf(q=1-alpha/2,df=numOfGaps-2)
    rejectionRegion = (critical_t*(numOfGaps-1))/(math.sqrt(numOfGaps)*math.sqrt(numOfGaps-2+math.pow(critical_t,2)))
    delta = (maxGapSize-meanGapSize)/stdOfGapSize
    if delta>rejectionRegion: #the value of the largest gap is an outlier
        bestDim = np.argmax(gaps)+1
    else:
        bestDim = dMax
        print('WARNING: Maybe there is no significant community structure in the given network!')
    if bestDim==1:
        d=2
    else:
        d=bestDim
    print('The chosen number of dimensions is '+str(d))
    
    #plot the spectrum
    dMaxPlot = dMax-1
    fig = plt.figure(figsize=(10,16))
    #plot the eigenvalues
    ax = plt.subplot(2,1,1)
    plt.plot(range(1,dMaxPlot+2),eigenValsOfL[:dMaxPlot+1],'g-',marker='.') #first eigenvalue=0 -> would need the eigenvalues from the second to the (d+1)th one for an embedding...
    plt.xlabel('Index of eigenvalue')
    plt.ylabel('Eigenvalue of the normalized Laplacian')
    plt.xlim((0.0,dMaxPlot+2.0))
    plt.title('Best number of dimensions: '+str(bestDim))
    #plot the gaps between the adjacent eigenvalues
    ax = plt.subplot(2,1,2)
    plt.plot(range(1,dMaxPlot+1),gaps[:dMaxPlot],'b-',marker='.') #first eigenvalue=0 -> would need the eigenvalues from the second to the (d+1)th one for an embedding...
    plt.xlabel('Number of dimensions')
    plt.ylabel('Gaps between the adjacent Eigenvalues of the normalized Laplacian')
    plt.xlim((0.0,dMaxPlot+2.0))
    plt.subplots_adjust(hspace=0.15) #set the distance between the figures (wspace=vizszintes)
    fig.savefig(path,bbox_inches="tight",dpi=300)
    plt.close(fig)

    return d



#This function calculates the hyperbolic distance h between two nodes of the network. The NumPy vectors coords1 and coords2 both contain d Cartesian coordinates, which describe the position of the given nodes in the native representation of the d-dimensional hyperbolic space of curvature K<0.
def hypDist(coords1,coords2,K=-1):
    zeta = math.sqrt(-K)
    r1 = np.linalg.norm(coords1) #the radial coordinate of node 2 in the native representation, i.e. the Euclidean length of the vector coords2
    r2 = np.linalg.norm(coords2) #the radial coordinate of node 2 in the native representation, i.e. the Euclidean length of the vector coords2
    if r1==0:
        h = r2
    elif r2==0:
        h = r1
    else:
        cos_angle = np.inner(coords1,coords2)/(r1*r2) #cosine of the angular distance between the two nodes
        if cos_angle==1: #the vectors coords1 and coords2 point in the same direction; in this case the hyperbolic distance between the two nodes is acosh(cosh(r1-r2))=|r1-r2|
            h = math.fabs(r1-r2)
        elif cos_angle==-1: #the vectors coords1 and coords2 point in the opposite direction
            h = r1+r2
        else:
            argument_of_acosh = math.cosh(zeta*r1)*math.cosh(zeta*r2)-math.sinh(zeta*r1)*math.sinh(zeta*r2)*cos_angle
            if argument_of_acosh<1: #a rounding error occurred, because the hyperbolic distance h is close to zero
                print("The argument of acosh is "+str(argument_of_acosh)+", less than 1.")
                h = 0 #acosh(1)=0
            else:
                h = math.acosh(argument_of_acosh)/zeta
    return h



#A function for creating a weighted version of a graph based on its d-dimensional hyperbolic embedding created by TREXPIC.
#G is the (weighted) NetworkX Graph to be embedded
#d is the number of dimensions of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal matrix elements towards 0)
#K<0 is the curvature of the hyperbolic space
#The function returns the weighted NetworkX Graph G_embWeighted
#Example for function call:
#   G_w=embedding.TREXPIC(G,d)
def TREXPIC(G,d,q=None,K=-1):
    N = len(G) #the number of nodes in graph G
    if N < d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of nodes the number of nodes-1, i.e. '+str(N-1)+'.\n\n')

    zeta = math.sqrt(-K)

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.

    #create the matrix to be reduced
    shortestPathLengthsDict = dict(nx.shortest_path_length(G,weight='weight'))
        #shortestPathLengthDict[source][target]=length of the shortest path from the source node to the target node

    if q==None: #use the default setting of the multiplying factor
        listOfAllSPLs = []
        for sour in shortestPathLengthsDict.keys():
            for targ in shortestPathLengthsDict[sour].keys():
                listOfAllSPLs.append(shortestPathLengthsDict[sour][targ])
        maxSPL = max(listOfAllSPLs)
        qmin = math.log(1.0/0.9999)*maxSPL
        qmax = math.log(10)*maxSPL
        q = math.exp((math.log(qmin)+math.log(qmax))/2)
        print('The default q multiplying factor is '+str(q)+'.')

    L = np.zeros((N,N)) #a matrix indicating the node-node distances; L[i][j] is the Lorentz product calculated from the length of the shortest path from the ith node to the jth node, where the nodes are ordered according to G.nodes
    rowID = 0
    for s in G: #iterate over the nodes as sources
        colID = 0
        for t in G: #iterate over all the nodes as targets
            if s==t:
                matrixElement = 0.0
            else:
                try:
                    matrixElement = math.exp(-q/shortestPathLengthsDict[s][t])
                except KeyError: #there is no path from node s to node t -> set the distance to the possible highest value (SPL=infinity -> exp(-q/SPL)=1)
                    matrixElement = 1.0
                except ZeroDivisionError: #two different nodes are connected with a link/links of weight 0
                    matrixElement = 0.0
            L[rowID,colID] = math.cosh(zeta*matrixElement)
            colID = colID+1
        rowID = rowID+1

    #dimension reduction
    if d == N-1:
        U,S,VT=np.linalg.svd(L) #find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
             #note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else: #d<N-1: only the first d+1 singular values (and the corresponding singular vectors) are retained
        U,S,VT = svds(L,d+1,solver='arpack') #the singular values are ordered from the smallest to the largest in S (increasing order)
        #reverse the order of the singular values to obtain a decreasing order:
        S = S[::-1]
        U = U[:,::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues<d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of positive singular values of the proximity matrix-1, i.e. '+str(numOfPositiveSingularValues-1)+'.\n\n')
    Ssqrt = np.sqrt(S[1:]) #d number of singular values are used for determining the directions of the position vectors: from the second to the d+1th one

    #create the dictionary of node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node in the native ball
    Coord = {} #initialize a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the native representation of the d-dimensional hyperbolic space
    nodeIndex = 0
    numOfErrors = 0
    for nodeName in G:
        #calculate the position of the given node
        Uarray = U[nodeIndex,:]
        firstCoordOnHyperboloid = math.fabs(math.sqrt(S[0])*Uarray[0]) #to select the upper sheet of the two-sheet hyperboloid, firstCoordOnHyperboloid has to be positive
            #we could also use: Varray = VT[:,nodeIndex] and then firstCoordOnHyperboloid = math.fabs(math.sqrt(S[0])*Varray[0])
        if firstCoordOnHyperboloid<1: #a numerical error has occurred
            r_native = 0
            numOfErrors = numOfErrors+1
        else:
            r_native = (1/zeta)*math.acosh(firstCoordOnHyperboloid)
        directionArray = (-1)*np.multiply(Uarray[1:],Ssqrt) #the jth element is the jth coordinate of the node named nodeName in the reduced space; we use the additive inverse, because we want to identify the elements of the reduced matrix as Lorentz products of the position vectors
            #we could also use: directionArray = np.multiply(Varray[1:],Ssqrt)
        originalNorm = np.linalg.norm(directionArray)
        Coord[nodeName] = r_native*directionArray/originalNorm #the Cartesian coordinates of the node named nodeName in the d-dimensional hyperbolic space
        nodeIndex = nodeIndex+1
    if numOfErrors>0:
        print('TREXPIC placed '+str(numOfErrors)+' nodes at an invalid position on the hyperboloid. The native radial coordinate of these nodes is set to 0. Consider changing the q parameter of the embedding!')
    
    #create the graph with embedding-based link weights
    G_embWeighted = nx.Graph()
    G_embWeighted.add_nodes_from(G.nodes) #keep the node order of the original graph
    for (i,j) in G.edges(): #(i,j)=a tuple, which denotes the edge between node u and v
        # assign a weight to the i-j edge:
        w = hypDist(Coord[i],Coord[j],K)
        G_embWeighted.add_edge(i,j,weight=w)
    
    return G_embWeighted



#A function for creating a weighted version of a graph based on its angular node arrangement in a d-dimensional Euclidean embedding created by node2vec.
#G is the (weighted) NetworkX Graph to be embedded
#d is the number of dimensions of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the weight function: link weight=exp(q*(1+cos(angular distance)))-1
#d, walkLength, numOfWalks, windowLength, batchWalks, pForEmbedding, qForEmbedding: the usual parameters of the node2vec methods
#The function returns the weighted NetworkX Graph G_embWeighted
#Example for function call:
#   G_w=embedding.node2vec(G,q)
def node2vec(G,q,d=128,walkLength=10,numOfWalks=80,windowLength=10,batchWalks=10000,pForEmbedding=1.0,qForEmbedding=1.0):
    #create the embedding
    listOfNodes = list(G.nodes())
    A = nx.adjacency_matrix(G,nodelist=listOfNodes,weight='weight') #adjacency matrix as a SciPy Compressed Sparse Row matrix
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
        w = math.exp(q*(1 + (np.inner(Coord[i],Coord[j]) / (np.linalg.norm(Coord[i])*np.linalg.norm(Coord[j])))))-1
        G_embWeighted.add_edge(i,j,weight=w)
    
    return G_embWeighted
