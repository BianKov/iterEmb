#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np 
import graph_tool.all as gt
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import os



#creating a directory for all of the files
def createDirectory(directoryName):
    filePath=os.getcwd()+"/"+directoryName+"/"
    directory=os.path.dirname(filePath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)



def generate_network(Cave, mixing_rate, N, q):

    memberships = np.sort(np.arange(N) % q)

    q = int(np.max(memberships) + 1)
    N = len(memberships)
    U = sparse.csr_matrix((np.ones(N), (np.arange(N), memberships)), shape=(N, q))

    Cout = np.maximum(1, mixing_rate * Cave)
    Cin = q * Cave - (q - 1) * Cout
    pout = Cout / N
    pin = Cin / N

    Nk = np.array(U.sum(axis=0)).reshape(-1)

    P = np.ones((q, q)) * pout + np.eye(q) * (pin - pout)
    probs = np.diag(Nk) @ P @ np.diag(Nk)
    gt_params = {
        "b": memberships,
        "probs": probs,
        "micro_degs": False,
        "in_degs": np.ones_like(memberships) * Cave,
        "out_degs": np.ones_like(memberships) * Cave,
    }

    # Generate the network until the degree sequence
    # satisfied the thresholds
    while True:
        g = gt.generate_sbm(**gt_params)

        A = gt.adjacency(g).T

        A.data = np.ones_like(A.data)
        # check if the graph is connected
        if connected_components(A)[0] == 1:
            break
        break
    return A, memberships
	





# Parameters
cave, n, K = 20, 1000, 10 # Average degree, number of nodes, number of communities
numOfGraphs = 10
mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Mixing rate
dirNameDict = {0.1:"0.1", 0.2:"0.2", 0.3:"0.3", 0.4:"0.4", 0.5:"0.5", 0.6:"0.6", 0.7:"0.7", 0.8:"0.8", 0.9:"0.9"}

net_list = {}
for mu in mu_list: 
    print('mu='+str(mu))
    createDirectory("SBM_"+dirNameDict[mu])
    for graphID in range(numOfGraphs):
        createDirectory("SBM_"+dirNameDict[mu]+"/graph"+str(graphID))

        A, memberships = generate_network(cave, mu, n, K) #A=sparse adjacency matrix

        #save the edge list
        G = nx.from_numpy_matrix(A.todense())
        nx.write_edgelist(G,os.getcwd()+"/SBM_"+dirNameDict[mu]+"/graph"+str(graphID)+"/SBM_edges.txt",delimiter='\t',data=False)

        #save the communities
        attributeDict = {} #key=groupID, value=list of node IDs in the given group
        for groupID in set(memberships):
            attributeDict[groupID] = []
        for nodeID in range(n):
            attributeDict[memberships[nodeID]].append(nodeID)
        fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/graph"+str(graphID)+"/blocks.txt","w")
        for groupID in attributeDict.keys():
            for nodeID in attributeDict[groupID]:
                fileHandler.write(str(nodeID)+"\t"+str(groupID)+"\n")
        fileHandler.close()

