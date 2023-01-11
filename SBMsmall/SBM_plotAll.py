#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from matplotlib import cm



mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Mixing rate
dirNameDict = {0.1:"0.1", 0.2:"0.2", 0.3:"0.3", 0.4:"0.4", 0.5:"0.5", 0.6:"0.6", 0.7:"0.7", 0.8:"0.8", 0.9:"0.9"}

AMI_node2vec = {}
AMI_TREXPIC = {}
AMI_Louvain = {}
AMI_alabprop = {}
AMI_Infomap = {}
Q_real = {}
Q_detected = {}
Q_Louvain = {}
Q_alabprop = {}
Q_Infomap = {}
for mu in mu_list: 
    print('mu='+str(mu))
    for graphID in [1]: #plot the results only for the graph of ID 1
#    fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/TREXPICresults.txt", "r")
#    listOfLines=fileHandler.readlines()
#    fileHandler.close()
#    AMI_Louvain[mu] = float(listOfLines[16])
#    AMI_alabprop[mu] = float(listOfLines[22])
#    AMI_Infomap[mu] = float(listOfLines[28])
    
        fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/graph1/node2vecResults.txt", "r")
        listOfLines=fileHandler.readlines()
        fileHandler.close()
        AMI_node2vec[mu] = float(listOfLines[3])
    
        fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/graph1/TREXPICresults.txt", "r")
        listOfLines=fileHandler.readlines()
        fileHandler.close()
        AMI_TREXPIC[mu] = float(listOfLines[3])

        fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/graph1/LouvainResults.txt", "r")
        listOfLines=fileHandler.readlines()
        fileHandler.close()
        AMI_Louvain[mu] = float(listOfLines[3])
    
        fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/graph1/InfomapResults.txt", "r")
        listOfLines=fileHandler.readlines()
        fileHandler.close()
        AMI_Infomap[mu] = float(listOfLines[3])

        fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/graph1/alabpropResults.txt", "r")
        listOfLines=fileHandler.readlines()
        fileHandler.close()
        AMI_alabprop[mu] = float(listOfLines[3])


#    fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/node2vecResults_Birch.txt", "r")
#    listOfLines=fileHandler.readlines()
#    fileHandler.close()
#    AMI_node2vec_Birch[mu] = float(listOfLines[5])
    
#    fileHandler=open(os.getcwd()+"/SBM_"+dirNameDict[mu]+"/node2vecResults_firstEmb_Birch.txt", "r")
#    listOfLines=fileHandler.readlines()
#    fileHandler.close()
#    AMI_node2vec_firstEmb_Birch[mu] = float(listOfLines[5])


#create figure from ONE given graph
fig = plt.figure(figsize=(10,8))  #(width,height)
plt.plot(mu_list,[AMI_node2vec[mu] for mu in mu_list],'mo',ls='-',label='iterated node2vec')
plt.plot(mu_list,[AMI_TREXPIC[mu] for mu in mu_list],'h',color='pink',ls='-',label='iterated TREXPIC')
plt.plot(mu_list,[AMI_Louvain[mu] for mu in mu_list],'r^',ls='-.',label='Louvain')
plt.plot(mu_list,[AMI_alabprop[mu] for mu in mu_list],'gs',ls='--',label='async. label prop.') #asynchronous label propagation
plt.plot(mu_list,[AMI_Infomap[mu] for mu in mu_list],'b*',ls=':',label='Infomap')
plt.xlabel('mixing parameter',fontsize=20)
plt.ylabel('adjusted mutual information',fontsize=20)
plt.xlim((0.09,0.91))
plt.ylim((-0.01,1.01))
plt.legend(loc=1,fontsize=12)
plt.title('SBM networks, N=1000, avDeg=20, numOfComms=10')
fig.savefig(os.getcwd()+'/smallSBMresults_graph1.png',bbox_inches="tight",dpi=300)
plt.close(fig)
