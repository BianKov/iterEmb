#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess as sp


#A benchmark nevű futtatót le kell másolni oda, ahol ez a python fájl van!!!


#creating a directory for all of the files
def createDirectory(directoryName):
    filePath=os.getcwd()+"/"+directoryName+"/"
    directory=os.path.dirname(filePath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)


#execute the C++ code of the LFR model, which creates and saves networks
def generate_network(N, k, maxk, mu, t1, t2, minc, maxc):
    args = [os.getcwd()+"/benchmark", "-N", str(N), "-k", str(k), "-maxk", str(maxk), "-mu" , str(mu), "-t1" , str(t1), "-t2" , str(t2), "-minc" , str(minc), "-maxc" , str(maxc)]
    proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE, shell=False)
    print(args)
    (out, err) = proc.communicate(input = None)
    print("Output from the program: ", out,err)
    if (err != None and len(err) > 0):
        print("Error while executing the program! Error code: ", proc.returncode, "Output: ", err)



# Parameters
N = 1000 #number of nodes
k = 20 #average degree
maxk = 50 #maximum degree
mu = 0.9 #mixing parameter
t1 = 2.0 #minus exponent for the degree sequence
t2 = 3.0 #minus exponent for the community size distribution
minc = 10 #minimum for the community sizes
maxc = 100 #maximum for the community sizes

graphID = 1

dirNameDict = {0.1:"0.1", 0.2:"0.2", 0.3:"0.3", 0.4:"0.4", 0.5:"0.5", 0.6:"0.6", 0.7:"0.7", 0.8:"0.8", 0.9:"0.9"}
createDirectory("LFR_"+dirNameDict[mu])
createDirectory("LFR_"+dirNameDict[mu]+"/graph"+str(graphID))
generate_network(N, k, maxk, mu, t1, t2, minc, maxc) #a python file helyére kerül a háló, nem a kívánt mappába!!!
