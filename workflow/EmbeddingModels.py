# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-04 07:31:04
import embcom

try:
    from workflow.iterativeEmbedding import *
except:
    from iterativeEmbedding import *

embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)


# @embedding_model
# def node2vec(network, dim, window_length=10, num_walks=40):
#    model = embcom.embeddings.Node2Vec(window_length=window_length, num_walks=num_walks)
#    model.fit(network)
#    return model.transform(dim=dim)
#
#
# @embedding_model
# def deepwalk(network, dim, window_length=10, num_walks=40):
#    model = embcom.embeddings.DeepWalk(window_length=window_length, num_walks=num_walks)
#    model.fit(network)
#    return model.transform(dim=dim)
from functools import partial


def node2vecIter(network, dim, numOfIters, window_length=10, num_walks=80):
    return iteratedExpNode2vec(
        A=network,
        numOfIters=numOfIters,
        dim=dim,
        windowLength=window_length,
        numWalks=num_walks,
    )


for numOfIters in range(0, 22, 2):
    embedding_models[f"node2vecIter{numOfIters}"] = partial(
        node2vecIter, numOfIters=numOfIters
    )


@embedding_model
def leigenmap(network, dim):
    model = embcom.embeddings.LaplacianEigenMap()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def modspec(network, dim):
    model = embcom.embeddings.ModularitySpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


# @embedding_model
# def nonbacktracking(network, dim):
#    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
#    model.fit(network)
#    return model.transform(dim=dim)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
