# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-04-05 14:48:22
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-05 15:17:28
import networkx as nx
import numpy as np
from scipy import sparse
import graphvec
import unittest


def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


class TestingEmbeddingMethods(unittest.TestCase):
    def setUp(self):
        G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(G)
        self.labels = np.unique(
            [d[1]["club"] for d in G.nodes(data=True)], return_inverse=True
        )[1]

    def test_embedding(self):

        for model in inheritors(graphvec.NodeEmbeddings):
            instance = model()
            instance.fit(self.A)
            instance.transform(dim=8)
            print(model)
