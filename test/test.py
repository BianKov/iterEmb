# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-02 16:30:10
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-02 16:31:30
import unittest

import networkx as nx
import numpy as np
from scipy import sparse


class TestIterativeEmbedding(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)
        self.A.data = self.A.data * 0 +1

    def test_iterative_embedding(self):


if __name__ == "__main__":
    unittest.main()