# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-25 16:46:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-08 12:29:28
from .node2vec import Node2Vec
from .deepwalk import DeepWalk
from .line import LINE
from .graph import Graph

__all__ = ["Graph", "Node2Vec", "DeepWalk", "LINE", "DepWalk"]
