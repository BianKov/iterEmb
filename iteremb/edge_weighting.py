# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-03 22:01:20
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-08 12:27:26
import numpy as np
from scipy import sparse
from scipy import stats

models = {}
weighting_model = lambda f: models.setdefault(f.__name__, f)

preprocessing_funcs = {}
preprocessing_function = lambda f: preprocessing_funcs.setdefault(f.__name__, f)


def find_edges(A):
    B = sparse.coo_matrix(A)
    return B.row, B.col, B.data


@weighting_model
def cosine_similarity(A, emb, **params):
    """
    Calculate the cosine similarity between embeddings and return a sparse matrix.

    Parameters
    ----------
    A : csr_matrix
        The adjacency matrix of a graph.
    emb : ndarray
        The node embedding matrix with shape (num_nodes, dim).
    **params : dict
        Additional parameters for future customization.

    Returns
    -------
    csr_matrix
        A sparse matrix representing the cosine similarity between embeddings.
    """
    nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
    src, trg, _ = find_edges(A)
    w = np.array(np.sum(nemb[src] * nemb[trg], axis=1)).reshape(-1)
    w = w + 1
    w = np.clip(w, -0.0, 2.0)
    
    A = sparse.csr_matrix((w+1, (src, trg)), shape=A.shape)
    A.data-=1
    return A


@weighting_model
def cosine_distance(A, emb, **params):
    """
    Computes the cosine distance between each pair of items in sparse matrix A using embeddings.

    Parameters:
    -----------
    A : sparse matrix
        Sparse matrix representing the user-item interactions.
    emb : array-like of shape (n_items, n_dimensions)
        Embedding vectors for each item.
    **params : dict
        Dictionary containing additional optional parameters.

    Returns:
    --------
    distance_matrix: csr_matrix
        Sparse cosine distance matrix between each pair of items in A.

    """
    nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
    src, trg, _ = find_edges(A)
    w = 1.0 - np.array(np.sum(nemb[src] * nemb[trg], axis=1)).reshape(-1)
    w = np.clip(w, 0.0, 2.0)
    A = sparse.csr_matrix((w+1, (src, trg)), shape=A.shape)
    A.data-=1
    return A



@weighting_model
def exp_cosine_similarity(A, emb, q=1):
    """
    Computes the exponential cosine similarity between each pair of items in sparse matrix A using embeddings.

    Parameters:
    -----------
    A : sparse matrix
        Sparse matrix representing the user-item interactions.
    emb : array-like of shape (n_items, n_dimensions)
        Embedding vectors for each item.
    q : float
        Exponential decay parameter.

    Returns:
    --------
    similarity_matrix : csr_matrix
        Sparse exponential cosine similarity matrix between each pair of items in A.
    """
    nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
    src, trg, _ = find_edges(A)
    w = np.array(np.sum(nemb[src] * nemb[trg], axis=1)).reshape(-1) - 1.0
    w = np.clip(w, -2.0, 0.0)

    w = np.exp(q * w)
    
    A = sparse.csr_matrix((w+1, (src, trg)), shape=A.shape)
    A.data-=1
    return A


@preprocessing_function
def q_factor_determination(A, emb_params, edge_weighting_params):
    """
    Preprocessing function for the iterative embedding algorithm.

    Parameters:
    -----------
    A : sparse matrix
        Sparse matrix representing the user-item interactions.
    emb_params : dict
        Dictionary containing the parameters for the embedding model.
    edge_weighting_params : dict
        Dictionary containing the parameters for the edge weighting model.

    Returns:
    --------
    emb_params : dict
        Dictionary containing the updated parameters for the embedding model.
    edge_weighting_params : dict
        Dictionary containing the updated parameters for the edge weighting model.
    """

    # Determine the q factor
    deg = A.sum(axis=1).A1
    deg_mode = stats.mode(deg)[0]
    if len(deg_mode) > 1:
        deg_mode = deg_mode.min()

    deg_ave = deg.mean()
    q = 10 * (deg_ave / deg_mode)
    edge_weighting_params["q"] = q
    return emb_params, edge_weighting_params
