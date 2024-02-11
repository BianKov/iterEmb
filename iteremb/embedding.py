# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-25 16:46:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-07 11:46:01
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from iteremb import utils
import numpy as np
import networkx as nx
import math
from scipy import sparse
from iteremb import fastnode2vec

models = {}
embedding_model = lambda f: models.setdefault(f.__name__, f)


@embedding_model
def TREXPIC(G, d, q=None, K=-1.0, verbose=False):
    """
    Create a weighted version of a graph based on its d-dimensional hyperbolic embedding generated by the TREXPIC method.

    Parameters
    ----------
    G : nx.Graph or scipy.sparse_csr_matrix
        The (weighted) NetworkX Graph or the adjacency matrix in form of the scipy sparse format.
        If link weights are inputted, these must mean distances,
        i.e. larger value:=less similarity or weaker connection!

    d : int
        The number of dimensions of the space to which the network will be embedded.

    q : Optional[float], default=None
        The multiplying factor in the exponent in the matrix to be reduced
        (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal
        matrix elements towards 0). If None, its default setting will be used.

    K : float, default=-1.0
        The curvature of the hyperbolic space.

    Returns
    -------
    emb:  numpy.ndarray of shape (n_nodes, dim)

    Example
    -------
    >>> emb = embedding.TREXPIC(G, d)
    """
    A = utils.to_scipy_matrix(G, return_node_labels=False)
    N = A.shape[0]  # the number of nodes in graph G
    if N < d + 1:
        raise ValueError(
            f"\n\nERROR: The number d of embedding dimensions in the function embedding. TREXPIC can not be larger than the number of nodes-1, i.e. {N-1}\n\n"
        )
    zeta = math.sqrt(-K)

    # create the matrix to be reduced
    shortestPathLengthMatrix = sparse.csgraph.shortest_path(A, directed=False)
    if q == None:  # use the default setting of the multiplying factor
        maxSPL = np.ma.masked_invalid(shortestPathLengthMatrix).max()
        qmin = math.log(1.0 / 0.9999) * maxSPL
        qmax = math.log(10) * maxSPL
        q = math.exp((math.log(qmin) + math.log(qmax)) / 2)
        if verbose:
            print("The default q multiplying factor is " + str(q) + ".")

    # create the matrix of expected Lorentz products
    with np.errstate(divide="ignore"):
        L = np.exp(-q / shortestPathLengthMatrix)
    L = np.cosh(zeta * L)

    # dimension reduction
    if d == N - 1:
        U, S, VT = np.linalg.svd(
            L
        )  # find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
        # note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else:  # d<N-1: only the first d+1 largest singular values (and the corresponding singular vectors) are retained
        U, S, VT = sparse.linalg.svds(
            L, d + 1, which="LM", solver="arpack"
        )  # the singular values are ordered from the smallest to the largest in S (increasing order)
        # reverse the order of the singular values to obtain a decreasing order, and reverse the order of singular vectors accordingly:
        S = S[::-1]
        U = U[:, ::-1]
        VT = VT[::-1]

    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues < d + 1:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of positive singular values of the Lorentz matrix-1, i.e. "
            + str(numOfPositiveSingularValues - 1)
            + ".\n\n"
        )
    Ssqrt = np.sqrt(
        S[1:]
    )  # d number of singular values are used for determining the directions of the position vectors: from the second to the d+1th one

    numOfErrors = 0
    firstCoordOnHyperboloid = np.abs(
        np.sqrt(S[0]) * U[:, 0]
    )  # to select the upper sheet of the two-sheet hyperboloid, firstCoordOnHyperboloid has to be positive
    is_numerical_error = firstCoordOnHyperboloid < 1
    numOfErrors = np.sum(is_numerical_error)
    r_negative = (1 / zeta) * np.arccosh(np.maximum(firstCoordOnHyperboloid, 1))
    r_negative[is_numerical_error] = 0

    # the jth element is the jth coordinate of the node named nodeName in the reduced space
    directionArray = np.einsum("ij,j->ij", U[:, 1:], Ssqrt)
    originalNorm = np.array(np.linalg.norm(directionArray, axis=1)).reshape(-1)
    Coord = np.einsum("ij,i->ij", directionArray, r_negative / originalNorm)

    if numOfErrors > 0:
        raise ValueError(
            "TREXPIC placed "
            + str(numOfErrors)
            + " nodes at an invalid position on the hyperboloid. The native radial coordinate of these nodes is set to 0. Consider changing the q parameter of the embedding!"
        )
    return Coord


@embedding_model
def expISO(G, d, q=None, verbose=False):
    """
    Create a weighted version of a graph based on its d-dimensional Euclidean embedding generated by the exponentialized version of Isomap.

    Parameters
    ----------
    G : nx.Graph or scipy.sparse_csr_matrix
        The (weighted) NetworkX Graph or the adjacency matrix in form of the scipy sparse format.
        If link weights are inputted, these must mean distances,
        i.e. larger value:=less similarity or weaker connection!

    d : int
        The number of dimensions of the space to which the network will be embedded.

    q : float, optional
        Multiplying factor in the exponent in the matrix to be reduced (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal matrix elements towards 0). If None, its default setting will be used.

    Returns
    -------
    emb:  numpy.ndarray of shape (n_nodes, dim)

    Example
    -------
    emb = embedding.expISO(G, d)
    """
    A = utils.to_scipy_matrix(G, return_node_labels=False)
    N = A.shape[0]  # the number of nodes in graph G
    if N < d:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.ISO can not be larger than the number of nodes, i.e. "
            + str(N)
            + ".\n\n"
        )

    # create the matrix to be reduced
    shortestPathLengthMatrix = sparse.csgraph.shortest_path(A, directed=False)

    if q == None:  # use the default setting of the multiplying factor
        maxSPL = np.ma.masked_invalid(shortestPathLengthMatrix).max()
        qmin = np.log(1.0 / 0.9999) * maxSPL
        qmax = np.log(10) * maxSPL
        q = np.exp((np.log(qmin) + np.log(qmax)) / 2)
        if verbose:
            print("The default q multiplying factor is " + str(q) + ".")

    # create the matrix of expected Lorentz products
    with np.errstate(divide="ignore"):
        D = np.exp(-q / shortestPathLengthMatrix)
    # in the original (non-exponential) algorithm: D=np.array([[shortestPathLengthsDict[s][t] for t in listOfNodes] for s in listOfNodes])

    # centering, i.e. the creation of the matrix of expected inner products
    # H = np.identity(N) - np.ones((N, N)) / N  # centering matrix
    # IP = (
    #    -np.matmul(np.matmul(H, np.multiply(D, D)), H) / 2
    # )  # multiply=element-wise product
    Dsq = D**2
    Dsq_H = Dsq - np.mean(Dsq, axis=0)[np.newaxis, :]
    IP = -(Dsq_H - np.mean(Dsq_H, axis=1)[:, np.newaxis]) / 2

    # dimension reduction
    if d == N:
        U, S, VT = np.linalg.svd(
            IP
        )  # find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
        # note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else:  # d<N: only the first d largest singular values (and the corresponding singular vectors) are retained
        U, S, VT = sparse.linalg.svds(
            IP, d, which="LM", solver="arpack"
        )  # the singular values are ordered from the smallest to the largest in S (increasing order)
        # reverse the order of the singular values to obtain a decreasing order, and reverse the order of singular vectors accordingly:
        S = S[::-1]
        U = U[:, ::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues < d:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.expISO can not be larger than the number of positive singular values of the inner product matrix, i.e. "
            + str(numOfPositiveSingularValues)
            + ".\n\n"
        )
    Ssqrt = np.sqrt(S)

    # create the dictionary of node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node in the Euclidean space
    Coord = np.einsum("ij,j->ij", U, Ssqrt)
    return Coord


@embedding_model
def ISO(G, d):
    """
    Create a weighted version of a graph based on its d-dimensional Euclidean embedding generated by the Isomap.

    Parameters
    ----------
    G : nx.Graph or scipy.sparse_csr_matrix
        The (weighted) NetworkX Graph or the adjacency matrix in form of the scipy sparse format.
        If link weights are inputted, these must mean distances,
        i.e. larger value:=less similarity or weaker connection!

    d : int
        The number of dimensions of the space to which the network will be embedded.

    q : float, optional
        Multiplying factor in the exponent in the matrix to be reduced (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal matrix elements towards 0). If None, its default setting will be used.

    Returns
    -------
    emb:  numpy.ndarray of shape (n_nodes, dim)

    Example
    -------
    emb = embedding.ISO(G, d)
    """
    A = utils.to_scipy_matrix(G, return_node_labels=False)
    N = A.shape[0]  # the number of nodes in graph G
    if N < d:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.ISO can not be larger than the number of nodes, i.e. "
            + str(N)
            + ".\n\n"
        )

    # create the matrix to be reduced
    D = sparse.csgraph.shortest_path(A, directed=False)

    # centering, i.e. the creation of the matrix of expected inner products
    # Dsq = D**2
    # H = sparse.eye(N) - np.ones((N, N)) / N  # centering matrix
    # IP = -(H @ Dsq) @ H / 2  # multiply=element-wise product
    # H_Dsq = Dsq - np.mean(Dsq, axis=0)[np.newaxis, :]
    # IP = -H_Dsq / 2
    Dsq = D**2
    Dsq_H = Dsq - np.mean(Dsq, axis=0)[np.newaxis, :]
    IP = -(Dsq_H - np.mean(Dsq_H, axis=1)[:, np.newaxis]) / 2

    # dimension reduction
    if d == N:
        U, S, VT = np.linalg.svd(
            IP
        )  # find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
        # note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else:  # d<N: only the first d largest singular values (and the corresponding singular vectors) are retained
        U, S, VT = sparse.linalg.svds(
            IP, d, which="LM", solver="arpack"
        )  # the singular values are ordered from the smallest to the largest in S (increasing order)
        # reverse the order of the singular values to obtain a decreasing order, and reverse the order of singular vectors accordingly:
        S = S[::-1]
        U = U[:, ::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues < d:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.expISO can not be larger than the number of positive singular values of the inner product matrix, i.e. "
            + str(numOfPositiveSingularValues)
            + ".\n\n"
        )
    Ssqrt = np.sqrt(S)

    # create the dictionary of node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node in the Euclidean space
    Coord = np.einsum("ij,j->ij", U, Ssqrt)
    return Coord


@embedding_model
def LE(G, d, scalingFactor=None, verbose=False):
    """
    Create a weighted version of a graph based on its d-dimensional Euclidean embedding generated by the Laplacian eigenmaps method.

    Parameters
    ----------
    G : nx.Graph or scipy.sparse_csr_matrix
        The (weighted) NetworkX Graph or the adjacency matrix in form of the scipy sparse format.
        If link weights are inputted, these must mean distances,
        i.e. larger value:=less similarity or weaker connection!

    d : int
        The number of dimensions of the space to which the network will be embedded.

    scalingFactor : float, optional
        The tunable parameter in the distance-proximity weight conversion formula. If None, its default setting will be used.

    verbose : bool, optional
        Whether to print information about the calculation process.

    Returns
    -------
    emb:  numpy.ndarray of shape (n_nodes, dim)

    Example
    -------
    >>> import embedding
    >>> emb = embedding.LE(G, d)
    """
    A = utils.to_scipy_matrix(G)
    N = A.shape[0]  # the number of nodes in graph G
    if N < d + 1:
        raise ValueError(
            "\n\nERROR: The number d of embedding dimensions in the function embedding.LE can not be larger than the number of nodes-1, i.e. "
            + str(N - 1)
            + ".\n\n"
        )

    weight = A.data.copy()  # edge weight
    is_unweighted = np.max(weight) == np.min(weight)
    if is_unweighted:
        deg = np.array(A.sum(axis=1)).reshape(-1)
        # Laplacian matrix
        D = sparse.diags(deg)
        L = D - A
    else:  # Laplacian Eigenmaps with heat kernel
        # default: the square of the average of all the link weights, as in https://www.nature.com/articles/s41467-017-01825-5
        if scalingFactor == None:  # use the default setting
            scalingFactor = np.mean(weight) ** 2
            if verbose:
                # another option could be e.g.: scalingFactor = math.pow(np.std(listOfWeights),2) #default: the square of the standard deviation of all the link weights, as in https://arxiv.org/abs/2304.06580
                print("The default scaling factor is " + str(scalingFactor) + ".")
        # assign a weight to the i-j edge:
        weight = np.exp(-(weight**2) / scalingFactor)
        Aw = A.copy()
        Aw.data = weight
        deg = np.array(Aw.sum(axis=1)).reshape(-1)
        D = sparse.diags(deg)
        L = D - Aw

    vals, Coord = sparse.linalg.eigsh(
        L, k=d + 1, M=D, which="SM", ncv=min(N, max(2 * (2 * (d + 1) + 1), 2 * 20))
    )
    vals, Coord = np.real(vals), np.real(Coord)
    order = np.argsort(vals)[1:]
    vals, Coord = vals[order], Coord[:, order]
    return Coord


@embedding_model
def node2vec(
    G,
    d=64,
    walkLength=10,
    numOfWalks=80,
    windowLength=10,
    batchWalks=10000,
    pForEmbedding=1.0,
    qForEmbedding=1.0,
):
    """
    Create a weighted version of a graph based on its angular node arrangement in a d-dimensional Euclidean embedding created by node2vec.

    Parameters
    ----------
    G : nx.Graph or scipy.sparse_csr_matrix
        The (weighted) NetworkX Graph to be embedded. If link weights are inputted, these must mean proximities, i.e. larger value:=higher similarity or stronger connection!

    d : int, optional
        Dimensionality of the embedding space. Default is 64.

    walkLength : int, optional
        Length of random walks. Default is 10.

    numOfWalks : int, optional
        Number of random walks per node. Default is 80.

    windowLength : int, optional
        Context size for optimization. Default is 10.

    batchWalks : int, optional
        Number of random walks processed at once. Default is 10000.

    pForEmbedding : float, optional
        Parameter for the node2vec random walk. Default is 1.0.

    qForEmbedding : float, optional
        Parameter for the node2vec random walk. Default is 1.0.

    Returns
    -------
    emb:  numpy.ndarray of shape (n_nodes, dim)

    Example
    -------
    >>> import embedding
    >>> emb = embedding.node2vec(G, d)
    """
    # create the embedding
    A = utils.to_scipy_matrix(G)
    model = fastnode2vec.Node2Vec(
        walk_length=walkLength,
        num_walks=numOfWalks,
        window_length=windowLength,
        batch_walks=batchWalks,
        p=pForEmbedding,
        q=qForEmbedding,
    )
    model.fit(A)
    Coord = model.transform(dim=d)  # d number of columns; 1 row=1 node
    return Coord
