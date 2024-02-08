[![Unit Test & Deploy](https://github.com/BianKov/iterEmb/actions/workflows/main.yml/badge.svg)](https://github.com/BianKov/iterEmb/actions/workflows/main.yml)

# iterEmb

This is a Python package for iterative embedding of networks. It is based on the paper "Iterative embedding and reweighting of complex networks reveals community structure" by Bianka Kovács, Sadamori Kojaku, Gergely Palla and Santo Fortunato.

# How to install
```bash
pip install git+https://github.com/BianKov/iterEmb
```

# Usage

## Basic

```python
import iteremb
import networkx as nx

net = nx.adjacency_matrix(
    nx.karate_club_graph()
)  # the adjacency matrix (scipy sparse matrix format)

new_net, emb_list, net_list = iteremb.iterative_embedding(
    net,  # Input network
    dim=32,  # Embedding dimension
    **iteremb.iterative_embedding_models["TREXPIC"],  # Embedding models
)
```

See the list of [the off-the-shelf models](#off-the-shelf-embedding-models) for the available models.

## Advanced

```python
import iteremb
import networkx as nx

net = nx.adjacency_matrix(
    nx.karate_club_graph()
)  # the adjacency matrix (scipy sparse matrix format)

new_net, emb_list, net_list = iteremb.iterative_embedding(
    net,  # Input network
    dim=32,  # Embedding dimension
    **iteremb.iterative_embedding_models["TREXPIC"],  # Embedding models
    max_iter=20,  # Maximum number of iterations
    emb_params={},  # parameter for the embedding function
    edge_weighting_params={},  # parameters for the edge weighting function
    preprocessing_func=None,  # Optional preprocessing function applied to the network before embedding
    tol=1e-2,  # Tolerance. Larger value yields fewer iterations.
)

new_net  # Final network after the iterations
emb_list  # List of embedding generated during the iteration. emb_list[t] is the one at the $t$th iteration
net_list  # List of networks generated during the iteration. net_list[t] is the one at the $t$th iteration
```

The `iteremb` package implements several embedding models as well as edge weighting functions. The embedding models can be accessible through
```python
from iteremb import embedding
embedding.models # dictionary of embedding functions
```
And the edge weighting function:
```python
from iteremb import edge_weighting
edge_weighting.models # dictionary of edge weighting functions
```

### Custom embedding and edge weighting functions

You can integrate your custom embedding/edge weighting functions into our frametwork through `iterative_embedding` `emb_model` and `weighting_model`, arguments. To this end, define the following two functions:

```python
def custom_embedding_model(G, d, **params):
    """
    A custom embedding model example that embeds a graph G into a d-dimensional space.

    Parameters:
    -----------
    G : nx.Graph or scipy.sparse_csr_matrix
        The (weighted) NetworkX Graph or the adjacency matrix in form of the scipy sparse format.
    d : int
        The number of dimensions of the space to which the network will be embedded.
    **params : dict
        Additional parameters for the embedding model.

    Returns:
    --------
    emb: numpy.ndarray of shape (n_nodes, dim)
        The embedding of the graph G in a d-dimensional space.
    """
    pass

def custom_edge_weighting_model(A, emb, **params):
    """
    A custom edge weighting model example that uses embeddings to weight the edges of a graph.

    Parameters:
    -----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of a graph.
    emb : numpy.ndarray
        The node embedding matrix with shape (num_nodes, dim).
    **params : dict
        Additional parameters for the edge weighting model.

    Returns:
    --------
    A_weighted : scipy.sparse.csr_matrix
        A sparse matrix representing the weighted adjacency matrix of the graph.
    """
    pass
```

Then, pass the functions to the `iterative_embedding` function by

```python
import iteremb
import networkx as nx

net = nx.adjacency_matrix(
    nx.karate_club_graph()
)  # the adjacency matrix (scipy sparse matrix format)

new_net, emb_list, net_list = iteremb.iterative_embedding(
    net,  # Input network
    dim=16,  # Embedding dimension
    emb_model = custom_embedding_model, # Your custom embedding model.
    weighting_model =custom_edge_weighting_mode,  # your edge weighting model.
    max_iter=20,  # Maximum number of iterations
    emb_params={},  # parameter for the embedding function
    edge_weighting_params={},  # parameters for the edge weighting function
    preprocessing_func=None,  # Optional preprocessing function applied to the network before embedding
    tol=1e-2,  # Tolerance. Larger value yields fewer iterations.
)

```

Sometimes it is necessary to configure the embedding and edge weighting functions based on the network structure before iteration. For example, you might want to determine the factor of the exponentialization of edge weight based on the input network before the iterative emebdding. This is where `preprocessing_func` is useful.

`preprocessing_func` takes the given network and both `emb_params` and `edge_weighting_params` as input. Then, it returns modified `emb_params` and `edge_weighting_params`. For example,

```python

def example_preprocessing_func(network, emb_params, edge_weighting_params):
    """
    An example preprocessing function that modifies embedding and edge weighting parameters based on the network structure.

    Parameters:
    -----------
    network : scipy.sparse.csr_matrix
        The adjacency matrix of a graph.
    emb_params : dict
        Dictionary containing parameters for the embedding model.
    edge_weighting_params : dict
        Dictionary containing parameters for the edge weighting model.

    Returns:
    --------
    emb_params : dict
        Modified embedding parameters.
    edge_weighting_params : dict
        Modified edge weighting parameters.
    """
    pass
```

## Off-the-shelf embedding models:


| Key          | Embedding model | Edge weighting method      | Preprocessing Function         |
|--------------|-----------------|----------------------------|--------------------------------|
| TREXPIC      | TREXPIC         | cosine_distance            | None                           |
| expISO       | expISO          | cosine_distance            | None                           |
| ISO          | ISO             | cosine_distance            | None                           |
| LE           | LE              | cosine_distance            | None                           |
| node2vec     | node2vec        | cosine_similarity          | None                           |
| expNode2vec  | node2vec        | exp_cosine_similarity      | q_factor_determination        |

See `iteremb/iterative_embedding_models.py`, `iteremb/embedding.py`, and `iteremb/edge_weighting.py` for the implementations.
