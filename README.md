# How to install
```bash
conda create -n iterEmb python=3.9
conda activate iterEmb
conda install -c conda-forge mamba -y
mamba install -c conda-forge -c bioconda -c conda-forge snakemake graph-tool scikit-learn numpy==1.23.5 numba scipy pandas networkx seaborn matplotlib gensim ipykernel tqdm black -y

pip install python-louvain
pip install infomap

pip install -e .
cd libs/fastnode2vec && pip install -e .
```

# Usage

```python
import iteremb
import networkx as nx

net = nx.adjacency_matrix(
    nx.karate_club_graph()
)  # the adjacency matrix (scipy sparse matrix format)

new_net, emb_list, net_list = iteremb.iterative_embedding(
    net,  # Input network
    dim=16,  # Embedding dimension
    **iteremb.iterative_embedding_models["TREXPIC"],  # Embedding models
    max_iter=20,  # Maximum number of iterations
    emb_params={},  # parameter for the embedding function
    edge_weighting_params={},  # parameters for the edge weighting function
    tol=1e-2,  # Tolerance. Larger value yields fewer iterations.
)

new_net  # Final network after the iterations
emb_list  # List of embedding generated during the iteration. emb_list[t] is the one at the $t$th iteration
net_list  # List of networks generated during the iteration. net_list[t] is the one at the $t$th iteration
```

# Off-the-shelf iterative embedding models

This package contains the following off-the-shelf models for iterative embedding:

```python
from iteremb import embedding, edge_weighting

iterative_embedding_models = {
    "TREXPIC": {
        "emb_model": embedding.models["TREXPIC"],
        "weighting_model": edge_weighting.models["cosine_distance"],
    },
    "expISO": {
        "emb_model": embedding.models["expISO"],
        "weighting_model": edge_weighting.models["cosine_distance"],
    },
    "ISO": {
        "emb_model": embedding.models["ISO"],
        "weighting_model": edge_weighting.models["cosine_distance"],
    },
    "LE": {
        "emb_model": embedding.models["LE"],
        "weighting_model": edge_weighting.models["cosine_distance"],
    },
    "node2vec": {
        "emb_model": embedding.models["node2vec"],
        "weighting_model": edge_weighting.models["cosine_similarity"],
    },
    "expNode2vec": {
        "emb_model": embedding.models["node2vec"],
        "weighting_model": edge_weighting.models["exp_cosine_similarity"],
    },
}
```

For example, the following is the iterative embedding with node2vec:

```python
import iteremb
import networkx as nx

net = nx.adjacency_matrix(
    nx.karate_club_graph()
)  # the adjacency matrix (scipy sparse matrix format)

new_net, emb_list, net_list = iteremb.iterative_embedding(
    net,  # Input network
    dim=16,  # Embedding dimension
    **iteremb.iterative_embedding_models["node2vec"],  # Embedding models
    max_iter=20,  # Maximum number of iterations
    emb_params={},  # parameter for the embedding function
    edge_weighting_params={},  # parameters for the edge weighting function
    tol=1e-2,  # Tolerance. Larger value yields fewer iterations.
)
```

# Custom embedding and edge weighting

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

You can also implement your custom embedding/edge weighting functions, and pass them to the `iterative_embedding` functions, with argument name `emb_model` and `weighting_model`, respectively. For example,

```python
import iteremb
import networkx as nx

net = nx.adjacency_matrix(
    nx.karate_club_graph()
)  # the adjacency matrix (scipy sparse matrix format)

new_net, emb_list, net_list = iteremb.iterative_embedding(
    net,  # Input network
    dim=16,  # Embedding dimension
    emb_model = ..., # Your custom embedding model.
    weighting_model = ...,  # your edge weighting model.
    max_iter=20,  # Maximum number of iterations
    emb_params={},  # parameter for the embedding function
    edge_weighting_params={},  # parameters for the edge weighting function
    tol=1e-2,  # Tolerance. Larger value yields fewer iterations.
)
```
The emb_model and weighting_model are expected to take the following arguments:
```python
    emb_t = emb_model(net_t, d=dim, **emb_params) # net_t: scipy adjacency matrix, emb_t is a numpy array of shape (num. nodes, dim)
    net_t = weighting_model(net_t, emb_t, **edge_weighting_params)
```

# iterEmb

Two executable Python files are attached:
- embedWithSpectralMethods.py shows through the example of Laplacian Eigenmaps how I use spectral embedding methods. Besides Laplacian Eigenmaps, the implemented embeddings are the Euclidean Isomap (which I use with a new, exponentializing step), and its hyperbolic analogue given by TREXPIC. Isomap and TREXPIC are slower (calculate all the shortest path lengths in the network) and more memory intensive (store the matrix of shortest path lengths) than Laplacian Eigenmaps.
- embedWithRandomWalkMethods.py shows how I iterate node2vec. An important difference compared to the applied spectral methods is that node2vec needs proximity-like link weights as input (where higher values indicate stronger connections and thus, smaller expected distance in the embedding space), while the other methods start from distance-like link weights (higher values correspond to higher expected distances in the embedding space). Similarly, the returned NetworkX graph contains proximity-like link weights in the case of node2vec and distance-like link weights in the case of the spectral methods.

I included in the examples a simple weight thresholding technique for community detection, as well as Louvain (that needs proximity-like weights, and therefore, its application is different in the case of node2vec and the spectral methods) and k-means clustering (that builds solely on the node coordinates and does not use the weighted graph, so it can be used in the same way for all the implemented Euclidean embedding methods).

The codes save the modularity of the detected community structures and also similarity measures (adjusted mutual information, adjusted rand index and element-centric similarity) calculated between the detected and the planted partitions (the example graph was generated by the planted partition model).

Please note that some functions that I use are not included in NetworkX 3.0 and 3.1. I tried to indicate in comments in embedding.py what could be done in newer versions of NetworkX where I noticed that problems will occur, but I mostly use older versions of NetworkX like 2.8.
