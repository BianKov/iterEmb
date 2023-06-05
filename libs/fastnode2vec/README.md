# fastnode2vec

This is an adaptation of [fastnode2vec](https://github.com/louisabraham/fastnode2vec) implemented by Louis Abraham. All adaptations are done for my personal use. 

Please cite the developper's repository if you use this package:

```
@software{fastnode2vec,
  author       = {Louis Abraham},
  title        = {fastnode2vec},
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3902632},
  url          = {https://doi.org/10.5281/zenodo.3902632}
}
```


# Adaptation

- In the original code, the nodes can have string labels, which are then reindexed as integer IDs. I remove this process of translating labels to integers because I have done this before making the graph, i.e., all nodes have integer IDs already, and I don't want to waste time for mapping them to new IDs.
- I personally prefer the `scikit-learn` style, where all you need to do, with any models, is `fit` and `transform`. I adapt the code to follow this design principle.


# Usage

```python
    model = fastnode2vec.Node2Vec()
    model.fit(A)
    center_vec = model.transform(dim=32)
    context_vec = model.out_vec
```

- `A`: adjacency matrix (`csr_matrix`)
