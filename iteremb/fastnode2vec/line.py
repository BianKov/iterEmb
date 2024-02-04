from .node2vec import Node2Vec


class LINE(Node2Vec):
    def __init__(
        self, walk_length=80, workers=1, num_walks=10, batch_walks=None, seed=None,
    ):
        super().__init__(
            walk_length=walk_length,
            window_length=1,
            p=1,
            q=1,
            start_node_sampling_method="degree",
            workers=workers,
            num_walks=num_walks,
            batch_walks=batch_walks,
            seed=seed,
        )
