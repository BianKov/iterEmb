from .node2vec import Node2Vec


class DeepWalk(Node2Vec):
    def __init__(
        self,
        walk_length=80,
        window_length=10,
        p=1.0,
        q=1.0,
        workers=1,
        num_walks=10,
        batch_walks=None,
        seed=None,
    ):
        super().__init__(
            walk_length=walk_length,
            window_length=window_length,
            p=p,
            q=q,
            workers=workers,
            num_walks=num_walks,
            batch_walks=batch_walks,
            seed=seed,
        )
        self.args["sg"] = 0
        self.args["hs"] = 1
