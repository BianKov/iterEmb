from gensim.models import Word2Vec
from gensim import __version__ as gensim_version
import numpy as np
from numba import njit
from .graph import Graph
from tqdm import tqdm


@njit
def set_seed(seed):
    np.random.seed(seed)


class Node2Vec(Word2Vec):
    def __init__(
        self,
        walk_length=80,
        window_length=10,
        p=1.0,
        q=1.0,
        start_node_sampling_method="uniform",
        start_node_sampling_prob=None,
        workers=1,
        num_walks=10,
        batch_walks=None,
        seed=None,
    ):
        if batch_walks is None:
            batch_words = 10000
        else:
            batch_words = min(walk_length * batch_walks, 10000)

        self.window_length = window_length
        self.workers = workers
        self.batch_words = batch_words
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.seed = seed
        self.num_walks = num_walks
        self.start_node_sampling_method = start_node_sampling_method
        self.start_node_sampling_prob = start_node_sampling_prob
        self.args = {"sg": 1, "min_count": 1}

    def fit(self, A):
        self.graph = Graph(A)
        self.num_nodes = A.shape[0]

        if self.start_node_sampling_method == "degree":
            self.start_node_sampling_prob = np.array(A.sum(axis=0)).reshape(-1).astype(float)
            self.start_node_sampling_prob /= np.sum(self.start_node_sampling_prob)
        elif self.start_node_sampling_method == "custom":
            self.start_node_sampling_prob /= np.sum(self.start_node_sampling_prob)

    def transform(self, dim, progress_bar=True, **kwargs):
        def gen_nodes(epochs, prob=None):
            if self.seed is not None:
                np.random.seed(self.seed)
            if prob is None:
                prob = np.ones(self.num_nodes) / self.num_nodes
                replace = False
            else:
                replace = True
            for _ in range(epochs):
                node_list = np.random.choice(
                    self.num_nodes, size=self.num_nodes, p=prob, replace=replace
                )
                for i in node_list:
                    # dummy walk with same length
                    yield [i] * self.walk_length

        if gensim_version < "4.0.0":
            self.args["iter"] = 1
            self.args["size"] = dim
        else:
            self.args["epochs"] = 1
            self.args["vector_size"] = dim

        super().__init__(
            window=self.window_length,
            workers=self.workers,
            batch_words=self.batch_words,
            **self.args,
        )
        self.build_vocab(([w] for w in range(self.num_nodes)))

        if progress_bar:

            def pbar(it):
                return tqdm(it, desc="Training", total=self.num_walks * self.num_nodes)

        else:

            def pbar(it):
                return it

        super().train(
            pbar(gen_nodes(self.num_walks, self.start_node_sampling_prob)),
            total_examples=self.num_walks * self.num_nodes,
            epochs=1,
            **kwargs,
        )

        self.in_vec = np.zeros((self.num_nodes, dim))
        self.out_vec = np.zeros((self.num_nodes, dim))
        for i in range(self.num_nodes):
            if i not in self.wv:
                continue
            self.in_vec[i, :] = self.wv[i]
            self.out_vec[i, :] = self.syn1neg[self.wv.key_to_index[i]]
        return self.in_vec

    def generate_random_walk(self, t):
        return self.graph.generate_random_walk(self.walk_length, self.p, self.q, t)

    def _do_train_job(self, sentences, alpha, inits):
        if self.seed is not None:
            set_seed(self.seed)
        sentences = [self.generate_random_walk(w[0]) for w in sentences]
        return super()._do_train_job(sentences, alpha, inits)
