# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-03 22:10:13
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-09-21 12:08:49
# %%
import numpy as np
from iteremb import embedding
from iteremb import edge_weighting
from iteremb import utils
from tqdm import tqdm


def iterative_embedding(
    net,
    dim,
    emb_model,
    weighting_model,
    max_iter=20,
    emb_params={},
    edge_weighting_params={},
    tol=1e-2,
):
    net_t = utils.to_scipy_matrix(net)
    emb_list, net_list = [], []
    prev_ave_edge_weight = np.inf
    pbar = tqdm(total=max_iter)
    for it in range(max_iter):
        # Embedding and network construction
        emb_t = emb_model(net_t, d=dim, **emb_params)
        net_t = weighting_model(net_t, emb_t, **edge_weighting_params)

        # Save
        emb_list.append(emb_t)
        net_list.append(net_t)

        # Decide whether to continue or stop
        ave_edge_weight = net_t.mean()
        edge_diff_ratio = (
            np.abs(ave_edge_weight - prev_ave_edge_weight) / ave_edge_weight
        )
        pbar.set_description(f"Diff ratio = {edge_diff_ratio:.2f}")
        pbar.update(1)

        if edge_diff_ratio < tol:
            break
        prev_ave_edge_weight = ave_edge_weight

    return net_t, emb_list, net_list


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
    "testNode2vec": {
        "emb_model": embedding.models["node2vec"],
        "weighting_model": edge_weighting.models["exp_cosine_similarity"],
    },
}
