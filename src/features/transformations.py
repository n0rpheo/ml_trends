import numpy as np


def tokens_to_mean_w2v(w2v, tokens, dim):
    transform_features = np.mean([w2v[w] for w in tokens if w in w2v] or [np.zeros(dim)], axis=0)
    return transform_features
