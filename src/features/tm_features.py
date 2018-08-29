import os
import gensim
import scipy.sparse
import numpy as np

from src.utils.LoopTimer import LoopTimer

import src.utils.corpora as corpora


def build_scipy_feature_file(dtype, prefix='', num_samples=10000):
    dirname = os.path.dirname(__file__)
    dictionary_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/dictionaries')
    tfidf_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/tfidf')
    feature_file = os.path.join(dirname, '../../data/processed/' + dtype + '/features/tm_features_' + prefix + '.npz')

    if os.path.isfile(feature_file):
        os.remove(feature_file)

    lemma_dic = gensim.corpora.Dictionary.load(os.path.join(dictionary_dir, 'lemma.dic'))

    lemma_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'lemma_model.tfidf'))

    lemma_corpus = corpora.lemma_doc_stream(dtype)

    lt = LoopTimer()

    row = []
    col = []
    data = []
    for idx, lemmas in enumerate(lemma_corpus):
        if num_samples != -1 and idx == num_samples:
            break

        lemma_bow = lemma_dic.doc2bow(lemmas[1])
        vec_lemma_tfidf = lemma_tfidf[lemma_bow]

        for entry in vec_lemma_tfidf:
            row.append(idx)
            col.append(entry[0])
            data.append(entry[1])
        lt.update("Build Features")

    m = idx + 1
    n = len(lemma_dic)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    feature_vector = scipy.sparse.csc_matrix((data, (row, col)), shape=(m, n))

    scipy.sparse.save_npz(feature_file, feature_vector)


def get_scipy_features(dtype, prefix=""):
    dirname = os.path.dirname(__file__)
    feature_file = os.path.join(dirname, '../../data/processed/' + dtype + '/features/tm_features_' + prefix + '.npz')

    if os.path.isfile(feature_file):
        features = scipy.sparse.load_npz(feature_file)
        return features
    else:
        return None
