import os
import gensim
import pickle
import numpy as np
import scipy.sparse

path_to_db = "/media/norpheo/mySQL/db/ssorc"

class TopicModelingGLDA:
    def __init__(self, dic_path, model_path):
        self.dictionary = gensim.corpora.Dictionary.load(dic_path)
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def num_topics(self):
        return self.model.n_topics

    def get_topic_dist(self, input_document):
        bow = self.dictionary.doc2bow(input_document)

        row = []
        col = []
        data = []

        for entry in bow:
            row.append(0)
            col.append(entry[0])
            data.append(entry[1])

        m = 1
        n = len(self.dictionary)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)

        feature_vector = scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n)).toarray()
        topic_distribution = self.model.transform(feature_vector)
        return topic_distribution[0]

    def get_topic(self, input_document):
        topic_dist = self.get_topic_dist(input_document)
        topic = topic_dist.argmax()
        return topic


class TopicModelingLDA:
    def __init__(self, info_fn):
        with open(os.path.join(path_to_db, "topic_modeling", info_fn), "rb") as handle:
            info = pickle.load(handle)
        self.dictionary = gensim.corpora.Dictionary.load(info["dic_path"])
        with open(info["model_path"], 'rb') as model_file:
            self.model = pickle.load(model_file)

    def num_topics(self):
        return self.model.components_.shape[0]

    def get_topic_dist(self, input_document):
        bow = self.dictionary.doc2bow(input_document)
        row = []
        col = []
        data = []

        for entry in bow:
            row.append(0)
            col.append(entry[0])
            data.append(entry[1])

        m = 1
        n = len(self.dictionary)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)

        feature_vector = scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n)).toarray()
        topic_distribution = self.model.transform(feature_vector)

        return topic_distribution[0]

    def get_topic(self, input_document):
        topic_dist = self.get_topic_dist(input_document)
        topic = topic_dist.argmax()
        return topic
