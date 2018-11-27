import os
import gensim
import pickle
import numpy as np
import scipy.sparse


class TopicModelingGLDA:
    def __init__(self, dictionary_name, model_name):
        path_to_db = "/media/norpheo/mySQL/db/ssorc"
        dictionary_file_path = os.path.join(path_to_db, 'dictionaries', dictionary_name)
        model_file_path = os.path.join(path_to_db, 'models', model_name)

        self.dictionary = gensim.corpora.Dictionary.load(dictionary_file_path)
        with open(model_file_path, 'rb') as model_file:
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

        return topic_distribution

    def get_topic(self, input_document):
        topic_dist = self.get_topic_dist(input_document)
        topic = topic_dist[0].argmax()

        return topic

    def phi(self, topic, word):
        if isinstance(word, str):
            word_str = word
            if word_str not in self.dictionary.token2id:
                return 0.0
        elif isinstance(word, int):
            word_str = self.dictionary[word]
        else:
            return 0.0

        word_dist = self.lda_model.show_topic(topic, topn=len(self.dictionary))
        for word_prob in word_dist:
            if word_prob[0] == word_str:
                return word_prob[1]

        return 0.0


class TopicModelingLDA:
    def __init__(self, dictionary_name, tfidf_name, model_name):
        path_to_db = "/media/norpheo/mySQL/db/ssorc"
        path_to_models = os.path.join(path_to_db, 'models')
        path_to_dictionaries = os.path.join(path_to_db, 'dictionaries')
        dictionary_path = os.path.join(path_to_dictionaries, dictionary_name)
        model_path = os.path.join(path_to_models, model_name)
        tfidf_path = os.path.join(path_to_models, tfidf_name)

        self.dictionary = gensim.corpora.Dictionary.load(dictionary_path)
        self.tfidf = gensim.models.TfidfModel.load(tfidf_path)
        self.lda_model = gensim.models.LdaMulticore.load(model_path)

    def num_topics(self):
        return self.lda_model.num_topics

    def get_topic_dist(self, input_document):
        bow = self.dictionary.doc2bow(input_document)
        vec_tfidf = self.tfidf[bow]
        topic_distribution = self.lda_model[vec_tfidf]

        return topic_distribution

    def get_topic(self, input_document):
        topic_dist = self.get_topic_dist(input_document)
        topic = None
        max_p = 0
        for entry in topic_dist:
            topic_id = entry[0]
            topic_p = entry[1]
            if topic_p > max_p:
                max_p = topic_p
                topic = topic_id

        return topic

    def phi(self, topic, word):
        if isinstance(word, str):
            word_str = word
            if word_str not in self.dictionary.token2id:
                return 0.0
        elif isinstance(word, int):
            word_str = self.dictionary[word]
        else:
            return 0.0

        word_dist = self.lda_model.show_topic(topic, topn=len(self.dictionary))
        for word_prob in word_dist:
            if word_prob[0] == word_str:
                return word_prob[1]

        return 0.0
