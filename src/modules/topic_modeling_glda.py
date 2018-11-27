import os
import gensim
import pickle


class TopicModeling:
    def __init__(self, dictionary_name, model_name):
        path_to_db = "/media/norpheo/mySQL/db/ssorc"
        dictionary_file_path = os.path.join(path_to_db, 'dictionaries', dictionary_name)
        model_file_path = os.path.join(path_to_db, 'models', model_name)

        self.dictionary = gensim.corpora.Dictionary.load(dictionary_file_path)
        with open(model_file_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

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
