import os

import gensim
import scipy.sparse
import numpy as np
import pickle

from src.utils.selector import select_path_from_dir
from src.utils.corpora import TokenSentenceStream
from src.utils.functions import makeBigrams
from src.utils.functions import append_vec2data


class AbstractParser:
    def __init__(self, model_name):

        path_to_db = "/media/norpheo/mySQL/db/ssorc"

        with open(os.path.join(path_to_db, "models", f"{model_name}.pickle"), "rb") as feature_file:
            classifier_dict = pickle.load(feature_file)

        settings = classifier_dict["settings"]

        self.feature_set = settings['feature_set']

        self.model = classifier_dict['model']

        self.word_dic = gensim.corpora.Dictionary.load(settings["word_dic"])
        self.wordbigram_dic = gensim.corpora.Dictionary.load(settings["wordbigram_dic"])
        self.pos_dic = gensim.corpora.Dictionary.load(settings["pos_dic"])
        self.posbigram_dic = gensim.corpora.Dictionary.load(settings["posbigram_dic"])

        self.word_tfidf = gensim.models.TfidfModel.load(settings["word_tfidf"])
        self.wordbigram_tfidf = gensim.models.TfidfModel.load(settings["wordbigram_tfidf"])
        self.pos_tfidf = gensim.models.TfidfModel.load(settings["pos_tfidf"])
        self.posbigram_tfidf = gensim.models.TfidfModel.load(settings["posbigram_tfidf"])

        self.word_vec_len = len(self.word_dic)
        self.wordbigram_vec_len = len(self.wordbigram_dic)
        self.pos_vec_len = len(self.pos_dic)
        self.posbigram_vec_len = len(self.posbigram_dic)

        conc_file_path = os.path.join(path_to_db, 'external', 'concreteness_ratings.txt')
        with open(conc_file_path) as conc_file:
            cratings = conc_file.readlines()

        self.conc_rating = dict()
        for line in cratings[1:]:
            elements = line.split('\t')
            if elements[1] == "0":
                self.conc_rating[elements[0]] = float(elements[2])

        self.vector_len = 0

        if 'location' in self.feature_set:
            self.vector_len += 1

        if 'concreteness' in self.feature_set:
            self.vector_len += 3

        if 'wordunigram' in self.feature_set:
            self.vector_len += self.word_vec_len

        if 'wordbigram' in self.feature_set:
            self.vector_len += self.wordbigram_vec_len

        if 'posunigram' in self.feature_set:
            self.vector_len += self.pos_vec_len

        if 'posbigram' in self.feature_set:
            self.vector_len += self.posbigram_vec_len

        print(self.vector_len)

    def predict(self, ot_tokens, pos_tokens):
        feature_vector = self.get_feature_vector(ot_tokens, pos_tokens)
        if feature_vector.shape[0] <= 1:
            return None
        prediction = self.model.predict(feature_vector)
        return prediction

    def get_feature_vector(self, word_corpus, pos_corpus):

        sent_infos = list()
        max_sent = 0
        sent_id = 0

        for words, pos in zip(word_corpus, pos_corpus):
            sent_id += 1
            max_sent += 1

            wordbigram = makeBigrams(words)
            posbigram = makeBigrams(pos)

            word_bow = self.word_dic.doc2bow(words)
            vec_word_tfidf = self.word_tfidf[word_bow]
            wordbigram_bow = self.wordbigram_dic.doc2bow(wordbigram)
            vec_wordbigram_tfidf = self.wordbigram_tfidf[wordbigram_bow]

            pos_bow = self.pos_dic.doc2bow(pos)
            vec_pos_tfidf = self.pos_tfidf[pos_bow]

            posbigram_bow = self.posbigram_dic.doc2bow(posbigram)
            vec_posbigram_tfidf = self.posbigram_tfidf[posbigram_bow]

            # Collecting Concreteness-Ratings
            cr_min = 1000
            cr_max = 0
            cr_mean = 0

            cr_words = [cr_word for cr_word in words if cr_word in self.conc_rating]

            for word in cr_words:
                rating = self.conc_rating[word]
                cr_mean += rating
                if rating > cr_max:
                    cr_max = rating
                if rating < cr_min:
                    cr_min = rating

            if cr_min > cr_max:
                cr_max_feature = 0
                cr_min_feature = 0
                cr_mean_feature = 0
            else:
                cr_mean = cr_mean / len(cr_words)
                cr_max_feature = cr_max
                cr_min_feature = cr_min
                cr_mean_feature = cr_mean

            sent_info = dict()
            sent_info['cr_max_feature'] = cr_max_feature
            sent_info['cr_min_feature'] = cr_min_feature
            sent_info['cr_mean_feature'] = cr_mean_feature
            sent_info['vec_word_tfidf'] = vec_word_tfidf
            sent_info['vec_wordbigram_tfidf'] = vec_wordbigram_tfidf
            sent_info['vec_pos_tfidf'] = vec_pos_tfidf
            sent_info['vec_posbigram_tfidf'] = vec_posbigram_tfidf
            sent_info['sent_id'] = sent_id
            sent_infos.append(sent_info)

        feature_data_array = []
        feature_row = []
        feature_col = []

        row_count = 0

        for feature_data in sent_infos:
            sid = feature_data['sent_id']

            vector_offset = 0

            if 'location' in self.feature_set:
                feature_row.append(row_count)
                feature_col.append(vector_offset)
                feature_data_array.append(sid / max_sent)

                vector_offset += 1

            if 'concreteness' in self.feature_set:
                cr_max_feature = float(feature_data['cr_max_feature'])
                cr_min_feature = float(feature_data['cr_min_feature'])
                cr_mean_feature = float(feature_data['cr_mean_feature'])

                feature_row.append(row_count)
                feature_col.append(vector_offset)
                feature_data_array.append(cr_max_feature)

                feature_row.append(row_count)
                feature_col.append(vector_offset + 1)
                feature_data_array.append(cr_min_feature)

                feature_row.append(row_count)
                feature_col.append(vector_offset + 2)
                feature_data_array.append(cr_mean_feature)
                vector_offset += 3

            if 'wordunigram' in self.feature_set:
                append_vec2data(feature_data['vec_word_tfidf'],
                                feature_data_array,
                                feature_row,
                                feature_col,
                                row_count,
                                vector_offset)
                vector_offset += self.word_vec_len

            if 'wordbigram' in self.feature_set:
                append_vec2data(feature_data['vec_wordbigram_tfidf'],
                                feature_data_array,
                                feature_row,
                                feature_col,
                                row_count,
                                vector_offset)
                vector_offset += self.wordbigram_vec_len

            if 'posunigram' in self.feature_set:
                append_vec2data(feature_data['vec_pos_tfidf'],
                                feature_data_array,
                                feature_row,
                                feature_col,
                                row_count,
                                vector_offset)
                vector_offset += self.pos_vec_len

            if 'posbigram' in self.feature_set:
                append_vec2data(feature_data['vec_posbigram_tfidf'],
                                feature_data_array,
                                feature_row,
                                feature_col,
                                row_count,
                                vector_offset)
                vector_offset += self.posbigram_vec_len

            row_count += 1

        feature_row = np.array(feature_row)
        feature_col = np.array(feature_col)
        feature_data_array = np.array(feature_data_array)

        feature_vector = scipy.sparse.csc_matrix((feature_data_array, (feature_row, feature_col)),
                                                 shape=(row_count, self.vector_len))

        return feature_vector
