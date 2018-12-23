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
    def __init__(self,
                 model_name,
                 feature_set=["location",
                              "concreteness",
                              "posunigramm",
                              "posbigramm",
                              "wordunigramm",
                              "wordbigramm"]):
        self.feature_set = feature_set

        path_to_db = "/media/norpheo/mySQL/db/ssorc"
        dictionary_dir = os.path.join(path_to_db, 'dictionaries')
        model_dir = os.path.join(path_to_db, 'models')

        with open(os.path.join(model_dir, model_name), 'rb') as model_file:
            self.svm_model = pickle.load(model_file)


        self.word_dic = gensim.corpora.Dictionary.load(select_path_from_dir(dictionary_dir,
                                                                            phrase="Select Word Dict: ",
                                                                            suffix=".dict",
                                                                            preselection="pruned_word_ml.dict"))
        self.wordbigramm_dic = gensim.corpora.Dictionary.load(select_path_from_dir(dictionary_dir,
                                                                                   phrase="Select Word-bigramm Dict: ",
                                                                                   suffix=".dict",
                                                                                   preselection="pruned_wordbigramm_ml.dict"))
        self.pos_dic = gensim.corpora.Dictionary.load(select_path_from_dir(dictionary_dir,
                                                                           phrase="Select POS Dict: ",
                                                                           suffix=".dict",
                                                                           preselection="full_pos_ml.dict"))
        self.posbigramm_dic = gensim.corpora.Dictionary.load(select_path_from_dir(dictionary_dir,
                                                                                  phrase="Select POS-bigram Dict: ",
                                                                                  suffix=".dict",
                                                                                  preselection="full_posbigramm_ml.dict"))

        self.word_tfidf = gensim.models.TfidfModel.load(select_path_from_dir(model_dir,
                                                                             phrase="Select Word-TFIDF Model: ",
                                                                             suffix=".tfidf",
                                                                             preselection="pruned_word_ml.tfidf"))
        self.wordbigramm_tfidf = gensim.models.TfidfModel.load(select_path_from_dir(model_dir,
                                                                                    phrase="Select Word-Bigramm-TFIDF Model: ",
                                                                                    suffix=".tfidf",
                                                                                    preselection="pruned_wordbigramm_ml.tfidf"))
        self.pos_tfidf = gensim.models.TfidfModel.load(select_path_from_dir(model_dir,
                                                                            phrase="Select POS-TFIDF Model: ",
                                                                            suffix=".tfidf",
                                                                            preselection="full_pos_ml.tfidf"))
        self.posbigramm_tfidf = gensim.models.TfidfModel.load(select_path_from_dir(model_dir,
                                                                                   phrase="Select POS-Bigramm-TFIDF Model: ",
                                                                                   suffix=".tfidf",
                                                                                   preselection="full_posbigramm_ml.tfidf"))
        self.word_vec_len = len(self.word_dic)
        self.wordbigramm_vec_len = len(self.wordbigramm_dic)
        self.pos_vec_len = len(self.pos_dic)
        self.posbigramm_vec_len = len(self.posbigramm_dic)

        conc_file_path = os.path.join(path_to_db, 'external', 'concreteness_ratings.txt')
        with open(conc_file_path) as conc_file:
            cratings = conc_file.readlines()

        self.conc_rating = dict()
        for line in cratings[1:]:
            elements = line.split('\t')
            if elements[1] == "0":
                self.conc_rating[elements[0]] = float(elements[2])

        self.vector_len = 0

        if 'location' in feature_set:
            self.vector_len += 1

        if 'concreteness' in feature_set:
            self.vector_len += 3

        if 'wordunigramm' in feature_set:
            self.vector_len += self.word_vec_len

        if 'wordbigramm' in feature_set:
            self.vector_len += self.wordbigramm_vec_len

        if 'posunigramm' in feature_set:
            self.vector_len += self.pos_vec_len

        if 'posbigramm' in feature_set:
            self.vector_len += self.posbigramm_vec_len

    def predict(self, ot_tokens, pos_tokens):
        feature_vector = self.get_feature_vector(ot_tokens, pos_tokens)
        if feature_vector.shape[0] <= 1:
            return None
        prediction = self.svm_model.predict(feature_vector)
        return prediction

    def predict_abstract(self, abstract_id):
        abstracts = [abstract_id]
        word_corpus = TokenSentenceStream(abstracts=abstracts,
                                          token_type='word',
                                          token_cleaned=0,
                                          output=None,
                                          lower=True)

        pos_corpus = TokenSentenceStream(abstracts=abstracts,
                                         token_type='pos',
                                         token_cleaned=0,
                                         output=None,
                                         lower=True)

        return self.predict(word_corpus, pos_corpus)

    def get_feature_vector(self, word_corpus, pos_corpus):

        sent_infos = list()
        max_sent = 0
        sent_id = 0

        for words, pos in zip(word_corpus, pos_corpus):
            sent_id += 1
            max_sent += 1

            wordbigramm = makeBigrams(words)
            posbigramm = makeBigrams(pos)

            word_bow = self.word_dic.doc2bow(words)
            vec_word_tfidf = self.word_tfidf[word_bow]
            wordbigramm_bow = self.wordbigramm_dic.doc2bow(wordbigramm)
            vec_wordbigramm_tfidf = self.wordbigramm_tfidf[wordbigramm_bow]

            pos_bow = self.pos_dic.doc2bow(pos)
            vec_pos_tfidf = self.pos_tfidf[pos_bow]

            posbigramm_bow = self.posbigramm_dic.doc2bow(posbigramm)
            vec_posbigramm_tfidf = self.posbigramm_tfidf[posbigramm_bow]

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
            sent_info['vec_wordbigramm_tfidf'] = vec_wordbigramm_tfidf
            sent_info['vec_pos_tfidf'] = vec_pos_tfidf
            sent_info['vec_posbigramm_tfidf'] = vec_posbigramm_tfidf
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

            if 'wordunigramm' in self.feature_set:
                append_vec2data(feature_data['vec_word_tfidf'],
                                feature_data_array,
                                feature_row,
                                feature_col,
                                row_count,
                                vector_offset)
                vector_offset += self.word_vec_len

            if 'wordbigramm' in self.feature_set:
                append_vec2data(feature_data['vec_wordbigramm_tfidf'],
                                feature_data_array,
                                feature_row,
                                feature_col,
                                row_count,
                                vector_offset)
                vector_offset += self.wordbigramm_vec_len

            if 'posunigramm' in self.feature_set:
                append_vec2data(feature_data['vec_pos_tfidf'],
                                feature_data_array,
                                feature_row,
                                feature_col,
                                row_count,
                                vector_offset)
                vector_offset += self.pos_vec_len

            if 'posbigramm' in self.feature_set:
                append_vec2data(feature_data['vec_posbigramm_tfidf'],
                                feature_data_array,
                                feature_row,
                                feature_col,
                                row_count,
                                vector_offset)
                vector_offset += self.posbigramm_vec_len

            row_count += 1

        feature_row = np.array(feature_row)
        feature_col = np.array(feature_col)
        feature_data_array = np.array(feature_data_array)

        feature_vector = scipy.sparse.csc_matrix((feature_data_array, (feature_row, feature_col)),
                                                 shape=(row_count, self.vector_len))

        return feature_vector
