import os

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from stanfordcorenlp import StanfordCoreNLP
import gensim
import scipy.sparse
import numpy as np

import xml.etree.ElementTree as ET

from src.features.ap_features import load_concratings
from src.features.ap_features import get_features_from_file
from src.features.const import known_features
import src.utils.functions as util


class AbstractParser:
    def __init__(self, dtype, feature_set="location,concretenes,posunigram,posbigram"):
        self.feature_set = feature_set.replace(' ', '').split(',')

        for feature_test in self.feature_set:
            if feature_test not in known_features:
                raise ValueError(feature_test + ' not in known_features')

        dirname = os.path.dirname(__file__)
        dictionarie_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/dictionaries')
        tfidf_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/tfidf')

        self.word_dic = gensim.corpora.Dictionary.load(os.path.join(dictionarie_dir, 'word.dic'))
        self.wordbigram_dic = gensim.corpora.Dictionary.load(os.path.join(dictionarie_dir, 'wordbigram.dic'))
        self.pos_dic = gensim.corpora.Dictionary.load(os.path.join(dictionarie_dir, 'pos.dic'))
        self.posbigram_dic = gensim.corpora.Dictionary.load(os.path.join(dictionarie_dir, 'posbigram.dic'))

        self.word_vec_len = len(self.word_dic)
        self.wordbigram_vec_len = len(self.wordbigram_dic)
        self.pos_vec_len = len(self.pos_dic)
        self.posbigram_vec_len = len(self.posbigram_dic)

        self.word_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'words_model.tfidf'))
        self.wordbigram_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'wordbigrams_model.tfidf'))
        self.pos_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'pos_model.tfidf'))
        self.posbigram_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'posbigrams_model.tfidf'))

        self.conc_rating = load_concratings()

        self.nlp = StanfordCoreNLP('../stanford-corenlp-full-2018-02-27')
        self.props = {'annotators': 'tokenize,ssplit,pos,lemma', 'pipelineLanguage': 'en', 'outputFormat': 'xml'}

    def string_to_ssorc_feature_sparse_vec(self, input_string, sent_id, sent_num):
        annotation = self.nlp.annotate(input_string, properties=self.props)
        root = ET.fromstring(annotation)
        num_sent = 0

        for sentence in root.iter('sentence'):
            num_sent += 1

        if num_sent != 1:
            print("Sentence count is " + str(num_sent) + " instead of 1")
            return None

        words = []
        pos = []
        for token in root.iter('token'):
            words.append(token.find('word').text.lower())
            pos.append(token.find('POS').text)

        words_cleaned, pos_cleaned = util.posFilterString(words, pos)

        vector_len = 0

        if 'location' in self.feature_set:
            vector_len += 1

        if 'concreteness' in self.feature_set:
            vector_len += 3

        if 'wordunigram' in self.feature_set:
            vector_len += self.word_vec_len

        if 'wordbigram' in self.feature_set:
            vector_len += self.wordbigram_vec_len

        if 'posunigram' in self.feature_set:
            vector_len += self.pos_vec_len

        if 'posbigram' in self.feature_set:
            vector_len += self.posbigram_vec_len

        features = scipy.sparse.lil_matrix((1, vector_len))
        vector_offset = 0

        if 'location' in self.feature_set:
            location_feature = [sent_id / sent_num]

            features[0, 0] = location_feature
            vector_offset += 1

        if 'concreteness' in self.feature_set:
            # Collecting Concreteness-Ratings
            cr_min = 1000
            cr_max = 0
            cr_mean = 0

            cr_words = [cr_word for cr_word in words_cleaned if cr_word in self.conc_rating]

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

            features[0, vector_offset] = cr_max_feature
            features[0, vector_offset + 1] = cr_min_feature
            features[0, vector_offset + 2] = cr_mean_feature
            vector_offset += 3

        if 'wordunigram' in self.feature_set:
            word_bow = self.word_dic.doc2bow(words)
            vec_word_tfidf = self.word_tfidf[word_bow]

            util.add_vector_to_sparse_matrix(features, vec_word_tfidf, vector_offset)
            vector_offset += self.word_vec_len

        if 'wordbigram' in self.feature_set:
            wordbigram = util.makeBigrams(words_cleaned)
            wordbigram_bow = self.wordbigram_dic.doc2bow(wordbigram)
            vec_wordbigram_tfidf = self.wordbigram_tfidf[wordbigram_bow]

            util.add_vector_to_sparse_matrix(features, vec_wordbigram_tfidf, vector_offset)
            vector_offset += self.wordbigram_vec_len

        if 'posunigram' in self.feature_set:
            pos_bow = self.pos_dic.doc2bow(pos)
            vec_pos_tfidf = self.pos_tfidf[pos_bow]

            util.add_vector_to_sparse_matrix(features, vec_pos_tfidf, vector_offset)
            vector_offset += self.pos_vec_len

        if 'posbigram' in self.feature_set:
            posbigram = util.makeBigrams(pos_cleaned)
            posbigram_bow = self.posbigram_dic.doc2bow(posbigram)
            vec_posbigram_tfidf = self.posbigram_tfidf[posbigram_bow]

            util.add_vector_to_sparse_matrix(features, vec_posbigram_tfidf, vector_offset)
            vector_offset += self.posbigram_vec_len

        return features


def train_models(reg_paras,
                 dtype,
                 size=20000,
                 feature_set='location,concreteness,posunigram,posbigram',
                 load_features=False,
                 load_suffix=""):

    dirname = os.path.dirname(__file__)
    feature_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/features')
    target_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/rf_targets')
    feature_file = os.path.join(feature_dir, 'ap_features_' + load_suffix + '.npz')
    target_file = os.path.join(target_dir, 'targets_' + load_suffix + '.npy')

    if load_features == False:
        # Read Features from File
        all_targets, all_features = util.measure(lambda: get_features_from_file(size, dtype=dtype,
                                                                                feature_set=feature_set),
                                                 "Read Features")

        scipy.sparse.save_npz(feature_file, all_features)
        np.save(target_file, all_targets)
    else:
        all_features = scipy.sparse.load_npz(feature_file)
        all_targets = np.load(target_file)

    print("Feature-Vector-Shape: " + str(all_features.shape))

    learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                                all_targets,
                                                                                                test_size=0.4,
                                                                                                random_state=42,
                                                                                                shuffle=True)

    top_score = 0

    for c_para in reg_paras:
        model = svm.SVC(decision_function_shape='ovr', C=c_para, kernel='rbf')
        scores = util.measure(lambda: cross_val_score(model, learning_features, learning_targets, cv=10, n_jobs=-1),
                              "Cross Val on C = " + str(c_para))
        print("Score: " + str(scores.mean()))
        if scores.mean() > top_score:
            top_score = scores.mean()
            best_model = model
            best_reg_para = c_para

    print()
    print("Best Reg-Para: " + str(best_reg_para))
    print()

    best_model.fit(learning_features, learning_targets)
    print_result(best_model, holdback_features, holdback_targets)

    return best_model


def print_result(model, hb_features, hb_targets):
    prediction = model.predict(hb_features)

    labels = list(set(hb_targets))

    relevant = {}
    retrieved = {}
    rel_ret = {}

    for label in labels:
        relevant[label] = 0
        retrieved[label] = 0
        rel_ret[label] = 0

    print("Occurences of Labels (Relevant | Retrieved)")
    for label in labels:
        relevant[label] = np.sum(hb_targets == label)
        retrieved[label] = np.sum(prediction == label)

        print(label[0:5] + ": " + str(relevant[label]) + " | " + str(retrieved[label]))

    for i in range(0, len(hb_targets)):
        label = hb_targets[i]
        if hb_targets[i] == prediction[i]:
            rel_ret[label] += 1

    print()
    print("[Results] - Score: " + str(model.score(hb_features, hb_targets)))
    print("     \tPreci | Recall")
    for label in labels:
        if retrieved[label] == 0:
            prec = 0.0
        else:
            prec = rel_ret[label] / retrieved[label]
        recall = rel_ret[label] / relevant[label]
        print(label[0:5] + ":\t" + str(prec)[0:5] + " | " + str(recall)[0:5])