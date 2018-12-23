import os

import numpy as np
import scipy.sparse

import json
import gensim

import src.utils.corpora as corpora
from src.features.const import known_features
from src.features.const import known_labels

import src.utils.functions as utils
from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"


def load_concratings():
    filename = os.path.join(path_to_db, 'external', 'concreteness_ratings.txt')
    with open(filename) as file:
        cratings = file.readlines()

    crating = dict()
    for line in cratings[1:]:
        elements = line.split('\t')
        if elements[1] == "0":
            crating[elements[0]] = float(elements[2])

    return crating


def build_feature_file(dtype):
    dictionary_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/dictionaries')
    tfidf_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/tfidf')
    feature_file = os.path.join(dirname, '../../data/processed/' + dtype + '/features/ap_features.json')

    if os.path.isfile(feature_file):
        os.remove(feature_file)

    word_dic = gensim.corpora.Dictionary.load(os.path.join(dictionary_dir, 'word.dic'))
    wordbigram_dic = gensim.corpora.Dictionary.load(os.path.join(dictionary_dir, 'wordbigram.dic'))
    pos_dic = gensim.corpora.Dictionary.load(os.path.join(dictionary_dir, 'pos.dic'))
    posbigram_dic = gensim.corpora.Dictionary.load(os.path.join(dictionary_dir, 'posbigram.dic'))

    word_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'words_model.tfidf'))
    wordbigram_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'wordbigrams_model.tfidf'))
    pos_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'pos_model.tfidf'))
    posbigram_tfidf = gensim.models.TfidfModel.load(os.path.join(tfidf_dir, 'posbigrams_model.tfidf'))

    conc_rating = load_concratings()

    word_corpus = corpora.word_sent_stream(dtype)
    pos_corpus = corpora.pos_sent_stream(dtype)

    with open(feature_file, "a") as featfile:
        information = {}
        information['word_vec_len'] = len(word_dic)
        information['wordbigram_vec_len'] = len(wordbigram_dic)
        information['pos_vec_len'] = len(pos_dic)
        information['posbigram_vec_len'] = len(posbigram_dic)

        json_line = json.JSONEncoder().encode(information)
        featfile.write(json_line + '\n')

        sent_infos = []
        last_doc_id = None
        lt = LoopTimer(update_after=100)

        for word_sent, pos_sent in zip(word_corpus, pos_corpus):
            if word_sent[0] != pos_sent[0]:  # Checking if ids are the same
                continue

            doc_id = word_sent[0]
            pid = word_sent[1]

            words = word_sent[2]
            pos = pos_sent[2]
            wordbigrams = utils.makeBigrams(words)
            posbigrams = utils.makeBigrams(pos)

            word_bow = word_dic.doc2bow(words)
            vec_word_tfidf = word_tfidf[word_bow]

            wordbigram_bow = wordbigram_dic.doc2bow(wordbigrams)
            vec_wordbigram_tfidf = wordbigram_tfidf[wordbigram_bow]

            pos_bow = pos_dic.doc2bow(pos)
            vec_pos_tfidf = pos_tfidf[pos_bow]

            posbigram_bow = posbigram_dic.doc2bow(posbigrams)
            vec_posbigram_tfidf = posbigram_tfidf[posbigram_bow]

            # Collecting Concreteness-Ratings
            cr_min = 1000
            cr_max = 0
            cr_mean = 0

            cr_words = [cr_word for cr_word in words if cr_word in conc_rating]

            for word in cr_words:
                rating = conc_rating[word]
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

            if (last_doc_id is not None) and (last_doc_id != doc_id):
                max_sent = len(sent_infos)
                for sent_info in sent_infos:
                    sent_info['max_sent'] = max_sent
                    json_line = json.JSONEncoder().encode(sent_info)
                    featfile.write(json_line + '\n')
                sent_infos.clear()

            sent_info = dict()
            sent_info['cr_max_feature'] = cr_max_feature
            sent_info['cr_min_feature'] = cr_min_feature
            sent_info['cr_mean_feature'] = cr_mean_feature
            sent_info['vec_word_tfidf'] = vec_word_tfidf
            sent_info['vec_wordbigram_tfidf'] = vec_wordbigram_tfidf
            sent_info['vec_pos_tfidf'] = vec_pos_tfidf
            sent_info['vec_posbigram_tfidf'] = vec_posbigram_tfidf
            sent_info['id'] = doc_id
            sent_info['paragraphID'] = pid
            sent_info['sent_id'] = len(sent_infos)
            sent_infos.append(sent_info)

            last_doc_id = doc_id

            lt.update("Build AP Features")

        max_sent = len(sent_infos)
        for sent_info in sent_infos:
            sent_info['max_sent'] = max_sent
            json_line = json.JSONEncoder().encode(sent_info)
            featfile.write(json_line + '\n')


def get_features_from_file(num_samples,
                           dtype,
                           feature_set="wordunigram,wordbigram,concreteness,posunigram,posbigram",
                           file_suffix=""):
    # Loads Features and Targets from json
    # Saves them as Numpy/Scipy File

    dirname = os.path.dirname(__file__)
    feature_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/features')
    feature_file = os.path.join(feature_dir, 'ap_features.json')



    target_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/rf_targets')
    target_file = os.path.join(target_dir, 'targets.json')

    feature_scipy_file = os.path.join(feature_dir, 'ap_features_' + file_suffix + '.npz')
    target_np_file = os.path.join(target_dir, 'targets_' + file_suffix + '.npy')

    feature_set = feature_set.replace(' ', '').split(',')

    for feature_test in feature_set:
        if feature_test not in known_features:
            raise ValueError(feature_test + ' not in known_features')

    num_per_label_samples = int(num_samples / len(known_labels))
    label_counter = {}
    for l in known_labels:
        label_counter[l] = 0


    print("Load Targets")
    with open(target_file) as targetfile:
        targets = {}
        for target in targetfile:
            target = json.loads(target)
            id = target['id']
            pid = target['paragraphID']
            targets[(id, pid)] = target['rflabel']
    print("Done loading Targets")

    if os.path.isfile(feature_file):
        with open(feature_file) as ffile:
            info_line = ffile.readline()
            information = json.loads(info_line)
            word_vec_len = int(information['word_vec_len'])
            wordbigram_vec_len = int(information['wordbigram_vec_len'])
            pos_vec_len = int(information['pos_vec_len'])
            posbigram_vec_len = int(information['posbigram_vec_len'])

            #feature_vector = None
            target_vector = []

            feature_data_array = []
            feature_row = []
            feature_col = []

            vector_len = 0

            if 'location' in feature_set:
                vector_len += 1

            if 'concreteness' in feature_set:
                vector_len += 3

            if 'wordunigram' in feature_set:
                vector_len += word_vec_len

            if 'wordbigram' in feature_set:
                vector_len += wordbigram_vec_len

            if 'posunigram' in feature_set:
                vector_len += pos_vec_len

            if 'posbigram' in feature_set:
                vector_len += posbigram_vec_len

            row_count = 0
            lc = LoopTimer(update_after=100)

            for sample_count, feature_line in enumerate(ffile):
                feature_data = json.loads(feature_line)
                id = feature_data['id']
                pid = int(feature_data['paragraphID'])
                label_key = (id, pid)

                if label_key in targets:
                    target = targets[label_key]
                else:
                    continue

                break_sum = np.sum([1 for t in label_counter if label_counter[t] == num_per_label_samples])

                if break_sum == len(known_labels):
                    break

                if target in known_labels and label_counter[target] < num_per_label_samples:
                    label_counter[target] += 1
                    target_vector.append(target)

                    vector_offset = 0

                    if 'location' in feature_set:
                        sent_id = feature_data['sent_id']
                        max_sent = feature_data['max_sent']

                        feature_row.append(row_count)
                        feature_col.append(vector_offset)
                        feature_data_array.append(sent_id / max_sent)

                        vector_offset += 1

                    if 'concreteness' in feature_set:
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

                    if 'wordunigram' in feature_set:
                        utils.append_vec2data(feature_data['vec_word_tfidf'],
                                              feature_data_array,
                                              feature_row,
                                              feature_col,
                                              row_count,
                                              vector_offset)
                        vector_offset += word_vec_len

                    if 'wordbigram' in feature_set:
                        utils.append_vec2data(feature_data['vec_wordbigram_tfidf'],
                                              feature_data_array,
                                              feature_row,
                                              feature_col,
                                              row_count,
                                              vector_offset)
                        vector_offset += wordbigram_vec_len

                    if 'posunigram' in feature_set:
                        utils.append_vec2data(feature_data['vec_pos_tfidf'],
                                              feature_data_array,
                                              feature_row,
                                              feature_col,
                                              row_count,
                                              vector_offset)
                        vector_offset += pos_vec_len

                    if 'posbigram' in feature_set:
                        utils.append_vec2data(feature_data['vec_posbigram_tfidf'],
                                              feature_data_array,
                                              feature_row,
                                              feature_col,
                                              row_count,
                                              vector_offset)
                        vector_offset += posbigram_vec_len

                    row_count += 1

                all_sum = np.sum(label_counter)
                lc.update("Read Feature " + str(all_sum))

            feature_row = np.array(feature_row)
            feature_col = np.array(feature_col)
            feature_data_array = np.array(feature_data_array)

            feature_vector = scipy.sparse.csc_matrix((feature_data_array, (feature_row, feature_col)),
                                                     shape=(row_count, vector_len))

            target_vector = np.array(target_vector)

            scipy.sparse.save_npz(feature_scipy_file, feature_vector)
            np.save(target_np_file, target_vector)

            print()
            return target_vector, feature_vector
    print(str(feature_file) + " not found")