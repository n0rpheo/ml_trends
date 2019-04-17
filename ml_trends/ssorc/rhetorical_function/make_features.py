import os
import numpy as np
import pickle
import scipy.sparse
import gensim
import pandas as pd

import src.utils.functions as utils
from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"

feat_file_name = "rf_pruned_features"  # output_name

target_path = os.path.join(path_to_db, 'features', 'rf_targets.pickle')  # Target Input

worddb_path = os.path.join(path_to_db, 'pandas', 'ml_word.pandas')
posdb_path = os.path.join(path_to_db, 'pandas', 'ml_pos.pandas')

dictionary_dir = os.path.join(path_to_db, 'dictionaries')
model_dir = os.path.join(path_to_db, 'models')

feature_file_path = os.path.join(path_to_db, "features", f"{feat_file_name}.pickle")  # OUTPUT!!!

settings = dict()
settings["feature_set"] = ["location",
                           "concreteness",
                           "posunigram",
                           "posbigram",
                           "wordunigram",
                           "wordbigram"]
settings["word_dic"] = os.path.join(dictionary_dir, "pruned_word_lower_pd.dict")
settings["wordbigram_dic"] = os.path.join(dictionary_dir, "pruned_wordbigram_lower_pd.dict")
settings["pos_dic"] = os.path.join(dictionary_dir, "pruned_pos_lower_pd.dict")
settings["posbigram_dic"] = os.path.join(dictionary_dir, "pruned_posbigram_lower_pd.dict")
settings["word_tfidf"] = os.path.join(model_dir, "pruned_word_lower_pd.tfidf")
settings["wordbigram_tfidf"] = os.path.join(model_dir, "pruned_wordbigram_lower_pd.tfidf")
settings["pos_tfidf"] = os.path.join(model_dir, "pruned_pos_lower_pd.tfidf")
settings["posbigram_tfidf"] = os.path.join(model_dir, "pruned_posbigram_lower_pd.tfidf")

# limit = math.inf
limit = 5*3300

print("Loading Panda DB")
wordDF = pd.read_pickle(worddb_path)
posDF = pd.read_pickle(posdb_path)
print("Done Loading")
df = wordDF.join(posDF)

word_dic = gensim.corpora.Dictionary.load(settings["word_dic"])
wordbigram_dic = gensim.corpora.Dictionary.load(settings["wordbigram_dic"])
pos_dic = gensim.corpora.Dictionary.load(settings["pos_dic"])
posbigram_dic = gensim.corpora.Dictionary.load(settings["posbigram_dic"])

word_tfidf = gensim.models.TfidfModel.load(settings["word_tfidf"])
wordbigram_tfidf = gensim.models.TfidfModel.load(settings["wordbigram_tfidf"])
pos_tfidf = gensim.models.TfidfModel.load(settings["pos_tfidf"])
posbigram_tfidf = gensim.models.TfidfModel.load(settings["posbigram_tfidf"])

word_vec_len = len(word_dic)
wordbigram_vec_len = len(wordbigram_dic)
pos_vec_len = len(pos_dic)
posbigram_vec_len = len(posbigram_dic)

conc_file_path = os.path.join(path_to_db, 'external', 'concreteness_ratings.txt')
with open(conc_file_path) as conc_file:
    cratings = conc_file.readlines()

conc_rating = dict()
for line in cratings[1:]:
    elements = line.split('\t')
    if elements[1] == "0":
        conc_rating[elements[0]] = float(elements[2])

target_vector = []

feature_data_array = []
feature_row = []
feature_col = []

vector_len = 0
row_count = 0

feature_set = settings['feature_set']

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


with open(target_path, 'rb') as target_file:
    label_dic = pickle.load(target_file)

label_count = dict()
for lkey in label_dic:
    label = label_dic[lkey]
    if label not in label_count:
        label_count[label] = 0

label_limit = limit / len(label_count)

last_abstract_id = None
lc = LoopTimer(update_after=5, avg_length=1000)
sent_infos = list()
max_sent = 0
breaker = 0

for abstract_id, row in df.iterrows():
    word_sentence_tokens = [sentence.split(" ") for sentence in row['word'].split("\t")]
    pos_sentence_tokens = [sentence.split(" ") for sentence in row['pos'].split("\t")]

    for sent_id, (word_tokens, pos_tokens) in enumerate(zip(word_sentence_tokens, pos_sentence_tokens)):

        if (last_abstract_id is not None) and (last_abstract_id != abstract_id):
            for feature_data in sent_infos:
                did = feature_data['id']
                sid = feature_data['sent_id']
                label_key = (did, sid)

                target = label_dic[label_key]

                target_vector.append(target)

                vector_offset = 0

                if 'location' in feature_set:
                    feature_row.append(row_count)
                    feature_col.append(vector_offset)
                    feature_data_array.append(sid / max_sent)

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

            max_sent = 0
            sent_infos.clear()

        last_abstract_id = abstract_id
        max_sent += 1
        label_key = (abstract_id, sent_id)

        if (label_key in label_dic) and (label_count[label_dic[label_key]] < label_limit):
            wordbigram = utils.makeBigrams(word_tokens)
            posbigram = utils.makeBigrams(pos_tokens)

            word_bow = word_dic.doc2bow(word_tokens)
            vec_word_tfidf = word_tfidf[word_bow]
            wordbigram_bow = wordbigram_dic.doc2bow(wordbigram)
            vec_wordbigram_tfidf = wordbigram_tfidf[wordbigram_bow]

            pos_bow = pos_dic.doc2bow(pos_tokens)
            vec_pos_tfidf = pos_tfidf[pos_bow]

            posbigram_bow = posbigram_dic.doc2bow(posbigram)
            vec_posbigram_tfidf = posbigram_tfidf[posbigram_bow]

            # Collecting Concreteness-Ratings
            cr_min = 1000
            cr_max = 0
            cr_mean = 0

            cr_words = [cr_word for cr_word in word_tokens if cr_word in conc_rating]

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

            sent_info = dict()
            sent_info['cr_max_feature'] = cr_max_feature
            sent_info['cr_min_feature'] = cr_min_feature
            sent_info['cr_mean_feature'] = cr_mean_feature
            sent_info['vec_word_tfidf'] = vec_word_tfidf
            sent_info['vec_wordbigram_tfidf'] = vec_wordbigram_tfidf
            sent_info['vec_pos_tfidf'] = vec_pos_tfidf
            sent_info['vec_posbigram_tfidf'] = vec_posbigram_tfidf
            sent_info['id'] = abstract_id
            sent_info['sent_id'] = sent_id
            sent_infos.append(sent_info)

            print_string = ""

            label_count[label_dic[label_key]] += 1
            breaker = 0
            for lco in label_count:
                counting = label_count[lco]
                if counting >= label_limit:
                    breaker += 1
                print_string += f"{lco}: {counting} | "
            breaker = breaker / len(label_count)
            #print_string = print_string[:len(print_string)-3]
            print_string += f"Breaker: {breaker}"
            lc.update(f"Build AP Features | {print_string}")
            if breaker >= 1:
                break

    if breaker >= 1:
        break

for feature_data in sent_infos:
    did = feature_data['id']
    sid = feature_data['sent_id']
    label_key = (did, sid)

    target = label_dic[label_key]

    target_vector.append(target)

    vector_offset = 0

    if 'location' in feature_set:
        feature_row.append(row_count)
        feature_col.append(vector_offset)
        feature_data_array.append(sid / max_sent)

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

feature_row = np.array(feature_row)
feature_col = np.array(feature_col)
feature_data_array = np.array(feature_data_array)

feature_vector = scipy.sparse.csc_matrix((feature_data_array, (feature_row, feature_col)),
                                         shape=(row_count, vector_len))

target_vector = np.array(target_vector)

print()
print(feature_vector.shape)
print(target_vector.shape)

feature_dict = dict()

feature_dict["features"] = feature_vector
feature_dict["targets"] = target_vector
feature_dict["settings"] = settings

with open(feature_file_path, "wb") as feature_file:
    pickle.dump(feature_dict, feature_file)
