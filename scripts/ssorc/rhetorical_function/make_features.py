import os
import mysql.connector
import numpy as np
import pickle
import scipy.sparse
import gensim

from src.utils.corpora import TokenSentenceStream
import src.utils.functions as utils
from src.utils.LoopTimer import LoopTimer
from src.utils.selector import select_path_from_dir

feat_file_name = "rf_medium_lcpupbwuwb"

path_to_db = "/media/norpheo/mySQL/db/ssorc"
feature_scipy_file = os.path.join(path_to_db, "features", f"{feat_file_name}_features.npz")
target_np_file = os.path.join(path_to_db, "features", f"{feat_file_name}_targets.npy")

#feature_set = "location concreteness wordunigramm wordbigramm posunigramm posbigramm".split()
feature_set = "location concreteness posunigramm posbigramm wordunigramm wordbigramm".split()
limit = 5000


dictionary_dir = os.path.join(path_to_db, 'dictionaries')
model_dir = os.path.join(path_to_db, 'models')
target_path = os.path.join(path_to_db, 'features', 'rf_targets.pickle')

word_dic = gensim.corpora.Dictionary.load(select_path_from_dir(dictionary_dir,
                                                               phrase="Select Word Dict: ",
                                                               suffix=".dict",
                                                               preselection="pruned_word_ml.dict"))
wordbigramm_dic = gensim.corpora.Dictionary.load(select_path_from_dir(dictionary_dir,
                                                                      phrase="Select Word-bigramm Dict: ",
                                                                      suffix=".dict",
                                                                      preselection="pruned_wordbigramm_ml.dict"))
pos_dic = gensim.corpora.Dictionary.load(select_path_from_dir(dictionary_dir,
                                                              phrase="Select POS Dict: ",
                                                              suffix=".dict",
                                                              preselection="full_pos_ml.dict"))
posbigramm_dic = gensim.corpora.Dictionary.load(select_path_from_dir(dictionary_dir,
                                                                     phrase="Select POS-bigram Dict: ",
                                                                     suffix=".dict",
                                                                     preselection="full_posbigramm_ml.dict"))

word_tfidf = gensim.models.TfidfModel.load(select_path_from_dir(model_dir,
                                                                phrase="Select Word-TFIDF Model: ",
                                                                suffix=".tfidf",
                                                                preselection="pruned_word_ml.tfidf"))
wordbigramm_tfidf = gensim.models.TfidfModel.load(select_path_from_dir(model_dir,
                                                                       phrase="Select Word-Bigramm-TFIDF Model: ",
                                                                       suffix=".tfidf",
                                                                       preselection="pruned_wordbigramm_ml.tfidf"))
pos_tfidf = gensim.models.TfidfModel.load(select_path_from_dir(model_dir,
                                                               phrase="Select POS-TFIDF Model: ",
                                                               suffix=".tfidf",
                                                               preselection="full_pos_ml.tfidf"))
posbigramm_tfidf = gensim.models.TfidfModel.load(select_path_from_dir(model_dir,
                                                                      phrase="Select POS-Bigramm-TFIDF Model: ",
                                                                      suffix=".tfidf",
                                                                      preselection="full_posbigramm_ml.tfidf"))

word_vec_len = len(word_dic)
wordbigramm_vec_len = len(wordbigramm_dic)
pos_vec_len = len(pos_dic)
posbigramm_vec_len = len(posbigramm_dic)

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

if 'location' in feature_set:
    vector_len += 1

if 'concreteness' in feature_set:
    vector_len += 3

if 'wordunigramm' in feature_set:
    vector_len += word_vec_len

if 'wordbigramm' in feature_set:
    vector_len += wordbigramm_vec_len

if 'posunigramm' in feature_set:
    vector_len += pos_vec_len

if 'posbigramm' in feature_set:
    vector_len += posbigramm_vec_len

connection = mysql.connector.connect(host="localhost",
                                     user="root",
                                     passwd="thesis")

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id FROM abstracts_ml WHERE entities LIKE '%machine learning%' AND annotated=1"
cursor.execute(sq1)

lc = LoopTimer(update_after=1000, avg_length=5000)
abstracts = set()
for row in cursor:
    abstracts.add(row[0])
    lc.update("Collect Abstracts to Process")
connection.close()

word_corpus = TokenSentenceStream(abstracts=abstracts,
                                  token_type='word',
                                  token_cleaned=0,
                                  output='all',
                                  lower=True)

pos_corpus = TokenSentenceStream(abstracts=abstracts,
                                 token_type='pos',
                                 token_cleaned=0,
                                 output='all',
                                 lower=True)

with open(target_path, 'rb') as target_file:
    label_dic = pickle.load(target_file)

label_count = dict()
for lkey in label_dic:
    label = label_dic[lkey]
    if label not in label_count:
        label_count[label] = 0

label_limit = limit / len(label_count)

last_doc_id = None
lc = LoopTimer(update_after=5, avg_length=1000)
sent_infos = list()
max_sent = 0

for word_sent, pos_sent in zip(word_corpus, pos_corpus):
    doc_id = word_sent[0]
    sent_id = word_sent[1]

    if (last_doc_id is not None) and (last_doc_id != doc_id):
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

            if 'wordunigramm' in feature_set:
                utils.append_vec2data(feature_data['vec_word_tfidf'],
                                      feature_data_array,
                                      feature_row,
                                      feature_col,
                                      row_count,
                                      vector_offset)
                vector_offset += word_vec_len

            if 'wordbigramm' in feature_set:
                utils.append_vec2data(feature_data['vec_wordbigram_tfidf'],
                                      feature_data_array,
                                      feature_row,
                                      feature_col,
                                      row_count,
                                      vector_offset)
                vector_offset += wordbigramm_vec_len

            if 'posunigramm' in feature_set:
                utils.append_vec2data(feature_data['vec_pos_tfidf'],
                                      feature_data_array,
                                      feature_row,
                                      feature_col,
                                      row_count,
                                      vector_offset)
                vector_offset += pos_vec_len

            if 'posbigramm' in feature_set:
                utils.append_vec2data(feature_data['vec_posbigram_tfidf'],
                                      feature_data_array,
                                      feature_row,
                                      feature_col,
                                      row_count,
                                      vector_offset)
                vector_offset += posbigramm_vec_len

            row_count += 1

        max_sent = 0
        sent_infos.clear()

    last_doc_id = doc_id
    max_sent += 1
    label_key = (doc_id, sent_id)

    if (label_key in label_dic) and (label_count[label_dic[label_key]] < label_limit):
        words = word_sent[2]
        pos = pos_sent[2]
        wordbigramm = utils.makeBigrams(words)
        posbigramm = utils.makeBigrams(pos)

        word_bow = word_dic.doc2bow(words)
        vec_word_tfidf = word_tfidf[word_bow]
        wordbigramm_bow = wordbigramm_dic.doc2bow(wordbigramm)
        vec_wordbigramm_tfidf = wordbigramm_tfidf[wordbigramm_bow]

        pos_bow = pos_dic.doc2bow(pos)
        vec_pos_tfidf = pos_tfidf[pos_bow]

        posbigramm_bow = posbigramm_dic.doc2bow(posbigramm)
        vec_posbigramm_tfidf = posbigramm_tfidf[posbigramm_bow]

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

        sent_info = dict()
        sent_info['cr_max_feature'] = cr_max_feature
        sent_info['cr_min_feature'] = cr_min_feature
        sent_info['cr_mean_feature'] = cr_mean_feature
        sent_info['vec_word_tfidf'] = vec_word_tfidf
        sent_info['vec_wordbigramm_tfidf'] = vec_wordbigramm_tfidf
        sent_info['vec_pos_tfidf'] = vec_pos_tfidf
        sent_info['vec_posbigramm_tfidf'] = vec_posbigramm_tfidf
        sent_info['id'] = doc_id
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
        if breaker == 1:
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

    if 'wordunigramm' in feature_set:
        utils.append_vec2data(feature_data['vec_word_tfidf'],
                              feature_data_array,
                              feature_row,
                              feature_col,
                              row_count,
                              vector_offset)
        vector_offset += word_vec_len

    if 'wordbigramm' in feature_set:
        utils.append_vec2data(feature_data['vec_wordbigramm_tfidf'],
                              feature_data_array,
                              feature_row,
                              feature_col,
                              row_count,
                              vector_offset)
        vector_offset += wordbigramm_vec_len

    if 'posunigramm' in feature_set:
        utils.append_vec2data(feature_data['vec_pos_tfidf'],
                              feature_data_array,
                              feature_row,
                              feature_col,
                              row_count,
                              vector_offset)
        vector_offset += pos_vec_len

    if 'posbigramm' in feature_set:
        utils.append_vec2data(feature_data['vec_posbigramm_tfidf'],
                              feature_data_array,
                              feature_row,
                              feature_col,
                              row_count,
                              vector_offset)
        vector_offset += posbigramm_vec_len

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

scipy.sparse.save_npz(feature_scipy_file, feature_vector)
np.save(target_np_file, target_vector)
