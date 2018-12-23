import os

import gensim
import scipy.sparse
import numpy as np
import mysql.connector

import src.utils.corpora as corpora
from src.utils.LoopTimer import LoopTimer
from src.utils.selector import select_path_from_dir

feature_file_name = 'lr_MLclassifier_bow_features'

token_type = 'originalText'

path_to_db = "/media/norpheo/mySQL/db/ssorc"
dic_path = select_path_from_dir(os.path.join(path_to_db, "dictionaries"),
                                phrase="Select Dictionary: ")
tfidf_path = select_path_from_dir(os.path.join(path_to_db, "models"),
                                  phrase="Select TFIDF Model: ",
                                  suffix=".tfidf")

path_to_feature_file = os.path.join(path_to_db, 'features', f"{feature_file_name}.npz")
path_to_target_file = os.path.join(path_to_db, 'features', f"{feature_file_name}_targets.npy")

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(dic_path)
print('Load TFIDF')
tfidf = gensim.models.TfidfModel.load(tfidf_path)

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = f"SELECT abstract_id, label FROM ml_topics_training"
cursor.execute(sq1)

abstracts = set()
abstract_labels = dict()
for idx, row in enumerate(cursor):
    abstract_id = row[0]
    abstract_label = row[1]

    abstracts.add(abstract_id)
    abstract_labels[abstract_id] = abstract_label
connection.close()

corpus = corpora.TokenDocStream(abstracts=abstracts, token_type=token_type, print_status=True, output='all', lower=True)
row = []
col = []
data = []
lt = LoopTimer()

labels = list()

for idx, document in enumerate(corpus):
    words = document[1]
    abstract_id = document[0]

    label = abstract_labels[abstract_id]
    labels.append(label)

    bow = dictionary.doc2bow(words)
    vec_tfidf = tfidf[bow]

    for entry in vec_tfidf:
        row.append(idx)
        col.append(entry[0])
        data.append(entry[1])
    lt.update("Build Features")

m = idx + 1
n = len(dictionary)

row = np.array(row)
col = np.array(col)
data = np.array(data)

labels = np.array(labels)
feature_vector = scipy.sparse.csc_matrix((data, (row, col)), shape=(m, n))

print(feature_vector.shape)
print(labels.shape)

scipy.sparse.save_npz(path_to_feature_file, feature_vector)
np.save(path_to_target_file, labels)
