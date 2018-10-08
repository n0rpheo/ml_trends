import os

import gensim
import scipy.sparse
import numpy as np
import mysql.connector

import src.utils.corpora as corpora
from src.utils.LoopTimer import LoopTimer

feature_file_name = 'tm_features'
token_type = 'word'
num_samples = 1000

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")
path_to_annotations = os.path.join(path_to_db, 'annotations')
path_to_models = os.path.join(path_to_db, 'models')
path_to_feature_file = os.path.join(path_to_db, 'features', feature_file_name + '.npz')
dic_path = os.path.join(path_to_dictionaries, "full_" + token_type + ".dict")
tfidf_path = os.path.join(path_to_models, token_type + "_model.tfidf")

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
sq1 = f"SELECT abstract_id FROM abstracts WHERE annotated=1 and dictionaried=1 LIMIT {num_samples}"
cursor.execute(sq1)

abstracts = set()
for row in cursor:
    abstracts.add(row[0])
connection.close()

corpus = corpora.TokenDocStream(token_type, abstracts=abstracts, print_status=True)
row = []
col = []
data = []
lt = LoopTimer()
for idx, document in enumerate(corpus):
    if num_samples != -1 and idx == num_samples:
        break

    bow = dictionary.doc2bow(document[1])
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

feature_vector = scipy.sparse.csc_matrix((data, (row, col)), shape=(m, n))
scipy.sparse.save_npz(path_to_feature_file, feature_vector)
