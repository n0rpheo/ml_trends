import os

import gensim
import scipy.sparse
import numpy as np
import mysql.connector

import src.utils.corpora as corpora
from src.utils.LoopTimer import LoopTimer
from src.utils.selector import select_path_from_dir

feature_file_name = 'mlc_bow_clfdata'

dic_file_name = "pruned_originalText_potML.dict"
tfidf_file_name = "pruned_oT_potML.tfidf"
token_type = 'originalText'

path_to_db = "/media/norpheo/mySQL/db/ssorc"
dic_path = select_path_from_dir(os.path.join(path_to_db, "dictionaries"),
                                phrase="Select Dictionary: ")
tfidf_path = select_path_from_dir(os.path.join(path_to_db, "models"),
                                  phrase="Select TFIDF Model: ",
                                  suffix=".tfidf")

path_to_feature_file = os.path.join(path_to_db, 'features', f"{feature_file_name}.npz")

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
sq1 = f"SELECT abstract_id FROM mlabstracts"
cursor.execute(sq1)

abstracts = set()
for idx, row in enumerate(cursor):
    abstract_id = row[0]

    abstracts.add(abstract_id)
connection.close()

corpus = corpora.TokenDocStream(abstracts=abstracts, token_type=token_type, print_status=True, output='all', lower=True)
row = []
col = []
data = []
lt = LoopTimer()

for idx, document in enumerate(corpus):
    words = document[1]
    abstract_id = document[0]

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

feature_vector = scipy.sparse.csc_matrix((data, (row, col)), shape=(m, n))

print(feature_vector.shape)

scipy.sparse.save_npz(path_to_feature_file, feature_vector)
