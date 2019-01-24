import os

import gensim
import scipy.sparse
import numpy as np
import mysql.connector

import src.utils.corpora as corpora
from src.utils.LoopTimer import LoopTimer
from src.utils.selector import select_path_from_dir


token_type = 'originalText'
feature_file_name = 'tm_features.npz'
dic_temp_file_name = "tm_dictionary.dict"
num_samples = 100000

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_feature_file = os.path.join(path_to_db, 'features', feature_file_name)
path_to_dictionary = select_path_from_dir(os.path.join(path_to_db, 'dictionaries'),
                                          phrase="Select Dictionary: ",
                                          suffix=".dict")
path_to_temp_dic = os.path.join(path_to_db, 'dictionaries', dic_temp_file_name)

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(path_to_dictionary)

print('Load Data')
connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = f"SELECT abstract_id FROM abstracts_ml WHERE entities LIKE '%machine learning%' and annotated=1"
#sq1 = f"SELECT abstract_id FROM abstracts WHERE isML=1"
cursor.execute(sq1)

abstracts = set()
for row in cursor:
    abstracts.add(row[0])
connection.close()

corpus = corpora.TokenDocStream(abstracts=abstracts,
                                token_type=token_type,
                                print_status=False,
                                token_cleaned=False,
                                output='all',
                                lower=False)
row = []
col = []
data = []
lt = LoopTimer(update_after=10, avg_length=200, target=len(abstracts))
for idx, document in enumerate(corpus):
    if num_samples != -1 and idx == num_samples:
        break

    bow = dictionary.doc2bow(document[1])

    for entry in bow:
        row.append(idx)
        col.append(entry[0])
        data.append(entry[1])
    lt.update("Build Features")

m = idx + 1
n = len(dictionary)

row = np.array(row)
col = np.array(col)
data = np.array(data)

feature_vector = scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n))

scipy.sparse.save_npz(path_to_feature_file, feature_vector)
dictionary.save(path_to_temp_dic)
