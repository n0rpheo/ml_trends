import os
import pickle

import mysql.connector
import numpy as np

from gensim.models import Word2Vec

from src.utils.corpora import TokenDocStream
from src.features.transformations import tokens_to_mean_w2v
from src.utils.selector import select_path_from_dir

feature_file_name = 'MLclassifier_w2v4classification.pickle'

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_word2vec_model = select_path_from_dir(os.path.join(path_to_db, 'models'),
                                              phrase="Select w2v-model: ",
                                              suffix=".model")
path_to_feature_file = os.path.join(path_to_db, 'features', feature_file_name)

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = f"SELECT abstract_id FROM mlabstracts"
cursor.execute(sq1)

abstract_ids = list()
for row in cursor:
    abstract_id = row[0]

    abstract_ids.append(abstract_id)
connection.close()

# load model
w2v_model = Word2Vec.load(path_to_word2vec_model)
w2v = {w: vec for w, vec in zip(w2v_model.wv.index2word, w2v_model.wv.syn0)}

documents = TokenDocStream(abstracts=abstract_ids, token_type='originalText', print_status=True, output='all', lower=True)
docs = dict()
for abstract_id, tokens in documents:
    docs[abstract_id] = np.array([tokens_to_mean_w2v(w2v, tokens, 100)])


with open(path_to_feature_file, "wb") as feature_file:
    pickle.dump(docs, feature_file, protocol=pickle.HIGHEST_PROTOCOL)