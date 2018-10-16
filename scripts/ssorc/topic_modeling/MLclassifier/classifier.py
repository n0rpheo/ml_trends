import os

import mysql.connector
import numpy as np

from sklearn.externals import joblib
from gensim.models import Word2Vec

from src.utils.LoopTimer import LoopTimer
from src.utils.corpora import TokenDocStream
from src.features.transformations import tokens_to_mean_w2v

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_mllr_model = os.path.join(path_to_db, 'models', 'mllr.joblib')
path_to_mlsvc_model = os.path.join(path_to_db, 'models', 'mlsvc.joblib')
path_to_word2vec_model = os.path.join(path_to_db, 'models', 'word2vec.model')

mllr = joblib.load(path_to_mllr_model)
mlsvc = joblib.load(path_to_mlsvc_model)

w2v_model = Word2Vec.load(path_to_word2vec_model)
w2v = {w: vec for w, vec in zip(w2v_model.wv.index2word, w2v_model.wv.syn0)}

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

documents = TokenDocStream(abstracts=abstract_ids, token_type='word', print_status=False, output='all')

pos = 0
neg = 0

for abstract_id, tokens in documents:
    transform_features = np.array([tokens_to_mean_w2v(w2v, tokens)])

    prediction = mllr.predict_proba(transform_features)

    if prediction[0][1] > 0.9:
        pos += 1
    else:
        neg += 1

print(f"POS: {pos}")
print(f"NEG: {neg}")
