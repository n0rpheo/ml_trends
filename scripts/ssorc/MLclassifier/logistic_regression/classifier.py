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
#mlsvc = joblib.load(path_to_mlsvc_model)

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


documents = TokenDocStream(abstracts=abstract_ids, token_type='word', print_status=False, output='all', lower=True)

pos = 0
neg = 0
lc = LoopTimer(update_after=50, avg_length=700, target=len(abstract_ids))
for abstract_id, tokens in documents:
    transform_features = np.array([tokens_to_mean_w2v(w2v, tokens, 100)])

    prediction = mllr.predict_proba(transform_features)

    if prediction[0][1] > 0.8:
        label = 1
        pos += 1
    else:
        label = -1
        neg += 1

    sql = f'UPDATE abstracts SET isML = {label} WHERE abstract_id = "{abstract_id}"'
    cursor.execute(sql)
    lc.update("Classify")

print()
print(f"POS: {pos}")
print(f"NEG: {neg}")

#connection.commit()
connection.close()
