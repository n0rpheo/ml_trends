import os
import pickle
import mysql.connector

from sklearn.externals import joblib
from gensim.models import Word2Vec

from src.utils.LoopTimer import LoopTimer
from src.utils.selector import select_path_from_dir

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_mllr_model = os.path.join(path_to_db, 'models', 'mllr.joblib')
path_to_mlsvc_model = os.path.join(path_to_db, 'models', 'mlsvc.joblib')
path_to_word2vec_model = select_path_from_dir(os.path.join(path_to_db, 'models'),
                                              phrase="Select w2v-model: ",
                                              suffix=".model")

mllr = joblib.load(path_to_mllr_model)
w2v_model = Word2Vec.load(path_to_word2vec_model)
w2v = {w: vec for w, vec in zip(w2v_model.wv.index2word, w2v_model.wv.syn0)}
path_to_feature_file = select_path_from_dir(os.path.join(path_to_db, 'features'),
                                            phrase="Select Feature File: ",
                                            suffix='.pickle')

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )
cursor = connection.cursor()
cursor.execute("USE ssorc;")


with open(path_to_feature_file, "rb") as feature_file:
    documents = pickle.load(feature_file)

pos = 0
neg = 0
lc = LoopTimer(update_after=50, avg_length=700, target=len(documents))
for abstract_id in documents:
    features = documents[abstract_id]
    prediction = mllr.predict_proba(features)

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

connection.commit()
connection.close()
