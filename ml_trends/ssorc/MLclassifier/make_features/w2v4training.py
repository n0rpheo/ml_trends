import os
import pickle

import mysql.connector
import numpy as np

from gensim.models import Word2Vec

from src.utils.corpora import TokenDocStream
from src.features.transformations import tokens_to_mean_w2v
from src.utils.selector import select_path_from_dir

feature_file_name = 'lr_MLclassifier_w2v_features'

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_word2vec_model = select_path_from_dir(os.path.join(path_to_db, 'models'),
                                              phrase="Select w2v-model: ",
                                              suffix=".model")
path_to_feature_file = os.path.join(path_to_db, 'features', feature_file_name + '.pickle')
path_to_target_file = os.path.join(path_to_db, 'features', feature_file_name + '_targets.pickle')

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = f"SELECT abstract_id, label FROM ml_topics_training"
cursor.execute(sq1)

abstract_ids = list()
abstract_labels_dict = dict()
for row in cursor:
    abstract_id = row[0]
    abstract_label = row[1]

    abstract_ids.append(abstract_id)
    abstract_labels_dict[abstract_id] = abstract_label
connection.close()

documents = TokenDocStream(abstracts=abstract_ids, token_type='originalText', print_status=True, output='all', lower=True)

features = list()
targets = list()

for abstract_id, document in documents:
    features.append(document)
    targets.append(abstract_labels_dict[abstract_id])

# load model
w2v_model = Word2Vec.load(path_to_word2vec_model)

w2v = {w: vec for w, vec in zip(w2v_model.wv.index2word, w2v_model.wv.syn0)}

transform_features = np.array([tokens_to_mean_w2v(w2v, words, 100) for words in features])

with open(path_to_feature_file, "wb") as feature_file:
    pickle.dump(transform_features, feature_file, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_to_target_file, "wb") as target_file:
    pickle.dump(targets, target_file, protocol=pickle.HIGHEST_PROTOCOL)