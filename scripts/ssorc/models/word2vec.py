import os

import mysql.connector
from gensim.models import Word2Vec

from src.utils.corpora import TokenSentenceStream

w2v_model_name = 'word2vec_isML.model'

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_word2vec_model = os.path.join(path_to_db, 'models', w2v_model_name)

# define training data
connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
#sq1 = f"SELECT abstract_id FROM abstracts WHERE annotated=1 AND dictionaried=1 LIMIT 100000"
sq1 = f"SELECT abstract_id FROM mlabstracts"
cursor.execute(sq1)

abstracts = set()
for row in cursor:
    abstract_id = row[0]
    abstracts.add(abstract_id)
connection.close()

sentences = TokenSentenceStream(abstracts=abstracts, token_type='word', print_status=True, lower=True)

# train model
print("Train Model")
model = Word2Vec(sentences, min_count=1, size=100, window=5, workers=8, sg=0)
print("Model Trained")
# save model
print("Save Model")
model.save(path_to_word2vec_model)
