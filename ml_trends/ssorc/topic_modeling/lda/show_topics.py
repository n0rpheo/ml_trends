import os
import gensim
import pickle
import numpy as np

path_to_db = "/media/norpheo/mySQL/db/ssorc"
model_file_name = "tm_lda_500topics.pickle"

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(os.path.join(path_to_db, "dictionaries", "tm_dictionary.dict"))

with open(os.path.join(path_to_db, "models", model_file_name), "rb") as model_file:
    lda_model = pickle.load(model_file)

n_top_words = 5
topic_words = {}
for topic, comp in enumerate(lda_model.components_):
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words[topic] = [dictionary[i] for i in word_idx]

for topic, words in topic_words.items():
    word_string = "\t".join(words)
    print(f'Topic: {topic}: {word_string}')
