import os
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import gensim
import scipy.sparse
import pickle

num_topics = 20
update_every = 1
passes = 1
model_name = "tm_lda_lemma.model"

path_to_db = "/media/norpheo/mySQL/db/ssorc"
model_file_name = "tm_lda_500topics.pickle"

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(os.path.join(path_to_db, "dictionaries", "tm_dictionary.dict"))
print('Load Features')
tm_features = scipy.sparse.load_npz(os.path.join(path_to_db, 'features', 'tm_features.npz'))
#tm_features = tm_features.transpose()


lda_model = LatentDirichletAllocation(n_components=500,
                                      random_state=0,
                                      learning_method='batch',
                                      verbose=1,
                                      max_iter=100)
print('Training Model.')
lda_model.fit(tm_features)

n_top_words = 5
topic_words = {}
for topic, comp in enumerate(lda_model.components_):
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words[topic] = [dictionary[i] for i in word_idx]

for topic, words in topic_words.items():
    word_string = "\t".join(words)
    print(f'Topic: {topic}: {word_string}')

print('Model Trained.')
with open(os.path.join(path_to_db, "models", model_file_name), "wb") as model_file:
    pickle.dump(lda_model, model_file)
print('Model Saved.')
