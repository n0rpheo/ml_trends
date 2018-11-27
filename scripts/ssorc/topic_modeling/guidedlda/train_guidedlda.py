import os
import gensim
import numpy as np
import scipy.sparse
import guidedlda
import pickle

token_type = 'originalText'

dic_file_name = "pruned_originalText_isML.dict"
model_file_name = "tm_glda_model.pickle"

update_every = 1
passes = 1

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_models = os.path.join(path_to_db, 'models')
path_to_dictionaries = os.path.join(path_to_db, 'dictionaries')

dic_file_path = os.path.join(path_to_dictionaries, dic_file_name)
feature_file_path = os.path.join(path_to_db, 'features', 'tm_originalText_features.npz')
model_file_path = os.path.join(path_to_db, 'models', model_file_name)

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(dic_file_path)
print('Load Features')
tm_features = scipy.sparse.load_npz(feature_file_path)
tm_features = tm_features.toarray()

seed_topic_list = [['logistic', 'regression', 'linear'],  # Regression
                   ['decision', 'trees'],  # Decision Trees
                   ['artificial', 'neural', 'network', 'rnn', 'recurrent', 'lstm'],  # Neural Network
                   ['bayesian', 'naive', 'bayes'],  # Bayes
                   ['nearest', 'neighbor', 'k-means', 'clustering', 'k-nearest'],  # Clustering
                   ['svm', 'support', 'vector', 'machines'],  # SVM
                   ['boosting', 'adaboost'],  # Boosting
                   ['pca', 'principal', 'component', 'analysis', 'crf', 'hmm']  # Others
                   ]
num_topics = len(seed_topic_list)

glda_model = guidedlda.GuidedLDA(n_topics=num_topics, n_iter=200, random_state=7, refresh=20)

seed_topics = dict()

for t_id, st in enumerate(seed_topic_list):
    for word in st:
        if word in dictionary.token2id:
            seed_topics[dictionary.token2id[word]] = t_id
        else:
            print(f"{word} not in Dictionary")

glda_model.fit(tm_features, seed_topics=seed_topics, seed_confidence=0.5)

topic_word = glda_model.topic_word_
n_top_words = 15
for i, topic_dist in enumerate(topic_word):
    thing = [dictionary[idx] for idx in np.argsort(topic_dist)[:-(n_top_words+1):-1] if idx in dictionary]
    print(thing)

with open(model_file_path, "wb") as model_file:
    pickle.dump(glda_model, model_file)