import os
import gensim
import numpy as np
import scipy.sparse
import guidedlda
import pickle


token_type = 'originalText'

model_file_name = "tm_glda_model_250topics.pickle"
dic_temp_file_name = "tm_dictionary.dict"

path_to_db = "/media/norpheo/mySQL/db/ssorc"
temp_dic_path = os.path.join(path_to_db, 'dictionaries', dic_temp_file_name)
feature_file_path = os.path.join(path_to_db, 'features', 'tm_features.npz')
model_file_path = os.path.join(path_to_db, 'models', model_file_name)

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(temp_dic_path)
print('Load Features')
tm_features = scipy.sparse.load_npz(feature_file_path)
#tm_features = tm_features.toarray()

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
num_topics = 250

glda_model = guidedlda.GuidedLDA(n_topics=num_topics, n_iter=1000, random_state=7, refresh=20)

seed_topics = dict()

for t_id, st in enumerate(seed_topic_list):
    for word in st:
        if word in dictionary.token2id:
            seed_topics[dictionary.token2id[word]] = t_id
        else:
            print(f"{word} not in Dictionary")

#glda_model.fit(tm_features, seed_topics=seed_topics, seed_confidence=0.5)
glda_model.fit(tm_features)

topic_word = glda_model.topic_word_
n_top_words = 30
for i, topic_dist in enumerate(topic_word):
    thing = [dictionary[idx] for idx in np.argsort(topic_dist)[:-(n_top_words+1):-1] if idx in dictionary]
    print(thing)

with open(model_file_path, "wb") as model_file:
    pickle.dump(glda_model, model_file)
