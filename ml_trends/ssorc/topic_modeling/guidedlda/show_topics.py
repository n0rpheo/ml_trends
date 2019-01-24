import os
import gensim
import numpy as np
import pickle

token_type = 'originalText'

model_file_name = "tm_glda_model_500topics.pickle"
dic_temp_file_name = "tm_dictionary.dict"

path_to_db = "/media/norpheo/mySQL/db/ssorc"
temp_dic_path = os.path.join(path_to_db, 'dictionaries', dic_temp_file_name)
feature_file_path = os.path.join(path_to_db, 'features', 'tm_features.npz')
model_file_path = os.path.join(path_to_db, 'models', model_file_name)
show_topics_file_path = os.path.join(path_to_db, "show_topics_500.tsv")

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(temp_dic_path)

print('Load Model')
with open(model_file_path, 'rb') as model_file:
    glda_model = pickle.load(model_file)

topic_word = glda_model.topic_word_
n_top_words = 30

with open(show_topics_file_path, "w") as st_file:
    for i, topic_dist in enumerate(topic_word):
        sorted_index = np.argsort(topic_dist)[::-1]

        word_list = list()
        proba_list = list()

        accu_proba = 0

        for idx in sorted_index:
            if idx not in dictionary:
                continue

            probability = topic_dist[idx]
            accu_proba += probability

            word_list.append(dictionary[idx])
            proba_list.append(str(probability*100)[0:5])

            if accu_proba > 50 or len(word_list) > 29:
                break

        words = "\t".join(word_list)
        probs = "\t".join(proba_list)
        st_file.write("\n")
        line = f"{i}\t\t{words}"
        st_file.write(f"{line}\n")
        line = f"{i}\t\t{probs}"
        st_file.write(f"{line}\n")
