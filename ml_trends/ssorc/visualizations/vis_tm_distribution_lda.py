import os
import gensim
import pickle
import scipy.sparse
import matplotlib.pyplot as plt
import operator

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"

model_file_name = "aiml_tm_lda_500topics_merged_word.pickle"
dic_file_name = "aiml_tm_dictionary_merged.dict"
feature_file_name = "aiml_tm_features_merged.npz"

dictionary = gensim.corpora.Dictionary.load(os.path.join(path_to_db, "dictionaries", dic_file_name))
with open(os.path.join(path_to_db, 'models', model_file_name), 'rb') as model_file:
    model = pickle.load(model_file)
tm_features = scipy.sparse.load_npz(os.path.join(path_to_db, 'features', feature_file_name))
print(f"Number of Features: {tm_features.shape[0]}")

topic_counter_top = dict()
topic_counter_dist = dict()
lc = LoopTimer(update_after=1000, avg_length=5000, target=tm_features.shape[0])
for tm_feature in tm_features:
    doc_topic = model.transform(tm_feature)[0]

    top_topic = doc_topic.argmax()
    if top_topic not in topic_counter_top:
        topic_counter_top[top_topic] = 0
    topic_counter_top[top_topic] += 1

    for idx, perc in enumerate(doc_topic):
        if idx not in topic_counter_dist:
            topic_counter_dist[idx] = 0
        topic_counter_dist[idx] += perc

    lc.update("Model Topics")
print()
min_t = 100000
max_t = 0

for topic in topic_counter_top:
    tc = topic_counter_top[topic]
    if min_t > tc:
        min_t = tc
    if max_t < tc:
        max_t = tc

sorted_tc = dict(sorted(topic_counter_top.items(), key=operator.itemgetter(0)))
print(sorted_tc)
print(f"Schwächstes Topic: {min_t}")
print(f"Stärkstes Topic:   {max_t}")

plt.bar(range(len(sorted_tc)), list(sorted_tc.values()), align='center')
plt.xticks(range(len(sorted_tc)), list(sorted_tc.keys()))
plt.show()

# ========================

min_t = 100000
max_t = 0

for topic in topic_counter_dist:
    tc = topic_counter_dist[topic]
    if min_t > tc:
        min_t = tc
    if max_t < tc:
        max_t = tc

sorted_tc = dict(sorted(topic_counter_dist.items(), key=operator.itemgetter(0)))
print(sorted_tc)
print(f"Schwächstes Topic: {min_t}")
print(f"Stärkstes Topic:   {max_t}")

plt.bar(range(len(sorted_tc)), list(sorted_tc.values()), align='center')
plt.xticks(range(len(sorted_tc)), list(sorted_tc.keys()))
plt.show()



