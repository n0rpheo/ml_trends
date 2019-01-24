import os
import gensim
import pickle
import scipy.sparse
import matplotlib.pyplot as plt
import operator

from src.utils.LoopTimer import LoopTimer
from src.utils.selector import select_path_from_dir

path_to_db = "/media/norpheo/mySQL/db/ssorc"

model_file_name = 'tm_glda_model_250topics.pickle'

model_file_path = select_path_from_dir(os.path.join(path_to_db, 'models'),
                                       phrase="Select Model: ",
                                       suffix=".pickle",
                                       preselection=model_file_name)
dictionary_path = select_path_from_dir(os.path.join(path_to_db, 'dictionaries'),
                                       phrase="Select Dictionary: ",
                                       suffix=".dict",
                                       preselection="tm_dictionary.dict")
feature_file_path = select_path_from_dir(os.path.join(path_to_db, 'features'),
                                         phrase="Select Features: ",
                                         suffix=".npz",
                                         preselection='tm_features.npz')

print('Load Dict')
dictionary = gensim.corpora.Dictionary.load(dictionary_path)
print('Load Model')
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)
print('Load Features')
tm_features = scipy.sparse.load_npz(feature_file_path)
#tm_features = tm_features.toarray()

topic_counter = dict()
topic_counter2 = dict()
print(tm_features.shape[0])
lc = LoopTimer(update_after=1000, avg_length=5000, target=tm_features.shape[0])
for tm_feature in tm_features:
    doc_topic = model.transform(tm_feature)[0]

    top_topic = doc_topic.argmax()
    if top_topic not in topic_counter:
        topic_counter[top_topic] = 0
    topic_counter[top_topic] += 1

    for idx, perc in enumerate(doc_topic):
        if idx not in topic_counter2:
            topic_counter2[idx] = 0
        topic_counter2[idx] += perc

    lc.update("Model Topics")
print()
min_t = 100000
max_t = 0

for topic in topic_counter:
    tc = topic_counter[topic]
    if min_t > tc:
        min_t = tc
    if max_t < tc:
        max_t = tc

sorted_tc = dict(sorted(topic_counter.items(), key=operator.itemgetter(0)))
print(sorted_tc)
print(f"Schwächstes Topic: {min_t}")
print(f"Stärkstes Topic:   {max_t}")

plt.bar(range(len(sorted_tc)), list(sorted_tc.values()), align='center')
plt.xticks(range(len(sorted_tc)), list(sorted_tc.keys()))
plt.show()

# ========================

min_t = 100000
max_t = 0

for topic in topic_counter2:
    tc = topic_counter2[topic]
    if min_t > tc:
        min_t = tc
    if max_t < tc:
        max_t = tc

sorted_tc = dict(sorted(topic_counter2.items(), key=operator.itemgetter(0)))
print(sorted_tc)
print(f"Schwächstes Topic: {min_t}")
print(f"Stärkstes Topic:   {max_t}")

plt.bar(range(len(sorted_tc)), list(sorted_tc.values()), align='center')
plt.xticks(range(len(sorted_tc)), list(sorted_tc.keys()))
plt.show()

