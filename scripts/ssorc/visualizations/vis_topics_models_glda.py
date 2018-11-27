import os
import gensim
import pickle
import scipy.sparse
import matplotlib.pyplot as plt
import operator

path_to_db = "/media/norpheo/mySQL/db/ssorc"

model_file_name = 'tm_glda_model.pickle'
dictionary_file_name = 'pruned_originalText_isML.dict'

model_file_path = os.path.join(path_to_db, 'models', model_file_name)
dictionary_path = os.path.join(path_to_db, 'dictionaries', dictionary_file_name)
feature_file_path = os.path.join(path_to_db, 'features', 'tm_originalText_features.npz')

print('Load Dict')
dictionary = gensim.corpora.Dictionary.load(dictionary_path)
print('Load Model')
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)
print('Load Features')
tm_features = scipy.sparse.load_npz(feature_file_path)
tm_features = tm_features.toarray()

doc_topic = model.transform(tm_features)
print(len(doc_topic))

for i in range(9):
    top_topic = doc_topic[i].argmax()

topic_counter = dict()
for document in doc_topic:
    top_topic = document.argmax()

    if top_topic not in topic_counter:
        topic_counter[top_topic] = 0
    topic_counter[top_topic] += 1

sorted_tc = dict(sorted(topic_counter.items(), key=operator.itemgetter(0)))
print(sorted_tc)

plt.bar(range(len(sorted_tc)), list(sorted_tc.values()), align='center')
plt.xticks(range(len(sorted_tc)), list(sorted_tc.keys()))
plt.show()

