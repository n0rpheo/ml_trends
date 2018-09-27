import os
import matplotlib.pyplot as plt
import operator
import scipy.sparse
import gensim
from src.modules.topic_modeling import TopicModeling

path_to_db = "/media/norpheo/mySQL/db/ssorc"
feature_file_path = os.path.join(path_to_db, 'features', 'tm_features.npz')

print('Initialize Model')
tm = TopicModeling(dictionary_name='full_word.dict',
                   tfidf_name='word_model.tfidf',
                   model_name='tm_lda.model')

print('Load Features')
tm_features = scipy.sparse.load_npz(feature_file_path)
corpus_tfidf = gensim.matutils.Sparse2Corpus(tm_features.transpose())
corpus_lda = tm.lda_model[corpus_tfidf]

topic_counter = dict()
for idx, doc_lda in enumerate(corpus_lda):
    max_p = 0
    for entry in doc_lda:
        topic_id = entry[0]
        topic_p = entry[1]
        if topic_p > max_p:
            max_p = topic_p
            maxi_idx = topic_id

    if maxi_idx in topic_counter:
        topic_counter[maxi_idx] += 1
    else:
        topic_counter[maxi_idx] = 1

sorted_tc = dict(sorted(topic_counter.items(), key=operator.itemgetter(0)))
print(sorted_tc)

plt.bar(range(len(sorted_tc)), list(sorted_tc.values()), align='center')
plt.xticks(range(len(sorted_tc)), list(sorted_tc.keys()))
plt.show()
