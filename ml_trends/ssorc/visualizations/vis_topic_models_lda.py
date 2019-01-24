import os
import matplotlib.pyplot as plt
import operator
import scipy.sparse
import gensim


model_name = 'tm_lda_lemma.model'

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_feature_file = os.path.join(path_to_db, 'features', 'tm_lemma_features.npz')
path_to_lda_model = os.path.join(path_to_db, 'models', model_name)

print('Initialize Model')
lda_model = gensim.models.LdaMulticore.load(path_to_lda_model)
print('Load Features')
tm_features = scipy.sparse.load_npz(path_to_feature_file)

corpus_tfidf = gensim.matutils.Sparse2Corpus(tm_features.transpose())
corpus_lda = lda_model[corpus_tfidf]

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
