import matplotlib.pyplot as plt
import operator
from src.modules.topic_modeling import TopicModeling

topic_modeler = TopicModeling('ssorc')
topic_modeler.load_model(model_name="lda_500_topics.model")
topic_modeler.load_corpus(prefix='lemma_10k')

corpus_lda = topic_modeler.lda_model[topic_modeler.corpus_tfidf]
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
