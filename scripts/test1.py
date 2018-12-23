from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition
import numpy

import pandas as pd

db_path = "/media/norpheo/mySQL/db/ssorc/pandas/ml_word.pandas"
dbDF = pd.read_pickle(db_path)
documents = [" ".join([item for sublist in document for item in sublist]) for document in dbDF['word'][:10]]

print("Fit Count")
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(documents)


#print(count_vect.vocabulary_.keys()[count_vect.vocabulary_.values().index('before')])
print(count_vect.vocabulary_.items())

exit()
print("Transform Count")
xtrain_count = count_vect.transform(documents)

print("Start LDA Train")
# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=500,
                                                    learning_method='online',
                                                    max_iter=50,
                                                    verbose=1)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

for i in range(len(topic_summaries)):
    print(f"Topic {i}: {topic_summaries[i]}")