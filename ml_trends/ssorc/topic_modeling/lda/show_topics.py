import os
import gensim
from prettytable import PrettyTable
import pickle
import numpy as np

path_to_db = "/media/norpheo/mySQL/db/ssorc"

info_fn = "word_lower_merged_300.info"

with open(os.path.join(path_to_db, "topic_modeling", info_fn), "rb") as handle:
    info = pickle.load(handle)

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(info["dic_path"])

with open(info["model_path"], "rb") as model_file:
    lda_model = pickle.load(model_file)

n_top_words = 20
topic_words = {}
for topic, comp in enumerate(lda_model.components_):
    sum = np.sum(comp)
    word_prop = np.sort(comp)[::-1][:n_top_words]
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words[topic] = [(dictionary[idx], prop/sum) for idx, prop in zip(word_idx, word_prop)]

for topic, words in topic_words.items():

    word_string = ""

    for word, prop in words:
        word_string += f"{word} ({round(prop, 2)}),\t"
    #print(f'Topic: {topic}: {word_string[:-2]}')

p = PrettyTable()
field_names = ["Topic"]
for i in range(n_top_words):
    field_names.append(f"Word #{i+1}")

p.field_names = field_names

for topic in topic_words:
    columns = list()
    columns.append(topic)
    for word, prop in topic_words[topic]:
        entry = f"{round(prop, 3)} | {word}"
        columns.append(entry)
    p.add_row(columns)


print(p.get_string(header=True, border=True))


"""

#001: Deep Neural Network - Convolutional
#022: Naive Bayes
#028: linear	least	square	discriminant	non-linear	lda	fisher	quadratic
#031: Q-Learning, Reinforcement
#075: Adaboos, Boosting
#078: Clustering, Unsupervised Learning, K-Means
#081: Random Forest
#095: Artificial Neural Network
#160: Hidden Markov
#168: Principal Component Analysis
#207: Neural Network
#245: SVM
#268: LSTM, Neural Network
#271: NLP
#273: Word Embeddings, NLP
#312: Kernel-Based
#338: Recurrent Neural Network
#345: Logistic Regression
#357: Naive Bayes
#392: Hierarchical, Tree-Structure
#393: Support Vector Machines
#410: Conditional Random Fields, CRF
#435: Decision Tree
#492: NLP
#497: KNN

NN: 1, 95, 207, 268, 338
Naive Bayes: 22, 357
Q-Learning: 31
Adaboost: 75
Clustering: 78, 497
Decision Trees: 81, 392, 435
HMM: 160
PCA: 168
SVM: 245, 393
NLP: 271, 273, 492
Logistic Regression: 345
CRF: 410
"""