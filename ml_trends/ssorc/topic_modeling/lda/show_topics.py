import os
import gensim
from prettytable import PrettyTable
import pickle
import numpy as np

path_to_db = "/media/norpheo/mySQL/db/ssorc"

nlp_model = "en_wa_v2"
path_to_tm = os.path.join(path_to_db, "topic_modeling", nlp_model)

info_fn = "merged_word_v2_300.info"

with open(os.path.join(path_to_tm, info_fn), "rb") as handle:
    info = pickle.load(handle)

print(info['otm_path'])
exit()

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(info["dic_path"])

with open(info["model_path"], "rb") as model_file:
    lda_model = pickle.load(model_file)

n_top_words = 100
topic_words = {}
for topic, comp in enumerate(lda_model.components_):
    comp_sum = np.sum(comp)
    word_prop = np.sort(comp)[::-1][:n_top_words]
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words[topic] = [(dictionary[idx], prop/comp_sum) for idx, prop in zip(word_idx, word_prop)]

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