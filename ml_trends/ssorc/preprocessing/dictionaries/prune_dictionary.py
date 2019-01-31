import os
import re
import gensim

import src.utils.selector as selector

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")


# sw_filename = "stopwords_default.txt"
sw_filename = "stopwords_triggers.txt"

dic_path = selector.select_path_from_dir(path_to_dictionaries,
                                         phrase="Select Dictionary to prune: ",
                                         suffix=".dict")
new_dic_filename = input("New Dictionary filename: ")

new_dic_path = os.path.join(path_to_dictionaries, new_dic_filename)
dictionary = gensim.corpora.Dictionary.load(dic_path)

stopwords = set()
with open(os.path.join(path_to_dictionaries, sw_filename)) as sw_file:
    for line in sw_file:
        sword = line.rstrip()
        if len(sword) > 0:
            stopwords.add(sword)

extended_sw = list(stopwords)

for num, idx in enumerate(dictionary):
    word = dictionary[idx]
    word_new = re.sub(r"[^a-zA-Z0-9]", "", word)
    t_length = len(word)
    n_length = len(word_new)

    perc = n_length / t_length

    if perc < 0.5:
        extended_sw.add(word)

# Filter Stopwords
dictionary.filter_tokens(bad_ids=[dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id])
# dictionary.filter_tokens(bad_ids=[dictionary.token2id[stopword] for stopword in extended_sw if stopword in dictionary.token2id])

print(dictionary)
# dictionary.filter_extremes(no_below=3, keep_n=None)
# print(dictionary)
dictionary.save(new_dic_path)
