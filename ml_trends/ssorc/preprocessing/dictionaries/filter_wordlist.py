import os
import re
import gensim

import src.utils.selector as selector

path_to_dictionaries = "/media/norpheo/mySQL/db/ssorc/dictionaries"

dictionary_input = "pruned_word_lower_pd.dict"
dictionary_output = "pruned_word_lower_notriggers_pd.dict"

dic_path = os.path.join(path_to_dictionaries, dictionary_input)

# sw_filename = "stopwords_default.txt"
sw_filename = "stopwords_triggers.txt"

dic_path_output = os.path.join(path_to_dictionaries, dictionary_output)
dictionary = gensim.corpora.Dictionary.load(dic_path)

stopwords = set()
with open(os.path.join(path_to_dictionaries, sw_filename)) as sw_file:
    for line in sw_file:
        sword = line.rstrip()
        if len(sword) > 0:
            stopwords.add(sword)

print(stopwords)

nowords = set()
# Adding Tokens to stoplist that are not considered as "words"
for idx in dictionary:
    word = dictionary[idx]
    word_new = re.sub(r"[^a-zA-Z0-9]", "", word)
    t_length = len(word)
    n_length = len(word_new)

    perc = n_length / t_length
    if perc <= 0.5:
        nowords.add(word)

# Filter Stopwords
print(dictionary)
dictionary.filter_tokens(bad_ids=[dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id])
dictionary.filter_tokens(bad_ids=[dictionary.token2id[stopword] for stopword in nowords if stopword in dictionary.token2id])
print(dictionary)

dictionary.save(dic_path_output)