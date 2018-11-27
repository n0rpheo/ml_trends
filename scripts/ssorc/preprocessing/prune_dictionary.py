import os
import re
import gensim

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")

dic_filename = "full_originalText_potML.dict"
new_dic_filename = 'pruned_originalText_potML.dict'
sw_filename = "stopwords_default.txt"

dic_path = os.path.join(path_to_dictionaries, dic_filename)
new_dic_path = os.path.join(path_to_dictionaries, new_dic_filename)
dictionary = gensim.corpora.Dictionary.load(dic_path)

stopwords = set()
with open(os.path.join(path_to_dictionaries, sw_filename)) as sw_file:
    for line in sw_file:
        sword = line.rstrip()
        if len(sword) > 0:
            stopwords.add(sword)

for num, idx in enumerate(dictionary):
    word = dictionary[idx]
    word_new = re.sub(r"[^a-zA-Z0-9]", "", word)
    t_length = len(word)
    n_length = len(word_new)

    perc = n_length / t_length

    if perc < 0.5:
        stopwords.add(word)

# Filter Stopwords
dictionary.filter_tokens(bad_ids=[dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id])

print(dictionary)
dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=None)
print(dictionary)
dictionary.save(new_dic_path)
