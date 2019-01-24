import os
import re
import gensim


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")


dic_filename = "full_originalText_isML.dict"
sw_filename = "stopwords_default.txt"

dic_path = os.path.join(path_to_dictionaries, dic_filename)
if os.path.isfile(dic_path):
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

dic_distribution = dict()
most_common_word = None
high_freq = 0

for idx in dictionary.dfs:
    freq = dictionary.dfs[idx]

    if freq > high_freq:
        high_freq = freq
        most_common_word = dictionary[idx]

    if freq not in dic_distribution:
        dic_distribution[freq] = list()

    if len(dic_distribution[freq]) < 10:
        dic_distribution[freq].append(dictionary[idx])

for idx, key in enumerate(sorted(dic_distribution.keys(), reverse=False)):
    print(f"{key}: {dic_distribution[key]}")

    if idx == 10:
        break

no_below = 0.1
no_below_abs = int(no_below * dictionary.num_docs)  # convert fractional threshold to absolute threshold

print(no_below_abs)
print(dictionary)
dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=None)
print(dictionary)