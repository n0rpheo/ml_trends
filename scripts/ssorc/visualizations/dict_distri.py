import os

import gensim

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")



dic_names = set(["word", "wordbigram", "pos", "posbigram", "lemma", "lemmabigram"])

dictionaries = dict()

dictionary = 'word'

dic_path = os.path.join(path_to_dictionaries, "full_" + dictionary + ".dict")
if os.path.isfile(dic_path):
    dictionaries[dictionary] = gensim.corpora.Dictionary.load(dic_path)


dic_distribution = dict()

most_common_word = None
high_freq = 0

for idx in dictionaries[dictionary].dfs:
    freq = dictionaries[dictionary].dfs[idx]

    if freq > high_freq:
        high_freq = freq
        most_common_word = dictionaries[dictionary][idx]

    if freq not in dic_distribution:
        dic_distribution[freq] = list()

    if len(dic_distribution[freq]) < 10:
        dic_distribution[freq].append(dictionaries[dictionary][idx])

for idx, key in enumerate(sorted(dic_distribution.keys(), reverse=False)):
    print("%s: %s" % (key, dic_distribution[key]))

    if idx == 1:
        break

no_below = 0.001
no_below_abs = int(no_below * dictionaries[dictionary].num_docs)  # convert fractional threshold to absolute threshold

print(no_below_abs)
print(dictionaries[dictionary])
dictionaries[dictionary].filter_extremes(no_below=no_below_abs, no_above=0.5, keep_n=None)
print(dictionaries[dictionary])