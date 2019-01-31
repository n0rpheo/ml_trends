import os

import gensim
import pandas as pd

from src.utils.functions import makeBigrams


path_to_db = "/media/norpheo/mySQL/db/ssorc"

token_types = [#"word",
               #"wordbigram",
               #"pos",
               #"posbigram",
               "lemma",
               "lemmabigram",
               #"originalText",
               #"originalTextbigram"
               ]
dic_paths = [os.path.join(path_to_db, "dictionaries", f"full_{toty}_lower_pd.dict") for toty in token_types]

lower = True

# Using Pandas DF
otdb_path = os.path.join(path_to_db, 'pandas', 'ml_ot.pandas')
posdb_path = os.path.join(path_to_db, 'pandas', 'ml_pos.pandas')
worddb_path = os.path.join(path_to_db, 'pandas', 'ml_word.pandas')
lemmadb_path = os.path.join(path_to_db, 'pandas', 'ml_lemma.pandas')

print("Loading Panda DB")
dataFrames = dict()
dataFrames['originalText'] = pd.read_pickle(otdb_path)
dataFrames['pos'] = pd.read_pickle(posdb_path)
dataFrames['word'] = pd.read_pickle(worddb_path)
dataFrames['lemma'] = pd.read_pickle(lemmadb_path)
print("Done Loading")

for i in range(len(token_types)):
    token_type = token_types[i]
    dic_path = dic_paths[i]

    is_bigram = False
    if "bigram" in token_type:
        is_bigram = True
        token_type = token_type[:-6]

    corpus = list()

    print(f"Build Corpus for {token_type} - Bigram: {is_bigram}")
    for abstract_id, row in dataFrames[token_type].iterrows():
        token_string = row[token_type]
        tokens = token_string.split()
        if is_bigram:
            tokens = makeBigrams(tokens)

        corpus.append(tokens)

    print("Build Dictionary")
    dictionary = gensim.corpora.Dictionary()
    dictionary.add_documents(corpus, prune_at=None)
    print("Save Dictionary")
    dictionary.save(dic_path)
    print(dictionary)


