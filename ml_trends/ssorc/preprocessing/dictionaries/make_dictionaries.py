import os

import gensim
import pandas as pd

from src.utils.functions import makeBigrams


path_to_db = "/media/norpheo/mySQL/db/ssorc"
pandas_path = os.path.join(path_to_db, "pandas")

token_types = [#"word",
               #"wordbigram",
               #"pos",
               #"posbigram",
               #"lemma",
               #"lemmabigram",
               #"coarse_pos",
               #"coarse_posbigram",
               #"ent_type",
               #"merged_word",
               #"merged_ent_type",
               "word_lower_merged"
               ]
dic_paths = [os.path.join(path_to_db, "dictionaries", f"aiml_full_ner_{toty}.dict") for toty in token_types]

print("Loading Panda DB")
wordDF = pd.read_pickle(os.path.join(pandas_path, 'aiml_ner_word.pandas'))
lemmaDF = pd.read_pickle(os.path.join(pandas_path, 'aiml_ner_lemma.pandas'))
fineposDF = pd.read_pickle(os.path.join(pandas_path, 'aiml_ner_finepos.pandas'))
coarseposDF = pd.read_pickle(os.path.join(pandas_path, 'aiml_ner_coarsepos.pandas'))
mergedwordDF = pd.read_pickle(os.path.join(pandas_path, 'aiml_ner_merged_word.pandas'))
wordlowermergedDF = pd.read_pickle(os.path.join(pandas_path, 'aiml_word_lower_merged.pandas'))
print("Done Loading")

df = wordDF.join(lemmaDF).join(fineposDF).join(coarseposDF).join(mergedwordDF).join(wordlowermergedDF)

for i in range(len(token_types)):
    token_type = token_types[i]
    dic_path = dic_paths[i]

    is_bigram = False
    if "bigram" in token_type:
        is_bigram = True
        token_type = token_type[:-6]

    corpus = list()

    print(f"Build Corpus for {token_type} - Bigram: {is_bigram}")
    for abstract_id, row in df.iterrows():
        token_string = row[token_type]
        tokens = token_string.replace("\t\t", "\t").split("\t")
        if is_bigram:
            tokens = makeBigrams(tokens)

        corpus.append(tokens)

    print("Build Dictionary")
    dictionary = gensim.corpora.Dictionary()
    dictionary.add_documents(corpus, prune_at=None)

    print("Save Dictionary")
    dictionary.save(dic_path)
    print(dictionary)
