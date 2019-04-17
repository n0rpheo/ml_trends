import itertools
import os
import pickle
import pandas as pd
from collections import Counter

from src.utils.LoopTimer import LoopTimer


def make_ngrams(series, n=2):
    multigram = [pair for pair in itertools.combinations(series, r=n)]
    return multigram


panda_path = "/media/norpheo/mySQL/db/ssorc/pandas"

with open(os.path.join(panda_path, "ml_abstracts.pickle"), "rb") as ml_file:
    ml_abstracts = pickle.load(ml_file)
with open(os.path.join(panda_path, "ai_abstracts.pickle"), "rb") as ai_file:
    ai_abstracts = pickle.load(ai_file)

all_bigrams = list()

lc = LoopTimer(update_after=5000, avg_length=50000, target=len(ml_abstracts))
for abstract_id in ml_abstracts:
    entities = ml_abstracts[abstract_id]
    entities.remove("machine learning")
    if "algorithm" in entities:
        entities.remove("algorithm")
    bigrams = make_ngrams(entities, n=2)

    all_bigrams.extend(bigrams)
    lc.update("ding")

print()
occurrences = Counter(all_bigrams)
most_occ = occurrences.most_common()[:50]
least_occ = occurrences.most_common()[-500:]
ml_abstract_ids = list()
for abstract_id in ml_abstracts:
    entities = ml_abstracts[abstract_id]
    if (any(occurrence[0][0] in entities for occurrence in most_occ) and
            not any(occurrence[0][0] in entities for occurrence in least_occ)):
        ml_abstract_ids.append(abstract_id)
    lc.update("ding")

print()
otdb_path = os.path.join(panda_path, 'ml_ot.pandas')
otDF = pd.read_pickle(otdb_path)

print(len(ml_abstract_ids))

for idx, abstract_id in enumerate(set(ml_abstract_ids)):
    print(f"{idx}:")
    print(otDF.loc[abstract_id]["originalText"])
    print("-----")

    if idx > 10:
        break


