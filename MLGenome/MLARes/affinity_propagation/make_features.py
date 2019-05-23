import os
import pickle
import numpy as np

from MLGenome.MLARes.mlares import similarity_ratio
from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_core_web_lg_mlalgo_v1"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_mlgenome_features = os.path.join(path_to_mlgenome, "features")

with open(os.path.join(path_to_mlgenome, "unique_mentions.pickle"), "rb") as handle:
    mentions = pickle.load(handle)

string_similarity = list()
words = list()

max_len = 100000

targ_len = min(max_len*max_len, len(mentions)*len(mentions))

lt = LoopTimer(avg_length=10000, update_after=1000, target=targ_len)
for i, mention in enumerate(mentions):
    m_string = mention["string"].lower()
    sublist = list()
    for j, candidate in enumerate(mentions):
        c_string = candidate["string"].lower()
        sublist.append(similarity_ratio(c_string, m_string))

        lt.update("Make Features")
        if j == max_len:
            break
    string_similarity.append(sublist)
    words.append(m_string)
    if i == max_len:
        break

string_similarity = np.array(string_similarity)

print(string_similarity.shape)

features = {"words": np.asarray(words),
            "string_sim": string_similarity}

with open(os.path.join(path_to_mlgenome_features, "affinity_propagation_features.pickle"), "wb") as handle:
    pickle.dump(features, handle)
