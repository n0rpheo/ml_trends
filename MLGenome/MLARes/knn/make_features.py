import os
import pickle

from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_core_web_lg_mlalgo_v1"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_mlgenome_features = os.path.join(path_to_mlgenome, "features")

if not os.path.isdir(path_to_mlgenome_features):
    print(f"Create Directory {path_to_mlgenome_features}")
    os.mkdir(path_to_mlgenome_features)

with open(os.path.join(path_to_mlgenome, "unique_mentions.pickle"), "rb") as handle:
    mentions = pickle.load(handle)

feature_vector = []

lt_target = len(mentions)
lt = LoopTimer(update_after=5000, avg_length=10000, target=lt_target)
for mention in mentions:
    m_string = mention["string"]
    m_is_acronym = mention['is_acronym']
    m_vec = mention['m_vec']

    #if m_is_acronym:
    #    continue

    feature_vector.append(m_vec)

    breaker = lt.update(f"Make Training-Set - {len(feature_vector)}")

print(len(feature_vector))

feature_dict = dict()

feature_dict["features"] = feature_vector


with open(os.path.join(path_to_mlgenome_features, "knn_features.pickle"), "wb") as handle:
    pickle.dump(feature_dict, handle)