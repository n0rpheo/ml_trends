import os
import pickle
import numpy as np
import scipy.sparse

from MLGenome.MLARes import mlares
from MLGenome.MLARes.occ import distances
from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_mlgenome_features = os.path.join(path_to_mlgenome, "features")

if not os.path.isdir(path_to_mlgenome_features):
    print(f"Create Directory {path_to_mlgenome_features}")
    os.mkdir(path_to_mlgenome_features)

feature_file_path = os.path.join(path_to_mlgenome_features, "entlinking_features.pickle")

with open(os.path.join(path_to_mlgenome, "mentions.pickle"), "rb") as handle:
    mentions = pickle.load(handle)

target_vector = []

feature_data_array = []
feature_row = []
feature_col = []

vector_len = 8
row_count = 0

lt_target = len(mentions)*len(mentions)
print(f"Mentions*Mentions: {round(lt_target/1000000000, 3)} Billion")

breaker = 0


lt = LoopTimer(update_after=5000, avg_length=10000, target=lt_target)
for mention in mentions:
    m_string = mention["string"]
    m_orth = mention["orth"]
    m_lemma = mention['lemma']
    m_pos = mention['pos']
    m_length = mention['length']
    m_swc = mention['starts_with_capital']

    expanded_mention = [mention['orth_with_ws'].copy()]
    distances.expand_mention(expanded_mention)

    for candidate in mentions:
        expanded_candidate = [candidate['orth_with_ws'].copy()]
        distances.expand_mention(expanded_candidate)

        mc_dist = distances.occ_s(expanded_mention, expanded_candidate)
        #print(mention["string"], candidate["string"])
        c_string = candidate["string"]
        c_orth = candidate["orth"]
        c_lemma = candidate['lemma']
        c_pos = candidate['pos']
        c_length = candidate['length']
        c_swc = candidate['starts_with_capital']

        """
        u_mention = {"string": m_string,
                                 "orth": m_orth,
                                 "lemma": m_lemma,
                                 "pos": m_pos,
                                 "length": m_length,
                                 "starts_with_capital": m_starts_with_cap
                                 }
        """

        breaker = lt.update(f"Make Training-Set - {target_vector.count(True)} - {target_vector.count(False)}")

    if breaker > lt_target:
        break

exit()

feature_row = np.array(feature_row)
feature_col = np.array(feature_col)
feature_data_array = np.array(feature_data_array)

feature_vector = scipy.sparse.csc_matrix((feature_data_array, (feature_row, feature_col)),
                                         shape=(row_count, vector_len))

target_vector = np.array(target_vector)

print()
print(feature_vector.shape)
print(target_vector.shape)

feature_dict = dict()

feature_dict["features"] = feature_vector
feature_dict["targets"] = target_vector
#feature_dict["settings"] = settings


with open(feature_file_path, "wb") as feature_file:
    pickle.dump(feature_dict, feature_file)