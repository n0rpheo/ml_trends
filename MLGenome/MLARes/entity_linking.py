import os
import pickle
import numpy as np
import scipy.sparse

from MLGenome.MLARes import mlares
from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_core_web_lg_mlalgo_v1"
path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)

with open(os.path.join(path_to_mlgenome, "svm_mlares.pickle"), "rb") as model_file:
    classifier = pickle.load(model_file)

with open(os.path.join(path_to_mlgenome, "mentions.pickle"), "rb") as handle:
    mentions = pickle.load(handle)

mlares_clf = classifier["model"]

lt_target = len(mentions)*len(mentions)
print(f"Mentions*Mentions: {round(lt_target/1000000000, 3)} Billion")

entity_linking = dict()

breaker = 0
lt = LoopTimer(update_after=5000, avg_length=10000, target=lt_target)
for mention in mentions:
    m_string = mention["mention_string"]
    m_tokens = mention["mention_tokens"]
    m_dvec = mention["doc_vector"]
    m_svec = mention["sentence_vector"]
    m_mvec = mention["mention_vector"]

    for candidate in mentions:
        c_string = candidate["mention_string"]
        c_tokens = candidate["mention_tokens"]
        c_dvec = candidate["doc_vector"]
        c_svec = candidate["sentence_vector"]
        c_mvec = candidate["mention_vector"]

        similarity = mlares.similarity_ratio(m_string, c_string)
        intersection = mlares.intersection_of_words(m_tokens, c_tokens)
        is_pfix = mlares.is_prefix(m_tokens, c_tokens)
        is_ifix = mlares.is_infix(m_tokens, c_tokens)
        is_sfix = mlares.is_suffix(m_tokens, c_tokens)

        dvec_sim = mlares.vec_sim(m_dvec, c_dvec)
        svec_sim = mlares.vec_sim(m_svec, c_svec)
        mvec_sim = mlares.vec_sim(m_mvec, c_mvec)

        feat_vec = [similarity,
                    intersection,
                    is_pfix,
                    is_ifix,
                    is_sfix,
                    dvec_sim,
                    svec_sim,
                    mvec_sim]

        entities = (m_string, c_string)

        entity_linking[entities] = mlares_clf.predict([feat_vec])

        lt.update(f"Make Training-Set - {len(entity_linking)}")


with open(os.path.join(path_to_mlgenome, "entity_linking.pickle"), "wb") as el_file:
    pickle.dump(entity_linking, el_file)