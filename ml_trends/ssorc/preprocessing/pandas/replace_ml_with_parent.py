import os
import pandas as pd
import numpy as np
import spacy
import pickle
from spacy.tokens import Doc
from spacy.vocab import Vocab

from src.utils.LoopTimer import LoopTimer


def token_conditions(token_):
    if token_.ent_iob == 3 or token_.ent_iob == 1:
        return True
    if token_.is_punct:
        return False
    if token_.is_stop:
        return False
    if len(token_.orth_) < 3:
        return False

    return True


def converted(token_):
    if token_.ent_iob == 3:
        mention = token_.orth_.lower()

        for cluster_id in top_words:
            if mention in cluster_words[cluster_id]:
                return top_words[cluster_id]
                break
    return token_.orth_


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_core_web_lg_mlalgo_v1"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_mlgenome_features = os.path.join(path_to_mlgenome, "features")

"""
 CLUSTERING
"""
with open(os.path.join(path_to_mlgenome, "affprop.pickle"), "rb") as handle:
    affprop = pickle.load(handle)
with open(os.path.join(path_to_mlgenome_features, "affinity_propagation_features.pickle"), "rb") as handle:
    features = pickle.load(handle)

words = features["words"]

top_words = dict()
cluster_words = dict()
for cluster_id in np.unique(affprop.labels_):
    top_words[cluster_id] = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster_words[cluster_id] = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
"""
 END CLUSTERING
"""
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)
path_to_pandas = os.path.join(path_to_db, "pandas", nlp_model)

infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))

abstract_id_list = list()
merged_and_replaced_word_list = list()

print("Starting")
targ = len(infoDF)
lt = LoopTimer(update_after=100, avg_length=10000, target=targ)
for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")

    doc = Doc(vocab).from_disk(file_path)

    abstract_id_list.append(abstract_id)

    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.orth_, ent.label_)

    merged_and_replaced_word_list.append("\t\t".join(["\t".join([converted(token) for token in sentence if token_conditions(token)])
                                         for sentence in doc.sents]))

    breaker = lt.update(f"Create Pandas - {len(abstract_id_list)}")

merged_and_replaced_wordDF = pd.DataFrame(merged_and_replaced_word_list, index=abstract_id_list, columns=["mar_word"])

merged_and_replaced_wordDF.to_pickle(os.path.join(path_to_pandas, 'mar_word.pandas'))