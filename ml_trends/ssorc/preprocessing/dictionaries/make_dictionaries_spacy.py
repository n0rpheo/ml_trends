import os

import gensim
import pandas as pd
from spacy.vocab import Vocab
from spacy.tokens import Doc

from src.utils.functions import makeBigrams
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

path_to_db = "/media/norpheo/mySQL/db/ssorc"
#nlp_model = "en_core_web_sm_nertrained_v3"
nlp_model = "en_wa_v2"

path_to_dictionaries = os.path.join(path_to_db, "dictionaries", nlp_model)
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)

if not os.path.isdir(path_to_dictionaries):
    print(f"Create Directory {path_to_dictionaries}")
    os.mkdir(path_to_dictionaries)

print("Loading Vocab")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

w_dic = gensim.corpora.Dictionary()
l_dic = gensim.corpora.Dictionary()
mw_dic = gensim.corpora.Dictionary()

lt = LoopTimer(update_after=1, avg_length=1000, target=len(infoDF))
for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    word = [token.text for token in doc if token_conditions(token)]
    lemma = [token.lemma_ for token in doc if token_conditions(token)]

    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.orth_, ent.label_)

    merged_word = [token.text for token in doc if token_conditions(token)]

    w_dic.add_documents([word], prune_at=None)
    l_dic.add_documents([lemma], prune_at=None)
    mw_dic.add_documents([merged_word], prune_at=None)

    breaker = lt.update("Build Dictionary")

print("Save Dictionary")
w_dic.save(os.path.join(path_to_dictionaries, f"full_word.dict"))
l_dic.save(os.path.join(path_to_dictionaries, f"full_lemma.dict"))
mw_dic.save(os.path.join(path_to_dictionaries, f"full_merged_word.dict"))

