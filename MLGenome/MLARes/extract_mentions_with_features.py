import os
from spacy.vocab import Vocab
from spacy.tokens import Doc
import pandas as pd
import pickle
import numpy as np

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


def span2vec(span):
    length = 0
    vec_ = None
    for token_ in span:
        if not token_conditions(token_):
            continue

        if vec_ is None:
            vec_ = token_.vector
        else:
            vec_ = (vec_ * length + token_.vector) / (length+1)

        length += 1
    return vec_


path_to_db = "/media/norpheo/mySQL/db/ssorc"
# nlp_model = "en_core_web_sm_nertrained_v3"
nlp_model = "en_wa_v2"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
if not os.path.isdir(path_to_mlgenome):
    print(f"Create Directory {path_to_mlgenome}")
    os.mkdir(path_to_mlgenome)

with open(os.path.join(path_to_mlgenome, "ml_acronyms.pickle"), "rb") as handle:
    acronyms = pickle.load(handle)

path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)

vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

window_size = 3

mentions = list()
unique_mentions = list()
um_set = dict()
ml_acronyms = dict()

lt = LoopTimer(update_after=100, avg_length=1000, target=len(infoDF))
for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    for sentence in doc.sents:
        for ent in sentence.ents:
            m_string = ent.text

            m_orth = [token.orth_.lower() for token in ent]
            m_orth_with_ws = [token.text_with_ws.lower() for token in ent]
            m_pos = [token.pos_ for token in ent]
            m_lemma = [token.lemma_.lower() for token in ent]
            m_lemma_with_ws = [f"{token.lemma_.lower()}{token.whitespace_}" for token in ent]
            m_length = sum([len(token.orth_) for token in ent])
            m_starts_with_cap = ent[0].orth_[0] == ent[0].orth_[0].upper()
            m_vec = span2vec(ent)

            left_lemma = []
            left_ner = []
            index = ent.start - 1
            while index >= 0:
                # Question: Remove Stopwords, Punctuation and short Words from Window List?
                if token_conditions(doc[index]):
                    left_lemma.append(doc[index].lemma)
                    left_ner.append(doc[index].ent_type)
                index -= 1
                if len(left_lemma) == window_size:
                    break
            while len(left_lemma) < 3:
                left_lemma.append(-1)
                left_ner.append(-1)

            left_lemma.reverse()
            left_ner.reverse()

            right_lemma = []
            right_ner = []
            index = ent.end
            while index < len(doc):
                # Question: Remove Stopwords, Punctuation and short Words from Window List?
                if token_conditions(doc[index]):
                    right_lemma.append(doc[index].lemma)
                    right_ner.append(doc[index].ent_type)
                index += 1
                if len(right_lemma) == window_size:
                    break
            while len(right_lemma) < 3:
                right_lemma.append(-1)
                right_ner.append(-1)

            mention = {"string": m_string,
                       "orth": m_orth,
                       "orth_with_ws": m_orth_with_ws,
                       "lemma": m_lemma,
                       "lemma_with_ws": m_lemma_with_ws,
                       "pos": m_pos,
                       "length": m_length,
                       "starts_with_capital": m_starts_with_cap,
                       "left_lemma": left_lemma,
                       "left_ner": left_ner,
                       "right_lemma": right_lemma,
                       "right_ner": right_ner,
                       "m_vec": m_vec
                       }
            mentions.append(mention)

            if m_string.lower() not in um_set:
                um_set[m_string.lower()] = 0
                u_mention = {"string": m_string.lower(),
                             "orth": m_orth,
                             "orth_with_ws": m_orth_with_ws,
                             "lemma": m_lemma,
                             "lemma_with_ws": m_lemma_with_ws,
                             "pos": m_pos,
                             "length": m_length,
                             "starts_with_capital": m_starts_with_cap,
                             "m_vec": m_vec
                             }
                unique_mentions.append(u_mention)
            else:
                um_set[m_string.lower()] += 1
    lt.update(f"Extract Mentions - {len(mentions)} | {len(unique_mentions)}")


with open(os.path.join(path_to_mlgenome, "mentions.pickle"), "wb") as handle:
    pickle.dump(mentions, handle)

with open(os.path.join(path_to_mlgenome, "unique_mentions.pickle"), "wb") as handle:
    pickle.dump(unique_mentions, handle)

with open(os.path.join(path_to_mlgenome, "unique_mentions_count.pickle"), "wb") as handle:
    pickle.dump(um_set, handle)
