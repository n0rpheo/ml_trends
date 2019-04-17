import os
import pickle
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

import pandas as pd
from src.utils.LoopTimer import LoopTimer

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations")
pandas_path = os.path.join(path_to_db, "pandas")
path_to_ner = os.path.join(path_to_db, "NER")

mla = set()
with open(os.path.join(path_to_ner, "ml_algos.txt"), "r") as handle:
    for line in handle:
        mla.add(line.replace("\n", ""))

for ml_algo in mla:
    doc = nlp(ml_algo)
    pattern = [{"ORTH": token.orth_} for token in doc]
    pattern_name = "".join([entity["ORTH"] for entity in pattern]).lower()
    matcher.add(pattern_name, None, pattern)

vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "spacy.vocab"))

infoDF = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'info_db.pandas'))

abstract_id_list = list()
lemma_list = list()

targ = len(infoDF)
lt = LoopTimer(update_after=10, avg_length=2000, target=targ)
exmpl = 0
TRAIN_DATA = list()

ent_counter = dict()

for abstract_id, row in infoDF.iterrows():
    ori_doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))

    for sent in ori_doc.sents:
        doc = sent.as_doc()

        matches = matcher(doc)

        skip_matches = set()
        n_matches = len(matches)
        for i in range(n_matches):
            match = matches[i]

            match_id = match[0]
            start = match[1]
            end = match[2]

            for j in range(i + 1, n_matches):
                c_match = matches[j]
                c_match_id = c_match[0]
                c_start = c_match[1]
                c_end = c_match[2]

                if ((c_start <= end and start <= c_end) or
                        (start <= c_end and c_start <= end)):
                    ent_size = end - start
                    c_ent_size = c_end - c_start

                    if ent_size < c_ent_size:
                        skip_matches.add(i)
                    elif c_ent_size < ent_size or start < c_start:
                        skip_matches.add(j)
                    else:
                        skip_matches.add(i)

        ent_list = list()
        for i, match in enumerate(matches):
            match_id = match[0]
            if i in skip_matches:
                continue
            start = match[1]
            end = match[2]

            start_idx = doc[start].idx
            end_idx = doc[end-1].idx + len(doc[end-1].text)

            entity_name = doc[start:end].text
            entity_name2 = doc.text[start_idx:end_idx]

            if entity_name != entity_name2:
                start_idx += 2
                end_idx += 2
                entity_name2 = doc.text[start_idx:end_idx]

                if entity_name != entity_name2:
                    print()
                    print(f"{entity_name} - {entity_name2}")
                    print()
                    continue

            if entity_name not in ent_counter:
                ent_counter[entity_name] = 0

            if ent_counter[entity_name] < 50000:
                ent_counter[entity_name] += 1
                entity = (start_idx, end_idx, "MLALGO")
                ent_list.append(entity)

        if len(ent_list) > 0:
            entities = {"entities": ent_list}
            TRAIN_DATA.append((doc.text, entities))
    breaker = lt.update(f"Make TD - {len(TRAIN_DATA)}")

    if len(TRAIN_DATA) > 50000000:
        break

print()
for entity in ent_counter:
    print(f"{entity}: {ent_counter[entity]}")

with open(os.path.join(path_to_db, "NER", "spacy_ner_mlalgo_traindata_all.pickle"), "wb") as handle:
    pickle.dump(TRAIN_DATA, handle)
