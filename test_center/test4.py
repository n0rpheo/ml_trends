import os
import pickle
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

import pandas as pd
from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations_ner")
pandas_path = os.path.join(path_to_db, "pandas")
path_to_ner = os.path.join(path_to_db, "NER")


nlp = spacy.load(os.path.join(path_to_db, "models", "en_core_web_sm_nertrained"))
#nlp = spacy.load("en_core_web_sm")
#vocab = nlp.vocab
vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "ner_spacy.vocab"))
matcher = Matcher(vocab)

mla = set()
with open(os.path.join(path_to_ner, "ml_algos2.txt"), "r") as handle:
    for line in handle:
        mla.add(line.replace("\n", ""))

for ml_algo in mla:
    doc = nlp(ml_algo)
    pattern = [{"LOWER": token.lower_} for token in doc]
    pattern_name = "".join([entity["LOWER"] for entity in pattern]).lower()
    matcher.add(pattern_name, None, pattern)

infoDF = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'ner_info_db.pandas'))

targ = len(infoDF)

TRAIN_DATA = list()

ent_counter = dict()

#forbidden_dep = ['csubj', 'nummod', 'cc', 'advmod', 'preconj', 'attr', 'det']
#forbidden_pos = ['VERB', 'ADP']

forbidden_dep = ['det', 'predet', 'nummod', 'cc', 'appos', 'punct', 'conj']
forbidden_pos = ['ADP', 'VERB', 'X']

collect_ml = set()
lt = LoopTimer(update_after=200, avg_length=2000, target=targ)
for abstract_id, row in infoDF.iterrows():
    ori_doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))

    for sent in ori_doc.sents:
        sentence = sent.as_doc()

        matches = matcher(sentence)

        ent_list = list()

        for match in matches:
            start = match[1]
            end = match[2]

            #print(f"'{sentence.text}'")

            token_set = set()

            for i in range(start, end):
                token_set.add(sentence[i])

            algo_set = set()

            visited = set()

            while len(token_set) > 0:
                token = token_set.pop()
                visited.add(token)
                algo_set.add(token.i)
                #print(token)
                if token.dep_ == "compound":
                    if (token.head not in visited
                            and token.head not in token_set):
                        #print(f"Adding '{token.head}' because of compound. '{token.dep_}'")
                        token_set.add(token.head)
                children = token.children
                for child in children:
                    if (child.dep_ not in forbidden_dep
                            and child.pos_ not in forbidden_pos
                            and child not in visited
                            and child not in token_set):
                        #print(f"Adding child '{child}' <- {child.dep_} <- '{token}'")
                        #print(f"POS: {child.pos_}")
                        #print(f"{child.tag_}")
                        token_set.add(child)
                #print("|==========|")
            algo_list = list(algo_set)
            algo_list.sort()

            start_id = min(algo_list)
            end_id = max(algo_list)

            entity_name = sentence[start_id:end_id+1].text.lower()
            if ',' not in entity_name and '(' not in entity_name and ')' not in entity_name:
                start_idx = sentence[start_id].idx
                end_idx = sentence[end_id].idx + len(sentence[end_id])
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

print()
for entity in ent_counter:
    print(f"{entity}: {ent_counter[entity]}")

with open(os.path.join(path_to_db, "NER", "spacy_ner_mlalgo_traindata_all.pickle"), "wb") as handle:
    pickle.dump(TRAIN_DATA, handle)