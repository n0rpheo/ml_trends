import os
import pickle
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

import pandas as pd
from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"

nlp_model = "en_wa_v2"
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)
pandas_path = os.path.join(path_to_db, "pandas")
path_to_ner = os.path.join(path_to_db, "NER")

threshold = 3

print("Loading NLP Model and Vocab")
nlp = spacy.load(os.path.join(path_to_db, "models", nlp_model))
vocab = nlp.vocab.from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
matcher = Matcher(vocab)

mla = set()
with open(os.path.join(path_to_ner, "ml_algos_noacronyms.txt"), "r") as handle:
    for line in handle:
        mla.add(line.replace("\n", ""))

for ml_algo in mla:
    ml_doc = nlp(ml_algo)
    pattern = [{"LOWER": token.lower_} for token in ml_doc]
    pattern_name = "".join([entity["LOWER"] for entity in pattern]).lower()
    matcher.add(pattern_name, None, pattern)

infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))
targ = len(infoDF)
TRAIN_DATA = list()
ent_counter = dict()

#forbidden_dep = ['csubj', 'nummod', 'cc', 'advmod', 'preconj', 'attr', 'det']
#forbidden_pos = ['VERB', 'ADP']

forbidden_dep = ['det', 'predet', 'nummod', 'cc', 'appos', 'punct', 'conj']
forbidden_pos = ['ADP', 'VERB', 'X', 'ADV']

forbidden_substrings = ['state-of-the-art', ',', '(', ')',
                        "approaches",
                        "approach",
                        "algorithm",
                        "algorithms",
                        "based",
                        "function",
                        "functions",
                        "other",
                        "large",
                        "larger",
                        "twitter",
                        "such"]

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
            entity_text_tokens = [etoken.orth_.lower() for etoken in sentence[start_id:end_id + 1]]
            entity_pos_tokens = [etoken.pos_ for etoken in sentence[start_id:end_id + 1]]
            if (all([(forbidden_string not in entity_text_tokens) for forbidden_string in forbidden_substrings]) and
                    all([(forbidden_pos_t not in entity_pos_tokens) for forbidden_pos_t in forbidden_pos]) and
                    ("markov model" in entity_name) ** ("model" in entity_name or "models" in entity_name)):
                start_idx = sentence[start_id].idx
                end_idx = sentence[end_id].idx + len(sentence[end_id])

                if entity_name in ent_counter:
                    en_idx_list = ent_counter[entity_name]["list"]
                    en_count = ent_counter[entity_name]["counter"]
                else:
                    en_idx_list = list()
                    en_count = 0

                if en_count < 50000:

                    en_count += 1
                    en_idx_list.append({"sid": len(TRAIN_DATA), "entid": len(ent_list)})

                    ent_counter[entity_name] = {"list": en_idx_list, "counter": en_count}

                    entity = (start_idx, end_idx, "MLALGO")
                    ent_list.append(entity)

                #print(f"Add: {entity_name}")
                #pos_string = " ".join([sentence[tid].tag_ for tid in range(start_id, end_id+1)])
                #print(f'     {pos_string}')
        if len(ent_list) > 0:
            entities = {"entities": ent_list}
            TRAIN_DATA.append((sentence.text, entities))

    breaker = lt.update(f"Make TD - {len(TRAIN_DATA)}")

    #if breaker > 100000:
    #    break

print()
for entity_name in ent_counter:
    en_count = ent_counter[entity_name]["counter"]
    en_list = ent_counter[entity_name]["list"]
    if en_count <= threshold:
        for entry in en_list:
            train_id = entry["sid"]
            ent_id = entry["entid"]

            TRAIN_DATA[train_id][1]["entities"][ent_id] = None
    else:
        print(f"{entity_name}: {en_count}")


for i in range(len(TRAIN_DATA)):
    td = TRAIN_DATA[i]
    for j in range(td[1]['entities'].count(None)):
        td[1]["entities"].remove(None)

    if len(td[1]["entities"]) == 0:
        TRAIN_DATA[i] = None

print()

for j in range(TRAIN_DATA.count(None)):
    TRAIN_DATA.remove(None)
print(len(TRAIN_DATA))

exit()
with open(os.path.join(path_to_ner, f"mlalgo_traindata_th{threshold}_noacronyms.pickle"), "wb") as handle:
    pickle.dump(TRAIN_DATA, handle)