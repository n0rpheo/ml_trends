import os
import pickle
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

import pandas as pd
from src.utils.LoopTimer import LoopTimer


def is_sublist(ls1, ls2):
    for word_b_idx in range(len(ls2) - len(ls1) + 1):
        for word_a_idx in range(len(ls1)):
            word_a = ls1[word_a_idx]
            word_b = ls2[word_b_idx + word_a_idx]

            if word_a != word_b:
                break
        else:
            return True
    return False

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations")
pandas_path = os.path.join(path_to_db, "pandas")
path_to_ner = os.path.join(path_to_db, "NER")

nlp = spacy.load(os.path.join(path_to_db, "models", "en_core_web_sm_nertrained"))
vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "spacy.vocab"))

with open(os.path.join(path_to_ner, "ml_algos.txt"), "r") as handle:
    ml_algos = set()
    ml_algos_list = list()
    for line in handle:
        algo = line.strip().lower()
        if algo not in ml_algos:
            ml_algos.add(algo)
            ml_algos_list.append(algo.split(" "))

print(len(ml_algos))

for i in range(len(ml_algos_list)):
    for j in range(len(ml_algos_list)):
        if i != j:
            algo1 = ml_algos_list[i]
            algo2 = ml_algos_list[j]

            if is_sublist(algo1, algo2):
                algo1 = ' '.join(algo1)
                algo2 = ' '.join(algo2)
                ml_algos.discard(algo2)

print(len(ml_algos))


matcher = Matcher(vocab)
for ml_algo in ml_algos:
    doc = nlp(ml_algo)
    pattern = [{"LOWER": token.lower_} for token in doc]
    pattern_name = "".join([entity["LOWER"] for entity in pattern])
    matcher.add(pattern_name, None, pattern)

doc = nlp("This is a support vector Machines.")

matches = matcher(doc)

for match in matches:
    match_id = match[0]
    match_start = match[1]
    match_end = match[2]
    print(f"{match_id} - {match_start} - {match_end}")
    print(doc[match[1]:match[2]])
