import os
import spacy
import pickle

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"

nlp = spacy.load(os.path.join(path_to_db, "models", "en_core_web_sm_nertrained_v3"))
#nlp = spacy.load("en_core_web_sm")

# training data
with open(os.path.join(path_to_db, "NER", "spacy_ner_mlalgo_traindata_th1_all.pickle"), "rb") as handle:
    TRAIN_DATA = pickle.load(handle)

n_tp = 0
n_fp = 0
n_fn = 0
n_inter = 0
n = 0

fp_list = set()
fn_list = set()

lt = LoopTimer(update_after=10, avg_length=100000, target=len(TRAIN_DATA))
for td in TRAIN_DATA:
    text = td[0]
    entities = td[1]['entities']

    e_ranges = list()
    e_ranges_gold = list()
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == 'MLALGO':
            e_ranges.append((ent.start_char, ent.end_char))

    for entity in entities:
        e_ranges_gold.append((entity[0], entity[1]))

    for er in e_ranges:
        breaker = False
        for gr in e_ranges_gold:
            if er[0] == gr[0] and er[1] == gr[1]:
                n_tp += 1
                breaker = True
                break
            elif er[0] <= gr[1] and er[1] >= gr[0]:
                n_inter += 1
                breaker = True
                break
        if not breaker:
            fp_list.add(doc.text[er[0]:er[1]])
            n_fp += 1

    for gr in e_ranges_gold:
        breaker = False
        for er in e_ranges:
            if er[0] == gr[0] and er[1] == gr[1]:
                breaker = True
                break
            elif er[0] <= gr[1] and er[1] >= gr[0]:
                breaker = True
                break
        if not breaker:
            fn_list.add(doc.text[gr[0]:gr[1]])
            n_fn += 1
    """
    for e_range in e_ranges:
        if e_range in e_ranges_gold:
            n_tp += 1
        else:
            fp_list.add(doc.text[e_range[0]:e_range[1]])
            n_fp += 1

    for e_range_gold in e_ranges_gold:
        if e_range_gold not in e_ranges:
            n_fn += 1
            fn_list.add(text[e_range_gold[0]:e_range_gold[1]])
    """

    lt.update(f"Test - tp: {n_tp}, fp: {n_fp}, fn: {n_fn}, inter: {n_inter}")

print()
print(f"TP: {n_tp}")
print(f"FP: {n_fp}")
print(f"FN: {n_fn}")

print()
print("False Positives")
print("===============")
for ent in fp_list:
    print(f"{ent}")

print()
print("False Negatives")
print("===============")
for ent in fn_list:
    print(f"{ent}")
