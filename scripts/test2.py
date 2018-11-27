import os
import pickle

path_to_ner = "/media/norpheo/mySQL/db/ssorc/NER"

# Model 1 Files
path_to_ml_algos_save1 = os.path.join(path_to_ner, "ml_algos.dict")
path_to_ml_algo_abstract_save1 = os.path.join(path_to_ner, "ml_algo_abstract.pickle")
path_to_skip_list1 = os.path.join(path_to_ner, "skip_list.pickle")

# Model 2 Files
path_to_ml_algos_save2 = os.path.join(path_to_ner, "ml_algos_new.dict")
path_to_ml_algo_abstract_save2 = os.path.join(path_to_ner, "ml_algo_abstract_new.pickle")
path_to_skip_list2 = os.path.join(path_to_ner, "skip_list_new.pickle")


with open(path_to_skip_list1, "rb") as skip_list_file:
    skip_list_old = pickle.load(skip_list_file) # set

with open(path_to_ml_algo_abstract_save1, "rb") as algo_abstract_file:
    ml_algo_abstract_old = pickle.load(algo_abstract_file) # set

with open(path_to_ml_algos_save1, "rb") as ml_algos_file:
    ml_algos_old = pickle.load(ml_algos_file) # dict

with open(path_to_skip_list2, "rb") as skip_list_file:
    skip_list_new = pickle.load(skip_list_file) # set

with open(path_to_ml_algo_abstract_save2, "rb") as algo_abstract_file:
    ml_algo_abstract_new = pickle.load(algo_abstract_file) # set

with open(path_to_ml_algos_save2, "rb") as ml_algos_file:
    ml_algos_new = pickle.load(ml_algos_file) # dict

algos_old = set()
for algo in ml_algos_old:
    algos_old.add(algo.lower())

algos_new = set()
for algo in ml_algos_new:
    algos_new.add(algo.lower())

algo_new_wo_old = algos_new - algos_old
abstract_new_wo_old = ml_algo_abstract_new - ml_algo_abstract_old

abs_new_inter_skip_old = ml_algo_abstract_new.intersection(skip_list_old)
abs_old_inter_skip_new = ml_algo_abstract_old.intersection(skip_list_new)

number_new_has_found = len(abs_old_inter_skip_new - ml_algo_abstract_old)
number_old_has_found = len(abs_new_inter_skip_old - ml_algo_abstract_new)

print(f"Alte Skip-Liste: {len(skip_list_old)}")
print(f"Neue Skip-Liste: {len(skip_list_new)}")

print(f"Alte Abstract-Liste: {len(ml_algo_abstract_old)}")
print(f"Neue Abstract-Liste: {len(ml_algo_abstract_new)}")

print(f"Anteil Alte Liste: {(len(ml_algo_abstract_old) / len(skip_list_old))}")
print(f"Anteil Neue Liste: {(len(ml_algo_abstract_new) / len(skip_list_new))}")

print(f"Anzahl Algorithmen, die Neu mehr hat als Alt: {len(algo_new_wo_old)}")
print(f"Anzahl Abstracts, die Neu mehr hat als Alt: {len(abstract_new_wo_old)}")

print(f"Anzahl Abstracts, die Neu gefunden hat, Alt aber nicht: {number_new_has_found}")
print(f"Anzahl Abstracts, die Alt gefunden hat, Neu aber nicht: {number_old_has_found}")