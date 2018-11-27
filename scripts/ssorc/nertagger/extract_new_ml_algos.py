import os
import pickle

path_to_ner = "/media/norpheo/mySQL/db/ssorc/NER"
path_to_ml_algos_save = os.path.join(path_to_ner, "ml_algos.dict")

if os.path.isfile(path_to_ml_algos_save):
    with open(path_to_ml_algos_save, "rb") as ml_algos_file:
        ml_algos = pickle.load(ml_algos_file)
else:
    ml_algos = dict()

with open(os.path.join(path_to_ner, "new_ml_algos.txt"), "w") as new_algo_file:
    for entry, value in sorted(ml_algos.items()):
        line = f"{entry}\n"
        new_algo_file.write(line)