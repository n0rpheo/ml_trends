import os
import pickle


path_to_ner = "/media/norpheo/mySQL/db/ssorc/NER"

# Model 1 Files
path_to_ml_algos_save1 = os.path.join(path_to_ner, "ml_algos.dict")

# Model 2 Files - USE THIS
path_to_ml_algos_save2 = os.path.join(path_to_ner, "ml_algos_new.dict")



with open(path_to_ml_algos_save1, "rb") as ml_algos_file:
    ml_algos = pickle.load(ml_algos_file)

with open(path_to_ml_algos_save2, "rb") as ml_algos_file:
    ml_algos_new = pickle.load(ml_algos_file)

ml_algos.update(ml_algos_new)

ml_algos = sorted(((v, k) for k, v in ml_algos.items()), reverse=True)

for algo in ml_algos:
    print(f"{algo[1]}: {algo[0]}")
print()