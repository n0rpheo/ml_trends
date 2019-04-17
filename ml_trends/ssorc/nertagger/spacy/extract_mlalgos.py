import os
import pandas as pd
import spacy
from spacy.tokens import Doc

from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations_ner")
pandas_path = os.path.join(path_to_db, "pandas")

nlp = spacy.load(os.path.join(path_to_db, "models", "en_core_web_sm_nertrained"))
vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "ner_spacy.vocab"))

infoDF = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'ner_info_db.pandas'))

targ = len(infoDF)
lt = LoopTimer(update_after=100, avg_length=10000, target=targ)

mlalgos = set()

for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    for ent in doc.ents:
        algo_name = ent.orth_.lower()
        if len(algo_name.split(" ")) <= 3:
            mlalgos.add(ent.orth_.lower())

    lt.update(f"Create MLalgo - {len(mlalgos)}")

    if len(mlalgos) > 50000:
        break

delete_set = set()
for e1 in mlalgos:
    for e2 in mlalgos:
        if e1 in e2 and e1 != e2:
            delete_set.add(e2)

print()
print(len(delete_set))

unique_mlalgos = mlalgos.difference(delete_set)
print(len(unique_mlalgos))

for algo in unique_mlalgos:
    print(algo)