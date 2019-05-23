import os
import pandas as pd
import pickle

from spacy.tokens import Doc
from spacy.vocab import Vocab


from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_core_web_lg_mlalgo_v1"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)
with open(os.path.join(path_to_mlgenome, "acronyms.pickle"), "rb") as handle:
    acronym_dictionary = pickle.load(handle)

vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

targ = len(infoDF)

lt = LoopTimer(update_after=10, avg_length=1000, target=targ)

acronyms = set()
adds = list()

for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    for ent in doc.ents:
        entity = ent.text.lower()

        if entity in acronym_dictionary:
            acronyms.add(entity)
            continue

        added = 0

        for acronym in acronym_dictionary:
            long_forms = acronym_dictionary[acronym]

            if entity in long_forms:
                acronyms.add(acronym)
                added += 1

        if added > 0:
            adds.append(added)
    lt.update(f"Get Acronyms - {len(acronyms)}")

for acronym in acronyms:
    print(f"{acronym}: {acronym_dictionary[acronym]}")

print(max(adds))
print(sum(adds) / len(adds))

