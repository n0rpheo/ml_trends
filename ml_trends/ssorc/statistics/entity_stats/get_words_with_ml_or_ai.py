import pickle
import os
import pandas as pd

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
panda_path = "/media/norpheo/mySQL/db/ssorc/pandas"


ml_words = set()
ai_words = set()

ml_abstracts = dict()
ai_abstracts = dict()

count = 0
filename = f"all_ent_{count}.dict"
while os.path.isfile(os.path.join(panda_path, filename)):
    with open(os.path.join(panda_path, filename), "rb") as dict_file:
        dict_frame = pickle.load(dict_file)
    print(filename)
    lt = LoopTimer(update_after=50000, avg_length=50000, target=len(dict_frame))
    for abstract_id in dict_frame:
        entities = dict_frame[abstract_id]

        if 'machine learning' in entities:
            ml_words.update(set(entities))
            ml_abstracts[abstract_id] = entities

        if 'artificial intelligence' in entities:
            ai_words.update(set(entities))
            ai_abstracts[abstract_id] = entities

        lt.update("Parse Dict")

    print()
    count += 1
    filename = f"all_ent_{count}.dict"

print(f"Num ml words {len(ml_words)}")
print(f"Num ai words {len(ai_words)}")
print(f"Num ml abstracts {len(ml_abstracts)}")
print(f"Num ai abstracts {len(ai_abstracts)}")

with open(os.path.join(panda_path, "ml_entities.pickle"), "wb") as ml_file:
    pickle.dump(ml_words, ml_file)
with open(os.path.join(panda_path, "ai_entities.pickle"), "wb") as ai_file:
    pickle.dump(ai_words, ai_file)

with open(os.path.join(panda_path, "ml_abstracts.pickle"), "wb") as ai_file:
    pickle.dump(ml_abstracts, ai_file)
with open(os.path.join(panda_path, "ai_abstracts.pickle"), "wb") as ai_file:
    pickle.dump(ai_abstracts, ai_file)
