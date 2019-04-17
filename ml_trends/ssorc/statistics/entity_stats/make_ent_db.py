import os
import json
import pickle

from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
raw_dir = "/media/norpheo/mySQL/db/raw/ssorc"
file_list = sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])

panda_path = "/media/norpheo/mySQL/db/ssorc/pandas"

req_keys = ['title',
            'authors',
            'inCitations',
            'outCitations',
            'year',
            'paperAbstract',
            'id',
            'entities']
categories = ['machine learning',
              'artificial intelligence']

count = 0

for filename in file_list[0:1]:
    cur_path = os.path.join(raw_dir, filename)
    print()
    print(filename)
    dictFrame = dict()
    lt = LoopTimer(update_after=1000, avg_length=50000, target=1000000)
    with open(cur_path) as file:
        for idx, file_line in enumerate(file):
            data = json.loads(file_line)

            if all(key in data for key in req_keys):
                abstract_id = data['id']
                entities = data['entities']

                elist = set([entity.lower() for entity in entities])

                dictFrame[abstract_id] = elist

            lt.update(f"Make Data")

    filename = f"all_ent_{count}.dict"
    with open(os.path.join(panda_path, filename), "wb") as dict_file:
        pickle.dump(dictFrame, dict_file)
    count += 1
