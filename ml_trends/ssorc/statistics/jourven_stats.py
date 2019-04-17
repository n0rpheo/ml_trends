import os
import json
import operator
import pickle

from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
raw_dir = "/media/norpheo/Daten/Masterarbeit/python_server/ml_trends/data/raw/ssorc"
file_list = sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])

panda_path = "/media/norpheo/mySQL/db/ssorc/pandas"

req_keys = ['title',
            'authors',
            'inCitations',
            'outCitations',
            'year',
            'paperAbstract',
            'id',
            'entities',
            'journalName',
            'venue']

journal_dict = dict()
venue_dict = dict()

lt = LoopTimer(update_after=50000, avg_length=500000, target=39*1000000+219709)
for filename in file_list[0:]:
    cur_path = os.path.join(raw_dir, filename)
    print()
    print(filename)
    dictFrame = dict()
    with open(cur_path) as file:
        for idx, file_line in enumerate(file):
            data = json.loads(file_line)

            if all(key in data for key in req_keys):
                entities = [entity.lower() for entity in data['entities']]

                journal = data['journalName'].lower()
                venue = data['venue'].lower()

                if journal not in journal_dict:
                    journal_dict[journal] = 0
                journal_dict[journal] += 1

                if venue not in venue_dict:
                    venue_dict[venue] = 0
                venue_dict[venue] += 1

            breaker = lt.update(f"Make Data")


sorted_venue = sorted(venue_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_journals = sorted(journal_dict.items(), key=operator.itemgetter(1), reverse=True)

print("\n\n")
print("Venues")
for item in sorted_venue[:20]:
    print(f"{item[0]}: {item[1]}")
print()
print("------------------------")
print("Journals")
for item in sorted_journals[:20]:
    print(f"{item[0]}: {item[1]}")

with open(os.path.join(path_to_db, "untitled", "journals.pickle"), 'wb') as handle:
    pickle.dump(journal_dict, handle)
with open(os.path.join(path_to_db, "untitled", "venues.pickle"), 'wb') as handle:
    pickle.dump(venue_dict, handle)