import os
import json
import operator

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

count = 0

journals = ["neurocomputing",
            "machine learning",
            "journal of machine learning research",
            "neural networks : the official journal of the international neural network society",
            "ai magazine",
            "artif. intell."]
venues = ["icml",
          "nips",
          "aaai",
          "journal of machine learning research",
          "ijcai",
          "machine learning",
          "cikm",
          "ai magazine",
          "icdm",
          "kdd",
          "uai",
          "cvpr",
          "iclr",
          "wsdm",
          "aistats"]

journal_dict = dict()
venue_dict = dict()

total_count = 0
ml_count = 0
ai_count = 0

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

                if journal in journals:
                    if journal not in journal_dict:
                        journal_dict[journal] = 0
                    journal_dict[journal] += 1

                if venue in venues:
                    if venue not in venue_dict:
                        venue_dict[venue] = 0
                    venue_dict[venue] += 1

                if venue in venues or journal in journals:
                    total_count += 1
                    if "machine learning" in entities:
                        ml_count += 1

                    if "artificial intelligence" in entities:
                        ai_count += 1

            breaker = lt.update(f"Analyze ({total_count})")


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

print()
print(f"Total Count: {total_count}")
print(f"ML Count: {ml_count} ({str((ml_count / total_count)*100)[0:5]} %)")
print(f"AI Count: {ai_count} ({str((ai_count / total_count)*100)[0:5]} %)")
