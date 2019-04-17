import os
import pickle
import operator

from src.utils.LoopTimer import LoopTimer

panda_path = "/media/norpheo/mySQL/db/ssorc/pandas"

with open(os.path.join(panda_path, "ml_entities.pickle"), "rb") as ml_file:
    ml_words = pickle.load(ml_file)
with open(os.path.join(panda_path, "ai_entities.pickle"), "rb") as ai_file:
    ai_words = pickle.load(ai_file)

dataDict = dict()
count = 0
filename = f"all_ent_{count}.dict"

while os.path.isfile(os.path.join(panda_path, filename)):
    with open(os.path.join(panda_path, filename), "rb") as dict_file:
        dict_frame = pickle.load(dict_file)

    print(filename)
    lt = LoopTimer(update_after=50000, avg_length=50000, target=len(dict_frame))
    for abstract_id in dict_frame:
        entities = dict_frame[abstract_id]

        for entity in entities:
            if entity in ml_words:
                if entity not in dataDict:
                    dataDict[entity] = {"wML": 0, "woML": 0, "wAI": 0, "woAI": 0}

                if "machine learning" in entities:
                    dataDict[entity]["wML"] += 1
                else:
                    dataDict[entity]["woML"] += 1

            if entity in ai_words:
                if entity not in dataDict:
                    dataDict[entity] = {"wML": 0, "woML": 0, "wAI": 0, "woAI": 0}

                if "artificial intelligence" in entities:
                    dataDict[entity]["wAI"] += 1
                else:
                    dataDict[entity]["woAI"] += 1

        lt.update("Analyze Dict")

    print()
    count += 1
    filename = f"all_ent_{count}.dict"

with open(os.path.join(panda_path, "dataDict.pickle"), "wb") as dd_file:
    pickle.dump(dataDict, dd_file)

exit()

ml_rate = dict()

for item in dataDict:
    sumation = (dataDict[item]['wML'] + dataDict[item]['woML'])
    ml_rate[item] = dataDict[item]['wML'] / sumation if sumation > 0 else 0

sorted_list = sorted(ml_rate.items(), key=operator.itemgetter(1), reverse=True)

for idx, item in enumerate(sorted_list):
    entity = item[0]
    rate = item[1]
    entry = dataDict[entity]

    print(f"{entity} - {rate}: {entry}")
    if idx > 500 or rate < 0.5:
        break