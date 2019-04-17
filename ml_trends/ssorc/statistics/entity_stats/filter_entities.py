import os
import pickle
import operator

from src.utils.LoopTimer import LoopTimer

panda_path = "/media/norpheo/mySQL/db/ssorc/pandas"

with open(os.path.join(panda_path, "dataDict.pickle"), "rb") as ml_file:
    dataDict = pickle.load(ml_file)

ml_rate = dict()
for item in dataDict:
    summation = (dataDict[item]['wML'] + dataDict[item]['woML'])
    if summation > 10000:
        ml_rate[item] = dataDict[item]['wML'] / summation

sorted_list = sorted(ml_rate.items(), key=operator.itemgetter(1), reverse=False)

for idx, item in enumerate(sorted_list):
    entity = item[0]
    rate = item[1]
    entry = dataDict[entity]

    #print(f"{entity} - {rate}:\t\t{entry}")
    #if idx > 500:
    #    break
    if rate > 0.03:
        break
print(idx)
