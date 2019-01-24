import os
import pandas as pd
import numpy as np

from src.utils.functions import print_scoring

path_to_db = "/media/norpheo/mySQL/db/ssorc"
feature_file_path = os.path.join(path_to_db, 'popularities', 'pop_feat.pandas')

featureFrame = pd.read_pickle(feature_file_path)

flength = len(featureFrame.columns)

split = np.hsplit(featureFrame.values, np.array([flength-1, flength]))

features = split[0]
targets = np.array(split[1].T)[0]


all_c = len(targets)
unique, counts = np.unique(targets, return_counts=True)
uni_count = dict(zip(unique, counts))

weighted_chance = [uni_count[0] / all_c, uni_count[1] / all_c, uni_count[2] / all_c]
unweighted_chance = [1/3, 1/3, 1/3]

#prediction_random = np.random.choice(3, len(targets), p=weighted_chance)
prediction_random = np.random.choice(3, len(targets), p=unweighted_chance)
prediction_majority = np.full(len(targets), 0)

print_scoring(targets, prediction_random)
#print_scoring(targets, prediction_majority)