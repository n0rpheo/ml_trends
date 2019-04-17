import os
import pandas as pd
import numpy as np

from src.utils.functions import Scoring

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

weighted_chance = [uni_count[i] / all_c for i in unique]
unweighted_chance = [1/len(unique) for i in unique]

prediction_random = np.random.choice(len(unique), len(targets), p=weighted_chance)
#prediction_random = np.random.choice(len(unique), len(targets), p=unweighted_chance)
prediction_majority = np.full(len(targets), 0)

scoring = Scoring(targets, prediction_random)
scoring.print()
#print_scoring(targets, prediction_majority)
