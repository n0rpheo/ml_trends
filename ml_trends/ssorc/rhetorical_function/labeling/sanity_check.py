import os
import pickle

from src.utils.functions import Scoring

path_to_db = "/media/norpheo/mySQL/db/ssorc"
hl_target_path = os.path.join(path_to_db, 'features', 'rf_targets_hl_sanity.pickle')
rule_target_path = os.path.join(path_to_db, 'features', 'rf_targets.pickle')

with open(hl_target_path, 'rb') as target_file:
    hl_targets = pickle.load(target_file)
with open(rule_target_path, 'rb') as target_file:
    rule_targets = pickle.load(target_file)

gold_labels = list()
prediction = list()

for key in hl_targets:
    if key in rule_targets:
        if hl_targets[key].lower() != 'conclusion':
            gold_labels.append(hl_targets[key].lower())
            prediction.append(rule_targets[key].lower())


scores = Scoring(gold_labels, prediction)

scores.print()