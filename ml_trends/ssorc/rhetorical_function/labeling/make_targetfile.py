import os
import json
import pandas as pd
import pickle
import numpy as np

from spacy.vocab import Vocab
from spacy.tokens import Doc

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)
path_to_patterns = os.path.join(path_to_db, "pattern_matching")
path_to_learned_patterns = os.path.join(path_to_patterns, nlp_model)
path_to_rfl = os.path.join(path_to_db, "rhet_func_labeling", nlp_model)

if not os.path.isdir(path_to_rfl):
    print(f"Create Directory {path_to_rfl}")
    os.mkdir(path_to_rfl)

"""

"""

feature_info_name = "feature_info.pickle"

settings = dict()
settings["feature_set"] = ["word_vector",
                           "location"
                           ]
"""
"""
target_path = os.path.join(path_to_learned_patterns, 'rf_targets.pickle')  # output!!

rules_filename = "leaned_rules.json"
rule_file_path = os.path.join(path_to_learned_patterns, rules_filename)

rules = dict()
with open(rule_file_path) as rule_file:
    for line in rule_file:
        data = json.loads(line)

        category = data['category']
        trigger_word = data['trigger_word']
        dependency = data['dependency']

        rule = (trigger_word, dependency)

        if category not in rules:
            rules[category] = set()

        rules[category].add(rule)


print("Loading Vocab...")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

db_size = len(infoDF)

lc = LoopTimer(update_after=100, avg_length=200, target=db_size)

predictions = dict()
targets = dict()

target_vector = list()
feature_vector = list()

for idx, (abstract_id, df_row) in enumerate(infoDF.iterrows()):

    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    for n_sents, sent in enumerate(doc.sents):
        pass

    if n_sents == 0:
        continue

    for s_id, sentence in enumerate(doc.sents):

        cat_assignment = list()
        for category in rules:
            for rule in rules[category]:
                trigger_word = rule[0]
                dependency = rule[1]

                for token in sentence:
                    if token.text == trigger_word:
                        for child in token.children:
                            if child.dep_ == dependency:
                                cat_assignment.append(category)

        if len(cat_assignment) > 0:
            catset = set(cat_assignment)
            top_cat = None
            top_cat_count = 0

            for category in catset:
                n_cat = cat_assignment.count(category)

                if n_cat > top_cat_count:
                    top_cat_count = n_cat
                    top_cat = category
            location = s_id / n_sents
            word_vector = sentence.vector

            features = np.append(word_vector, [location])

            target_vector.append(top_cat)
            feature_vector.append(features)

    info_count = [(cat, target_vector.count(cat)) for cat in set(target_vector)]
    breaker = lc.update(f"Make Features - {info_count}")


feature_vector = np.array(feature_vector)
target_vector = np.array(target_vector)

print(feature_vector.shape)
print(target_vector.shape)

feature_dict = dict()

feature_dict["features"] = feature_vector
feature_dict["targets"] = target_vector
feature_dict["settings"] = settings

with open(os.path.join(path_to_rfl, f"{feature_info_name}.pickle"), "wb") as handle:
    pickle.dump(feature_dict, handle)
