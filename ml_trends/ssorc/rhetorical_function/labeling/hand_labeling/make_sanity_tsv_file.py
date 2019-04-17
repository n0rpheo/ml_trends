import os
import pickle
import pandas as pd

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
otdb_path = os.path.join(path_to_db, 'pandas', 'ml_ot.pandas')

rule_target_path = os.path.join(path_to_db, 'features', 'rf_targets.pickle')

output_file_path = os.path.join(path_to_db, 'rf_hand_labels', 'unlabeled_sanity_data.tsv')

print("Loading Panda DB")
otDF = pd.read_pickle(otdb_path)
print("Done Loading")

with open(rule_target_path, 'rb') as target_file:
    rule_targets = pickle.load(target_file)

lc = LoopTimer(update_after=500, avg_length=3000)

counter = 0

with open(output_file_path, 'w') as label_file:
    for abstract_id, row in otDF.iterrows():
        sentences = row['originalText'].split("\t")

        for s_id, sentence in enumerate(sentences):
            askey = (abstract_id, s_id)

            if askey in rule_targets:
                counter += 1
                line = f"{abstract_id}\t{s_id}\t{sentence}\t0\n"
                label_file.write(line)

                if counter > 5000:
                    exit()
        lc.update("Make TSV File")