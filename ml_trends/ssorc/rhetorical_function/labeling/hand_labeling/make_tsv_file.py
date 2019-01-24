import os
import pandas as pd

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
otdb_path = os.path.join(path_to_db, 'pandas', 'ml_ot.pandas')

output_file_path = os.path.join(path_to_db, 'rf_hand_labels', 'unlabeled_data.tsv')

print("Loading Panda DB")
otDF = pd.read_pickle(otdb_path)
print("Done Loading")

lc = LoopTimer(update_after=500, avg_length=3000)

with open(output_file_path, 'w') as label_file:
    for abstract_id, row in otDF.iterrows():
        sentences = row['originalText'].split("\t")

        for s_id, sentence in enumerate(sentences):
            line = f"{abstract_id}\t{s_id}\t{sentence}\t0\n"
            label_file.write(line)
        lc.update("Make TSV File")