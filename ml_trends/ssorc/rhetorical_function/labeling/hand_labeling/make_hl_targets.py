import os
import pickle

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
target_path = os.path.join(path_to_db, 'features', 'rf_targets_hl_sanity.pickle')
label_path = os.path.join(path_to_db, 'rf_hand_labels', 'sanity_data.csv')


targets = dict()
label_set = dict()
lc = LoopTimer(update_after=1000, avg_length=5000)
with open(label_path, 'r') as label_file:
    for line in label_file:
        info = line.replace('\n', '').split('\t')

        if len(info) != 4:
            print(len(info))
            print(line)
            continue

        abstract_id = info[0]
        sent_id = info[1]
        label = info[3]

        if label == '0':
            continue

        if label not in label_set:
            label_set[label] = 0

        label_set[label] += 1

        label_key = (abstract_id, int(sent_id))

        targets[label_key] = label
        lc.update("Make Targets")
print()
print(f"labels: {label_set}")
print(f"Size: {len(targets)}")

with open(target_path, 'wb') as target_file:
    pickle.dump(targets, target_file, protocol=pickle.HIGHEST_PROTOCOL)
