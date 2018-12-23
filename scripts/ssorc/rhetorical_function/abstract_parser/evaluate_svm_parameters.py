import os

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy.sparse
import numpy as np

import src.utils.functions as util

reg_paras = [100000, 200000, 500000, 1000000, 5000000]

path_to_db = "/media/norpheo/mySQL/db/ssorc"

feature_set_name = "rf_medium_lcpupbwuwb"

feature_file = os.path.join(path_to_db, "features", f"{feature_set_name}_features.npz")
target_file = os.path.join(path_to_db, "features", f"{feature_set_name}_targets.npy")

all_features = scipy.sparse.load_npz(feature_file)
all_targets = np.load(target_file)

print("Feature-Vector-Shape: " + str(all_features.shape))

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_targets,
                                                                                            test_size=0.4,
                                                                                            random_state=42,
                                                                                            shuffle=True)
top_score = 0

for c_para in reg_paras:
    model = svm.SVC(decision_function_shape='ovo', C=c_para, kernel='rbf')
    scores = util.measure(lambda: cross_val_score(model, learning_features, learning_targets, cv=10, n_jobs=-1),
                          "Cross Val on C = " + str(c_para))
    print("Score: " + str(scores.mean()))
    if scores.mean() > top_score:
        top_score = scores.mean()
        best_reg_para = c_para

print()
print("Best Reg-Para: " + str(best_reg_para))
print()

best_model = svm.SVC(decision_function_shape='ovo', C=best_reg_para, kernel='rbf')
best_model.fit(learning_features, learning_targets)

prediction = best_model.predict(holdback_features)

labels = list(set(holdback_targets))

relevant = {}
retrieved = {}
rel_ret = {}

for label in labels:
    relevant[label] = 0
    retrieved[label] = 0
    rel_ret[label] = 0

print("Occurences of Labels (Relevant | Retrieved)")
for label in labels:
    relevant[label] = np.sum(holdback_targets == label)
    retrieved[label] = np.sum(prediction == label)

    print(label[0:5] + ": " + str(relevant[label]) + " | " + str(retrieved[label]))

for i in range(0, len(holdback_targets)):
    label = holdback_targets[i]
    if holdback_targets[i] == prediction[i]:
        rel_ret[label] += 1

print()
print("[Results] - Score: " + str(best_model.score(holdback_features, holdback_targets)))
print("     \tPreci | Recall")
for label in labels:
    if retrieved[label] == 0:
        prec = 0.0
    else:
        prec = rel_ret[label] / retrieved[label]
    recall = rel_ret[label] / relevant[label]
    print(label[0:5] + ":\t" + str(prec)[0:5] + " | " + str(recall)[0:5])

