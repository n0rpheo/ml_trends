import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
feature_file_path = os.path.join(path_to_db, 'popularities', 'pop_feat.pandas')

featureFrame = pd.read_pickle(feature_file_path)

flength = len(featureFrame.columns)

split = np.hsplit(featureFrame.values, np.array([flength-1, flength]))

features = split[0]
targets = np.array(split[1].T)[0]

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(features,
                                                                                            targets,
                                                                                            test_size=0.4,
                                                                                            random_state=42,
                                                                                            shuffle=True)
lc = LoopTimer(update_after=10, avg_length=500, target=950)

best_para = 0
best_score = 0

for c_para in range(1, 1000):
    lr_model = LogisticRegression(random_state=0,
                                  solver='lbfgs',
                                  multi_class='multinomial',
                                  verbose=0,
                                  C=c_para,
                                  max_iter=1000)

    scores = cross_val_score(estimator=lr_model,
                             X=learning_features,
                             y=learning_targets,
                             cv=10,
                             n_jobs=-1,
                             scoring='f1_micro')
    mean_score = scores.mean()

    if mean_score > best_score:
        best_score = mean_score
        best_para = c_para

    lc.update(f"Best Para: {best_para} - Best Score: {best_score}")

print()
print(f"Best Para: {best_para}")
lr_model = LogisticRegression(random_state=0,
                              solver='lbfgs',
                              multi_class='multinomial',
                              verbose=0,
                              C=best_para,
                              max_iter=1000)

lr_model.fit(learning_features, learning_targets)
prediction = lr_model.predict(holdback_features)

labels = list(set(holdback_targets))
num_labels = len(labels)

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

    print(f"{label}: {relevant[label]} | {retrieved[label]}")

for i in range(0, len(holdback_targets)):
    label = holdback_targets[i]
    if holdback_targets[i] == prediction[i]:
        rel_ret[label] += 1

print()
print("[Results] - Score: " + str(lr_model.score(holdback_features, holdback_targets)))
print("     \tPreci | Recall")
for label in labels:
    if retrieved[label] == 0:
        prec = 0.0
    else:
        prec = rel_ret[label] / retrieved[label]
    recall = rel_ret[label] / relevant[label]
    print(f"{label}:\t{str(prec)[0:5]} | {str(recall)[0:5]}")