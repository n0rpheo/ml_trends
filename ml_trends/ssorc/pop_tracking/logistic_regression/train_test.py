import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from src.utils.LoopTimer import LoopTimer

from src.utils.functions import print_scoring

path_to_db = "/media/norpheo/mySQL/db/ssorc"
feature_file_path = os.path.join(path_to_db, 'popularities', 'pop_feat.pandas')

print(feature_file_path)

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

best_para = 0
best_score = 0

reg_paras = list()

for i in range(1, 10):
    reg_paras.append(i/10)
for i in range(0, 10):
    reg_paras.append(1+i)
for i in range(1, 10):
    reg_paras.append(10+i*10)
for i in range(1, 10):
    reg_paras.append(100+i*100)
for i in range(1, 10):
    reg_paras.append(1000+i*1000)
for i in range(1, 10):
    reg_paras.append(10000+i*10000)
for i in range(1, 10):
    reg_paras.append(100000+i*100000)

lc = LoopTimer(update_after=10, avg_length=100, target=len(reg_paras))

for c_para in reg_paras:
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
print()
lr_model = LogisticRegression(random_state=0,
                              solver='lbfgs',
                              multi_class='multinomial',
                              verbose=0,
                              C=best_para,
                              max_iter=1000)

lr_model.fit(learning_features, learning_targets)
prediction = lr_model.predict(holdback_features)


print_scoring(holdback_targets, prediction)