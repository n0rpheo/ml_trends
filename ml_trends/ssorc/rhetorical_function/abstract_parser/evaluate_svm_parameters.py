import os

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy.sparse
import numpy as np

from src.utils.LoopTimer import LoopTimer
from src.utils.functions import print_scoring

reg_paras = list()

#for i in range(1, 10):
#    reg_paras.append(i/10)

#for i in range(0, 10):
#    reg_paras.append(1+i)
for i in range(1, 10):
    reg_paras.append(10+i*10)
for i in range(1, 10):
    reg_paras.append(100+i*100)
for i in range(1, 10):
    reg_paras.append(1000+i*1000)
for i in range(1, 10):
    reg_paras.append(10000 + i*10000)


#reg_paras = [100000, 300000, 500000]
path_to_db = "/media/norpheo/mySQL/db/ssorc"

feature_set_name = "rf_balanced_pruned_lcpupbwuwb"

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
best_para = 0
best_score = 0

score_list = list()

print("Start Training:")
lc = LoopTimer(update_after=1, avg_length=5, target=len(reg_paras))
for c_para in reg_paras:
    model = svm.SVC(decision_function_shape='ovo',
                    C=c_para,
                    kernel='rbf')
    scores = cross_val_score(model,
                             learning_features,
                             learning_targets,
                             cv=10,
                             n_jobs=-1)
    mean_score = scores.mean()

    if mean_score > best_score:
        best_score = mean_score
        best_para = c_para

    score_list.append((c_para, mean_score))

    lc.update(f"Best Para: {best_para} - Best Score: {best_score}")

print()
print(score_list)
print()
print("Best Reg-Para: " + str(best_para))
print()

best_model = svm.SVC(decision_function_shape='ovo', C=best_para, kernel='rbf')
best_model.fit(learning_features, learning_targets)

prediction = best_model.predict(holdback_features)

print_scoring(holdback_targets, prediction)
