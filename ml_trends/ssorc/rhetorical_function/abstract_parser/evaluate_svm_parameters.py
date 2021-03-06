import os
import pickle

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from src.utils.LoopTimer import LoopTimer
from src.utils.functions import Scoring

reg_paras = list()

for i in range(1, 10):
    reg_paras.append(i/10)
for i in range(0, 10):
    reg_paras.append(1+i)
#for i in range(1, 10):
#    reg_paras.append(10+i*10)
#for i in range(1, 10):
#    reg_paras.append(100+i*100)
#for i in range(1, 10):
#    reg_paras.append(1000+i*1000)
#for i in range(1, 10):
#    reg_paras.append(10000 + i*10000)
#for i in range(1, 5):
#    reg_paras.append(100000 + i*100000)

#reg_paras = [100000, 300000, 500000]
path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"
path_to_rfl = os.path.join(path_to_db, "rhet_func_labeling", nlp_model)


"""

"""
feature_info_name = "feature_info.pickle"
"""
"""

with open(os.path.join(path_to_rfl, f"{feature_info_name}.pickle"), "rb") as handle:
    feature_dict = pickle.load(handle)

all_features = feature_dict["features"]
all_targets = feature_dict["targets"]

print("Feature-Vector-Shape: " + str(all_features.shape))

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_targets,
                                                                                            test_size=0.4,
                                                                                            random_state=4,
                                                                                            shuffle=True)
best_para = 0
best_score = 0

score_list = list()

print("Start Training:")
lc = LoopTimer(update_after=1, avg_length=5, target=len(reg_paras))
for c_para in reg_paras:
    #model = svm.SVC(decision_function_shape='ovo',
    #                C=c_para,
    #                kernel='rbf',
    #                gamma='auto')
    model = svm.SVC(kernel="linear", C=c_para, decision_function_shape='ovo')
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

#best_model = svm.SVC(decision_function_shape='ovo',
#                     C=best_para,
#                     kernel='rbf',
#                     gamma='auto')
best_model = svm.SVC(kernel="linear", C=best_para, decision_function_shape='ovo')
best_model.fit(learning_features, learning_targets)
prediction = best_model.predict(holdback_features)

scores = Scoring(holdback_targets, prediction)
scores.print()
