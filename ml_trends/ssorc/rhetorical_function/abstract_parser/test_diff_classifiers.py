import os
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src.utils.LoopTimer import LoopTimer
from src.utils.functions import Scoring

names = [
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Neural Net"
]

svclin_p = [0.025, 1, 100, 1000, 10000]
svcrbf_p = [0.025, 1, 100, 1000, 10000]
dt_p = [5, 10, 50, 100]
mlp_p = [0.01, 0.1, 1]

classifiers = [
    [(f"SVC-Lin @ {p}", SVC(kernel="linear", C=p, decision_function_shape='ovo')) for p in svclin_p],
    [(f"SVC-RBF @ {p}", SVC(kernel='rbf', gamma='auto', C=p, decision_function_shape='ovo')) for p in svcrbf_p],
    [(f"DT @ {p}", DecisionTreeClassifier(max_depth=p)) for p in dt_p],
    [(f"MLP @ {p}", MLPClassifier(alpha=p)) for p in mlp_p]
]

clfs = [element for sublist in classifiers for element in sublist]


path_to_db = "/media/norpheo/mySQL/db/ssorc"

feature_set_name = "rf_pruned_features"

with open(os.path.join(path_to_db, "features", f"{feature_set_name}.pickle"), "rb") as feature_file:
    feature_dict = pickle.load(feature_file)

all_features = feature_dict["features"]
all_targets = feature_dict["targets"]

print(f"Feature-Set: {feature_set_name}")
print("Feature-Vector-Shape: " + str(all_features.shape))

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_targets,
                                                                                            test_size=0.4,
                                                                                            random_state=4,
                                                                                            shuffle=True)
result_list = list()

lc = LoopTimer(update_after=1, avg_length=1, target=len(clfs))
for name, clf in clfs:
    lc.update(f"{name} starting")
    clf.fit(learning_features, learning_targets)
    prediction = clf.predict(holdback_features)

    result_list.append((name, prediction))

print()

for result in result_list:
    name = result[0]
    prediction = result[1]
    print(f"{name}:")
    print("----------")
    scoring = Scoring(holdback_targets, prediction)
    scoring.print()
    print("----------")
    print("----------")
    print("----------")
    print()