import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn import svm

from src.utils.functions import Scoring


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_core_web_lg_mlalgo_v1"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_mlgenome_features = os.path.join(path_to_mlgenome, "features")
feature_file_path = os.path.join(path_to_mlgenome_features, "entlinking_features.pickle")

with open(os.path.join(path_to_mlgenome_features, feature_file_path), "rb") as feature_file:
    feature_dict = pickle.load(feature_file)

reg_para = 0.01

model_file_name = f"svm_mlares.pickle"

all_features = feature_dict["features"]
all_targets = feature_dict["targets"]

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_targets,
                                                                                            test_size=0.0,
                                                                                            random_state=42,
                                                                                            shuffle=True)

print("Learning Feature-Vector-Shape: " + str(learning_features.shape))
print("Holdback Feature-Vector-Shape: " + str(holdback_features.shape))


#best_model = svm.SVC(decision_function_shape='ovo',
#                     C=reg_para,
#                     kernel='rbf',
#                     gamma='auto',
#                     verbose=1)
best_model = svm.SVC(kernel="linear",
                     C=reg_para,
                     decision_function_shape='ovo',
                     verbose=1)
best_model.fit(learning_features, learning_targets)

classifier = dict()
classifier["model"] = best_model

print("\n\n")
print('Save Model')
with open(os.path.join(path_to_mlgenome, model_file_name), "wb") as model_file:
    pickle.dump(classifier, model_file)

if holdback_features.shape[0] > 0:
    print('Test Model')
    prediction = best_model.predict(holdback_features)

    scores = Scoring(holdback_targets, prediction)
    scores.print()
