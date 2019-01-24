import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn import svm
import scipy.sparse
import numpy as np

from src.utils.functions import print_scoring


path_to_db = "/media/norpheo/mySQL/db/ssorc"

feature_set_name = "rf_hl_lcpupbwuwb"
reg_para = 150000

model_file_name = "svm_rf_lcpupbwuwb_hl.pickle"


feature_file = os.path.join(path_to_db, "features", f"{feature_set_name}_features.npz")
target_file = os.path.join(path_to_db, "features", f"{feature_set_name}_targets.npy")

all_features = scipy.sparse.load_npz(feature_file)
all_targets = np.load(target_file)

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_targets,
                                                                                            test_size=0.0,
                                                                                            random_state=42,
                                                                                            shuffle=True)

print("Learning Feature-Vector-Shape: " + str(learning_features.shape))
print("Holdback Feature-Vector-Shape: " + str(holdback_features.shape))

best_model = svm.SVC(decision_function_shape='ovo', C=reg_para, kernel='rbf')
best_model.fit(learning_features, learning_targets)

print('Save Model')
with open(os.path.join(path_to_db, "models", model_file_name), "wb") as model_file:
    pickle.dump(best_model, model_file)

if holdback_features.shape[0] > 0:
    print('Test Model')
    prediction = best_model.predict(holdback_features)
    print_scoring(holdback_targets, prediction)
