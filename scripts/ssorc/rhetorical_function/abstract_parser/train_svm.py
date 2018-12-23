import os
import pickle

from sklearn import svm
import scipy.sparse
import numpy as np


path_to_db = "/media/norpheo/mySQL/db/ssorc"

feature_set_name = "rf_medium_lcpupbwuwb"
reg_para = 500000

model_file_name = "svm_rf_lcpupbwuwb.pickle"


feature_file = os.path.join(path_to_db, "features", f"{feature_set_name}_features.npz")
target_file = os.path.join(path_to_db, "features", f"{feature_set_name}_targets.npy")

features = scipy.sparse.load_npz(feature_file)
targets = np.load(target_file)

print("Feature-Vector-Shape: " + str(features.shape))

best_model = svm.SVC(decision_function_shape='ovo', C=reg_para, kernel='rbf')
best_model.fit(features, targets)

with open(os.path.join(path_to_db, "models", model_file_name), "wb") as model_file:
    pickle.dump(best_model, model_file)
