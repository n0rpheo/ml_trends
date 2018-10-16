import os
import pickle

from tabulate import tabulate

import scipy.sparse
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm

ftype = 'w2v'

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_mllr_model = os.path.join(path_to_db, 'models', 'mllr.joblib')
path_to_mlsvc_model = os.path.join(path_to_db, 'models', 'mlsvc.joblib')

if ftype == 'bow':
    feature_file_name = 'lr_MLclassifier_bow_features'
    path_to_feature_file = os.path.join(path_to_db, 'features', feature_file_name + '.npz')
    path_to_target_file = os.path.join(path_to_db, 'features', feature_file_name + '_targets.npy')
    all_features = scipy.sparse.load_npz(path_to_feature_file)
    all_targets = np.load(path_to_target_file)
elif ftype == 'w2v':
    feature_file_name = 'lr_MLclassifier_w2v_features'
    path_to_feature_file = os.path.join(path_to_db, 'features', feature_file_name + '.pickle')
    path_to_target_file = os.path.join(path_to_db, 'features', feature_file_name + '_targets.pickle')

    with open(path_to_feature_file, "rb") as feature_file:
        all_features = pickle.load(feature_file)

    with open(path_to_target_file, "rb") as target_file:
        all_targets = pickle.load(target_file)

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_targets,
                                                                                            test_size=0.4,
                                                                                            #random_state=550,
                                                                                            shuffle=True)
learning_features = all_features
learning_targets = all_targets

print('Model Training starts.')
lr_model = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced')
lr_model.fit(learning_features, learning_targets)

svc_model = svm.SVC(kernel='linear')
svc_model.fit(learning_features, learning_targets)

print('Model Trained.')
joblib.dump(lr_model, path_to_mllr_model)
joblib.dump(svc_model, path_to_mlsvc_model)
print('Model Saved.')

label = holdback_targets
prediction_svc = svc_model.predict(holdback_features)
prediction_mlc = lr_model.predict(holdback_features)
print(len([x for x in label if x == -1]))
conf_matrix_svc = confusion_matrix(label, prediction_svc)
conf_matrix_mlc = confusion_matrix(label, prediction_mlc)

print(tabulate(conf_matrix_svc, floatfmt=".0f", headers=('No ML Pred', 'ML Pred'), showindex=['No ML True', 'ML True']))
print()
print(tabulate(conf_matrix_mlc, floatfmt=".0f", headers=('No ML Pred', 'ML Pred'), showindex=['No ML True', 'ML True']))
