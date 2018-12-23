import os
import pickle

from tabulate import tabulate

import scipy.sparse
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm

from src.utils.selector import select_path_from_dir

ftype = 'w2v'

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_mllr_model = os.path.join(path_to_db, 'models', 'mllr.joblib')
path_to_mlsvc_model = os.path.join(path_to_db, 'models', 'mlsvc.joblib')

if ftype == 'bow':
    path_to_feature_file = select_path_from_dir(os.path.join(path_to_db, 'features'),
                                                phrase="Select Feature File: ",
                                                suffix='.npz')
    path_to_target_file = select_path_from_dir(os.path.join(path_to_db, 'features'),
                                               phrase="Select Target File: ",
                                               suffix='_targets.npy')
    all_features = scipy.sparse.load_npz(path_to_feature_file)
    all_targets = np.load(path_to_target_file)
elif ftype == 'w2v':
    path_to_feature_file = select_path_from_dir(os.path.join(path_to_db, 'features'),
                                                phrase="Select Feature File: ",
                                                suffix='.pickle')
    path_to_target_file = select_path_from_dir(os.path.join(path_to_db, 'features'),
                                               phrase="Select Target File: ",
                                               suffix='_targets.pickle')

    with open(path_to_feature_file, "rb") as feature_file:
        all_features = pickle.load(feature_file)

    with open(path_to_target_file, "rb") as target_file:
        all_targets = pickle.load(target_file)

learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(all_features,
                                                                                            all_targets,
                                                                                            test_size=0.4,
                                                                                            #random_state=550,
                                                                                            shuffle=True)

sample_size = len(all_features)
iterations = 100
folds = 5

rs = ShuffleSplit(n_splits=folds, test_size=1/folds)

conf_matrix_svc = None
conf_matrix_mlc = None

count = 0

for i in range(0, iterations):
    for learn_ident, test_ident in rs.split(all_features):
        learn_features = [all_features[sample_id] for sample_id in range(0, sample_size) if sample_id in learn_ident]
        test_features = [all_features[sample_id] for sample_id in range(0, sample_size) if sample_id not in learn_ident]

        learn_targets = [all_targets[sample_id] for sample_id in range(0, sample_size) if sample_id in learn_ident]
        test_targets = [all_targets[sample_id] for sample_id in range(0, sample_size) if sample_id not in learn_ident]

        lr_model = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced')
        lr_model.fit(learn_features, learn_targets)
        svc_model = svm.SVC(kernel='linear')
        svc_model.fit(learn_features, learn_targets)

        prediction_svc = svc_model.predict(test_features)
        prediction_mlc_proba = lr_model.predict_proba(test_features)

        prediction_mlc = list()

        for pred in prediction_mlc_proba:
            if pred[1] > 0.8:
                prediction_mlc.append(1)
            else:
                prediction_mlc.append(-1)

        if conf_matrix_svc is None:
            conf_matrix_svc = confusion_matrix(test_targets, prediction_svc)
        else:
            conf_matrix_svc = conf_matrix_svc + confusion_matrix(test_targets, prediction_svc)

        if conf_matrix_mlc is None:
            conf_matrix_mlc = confusion_matrix(test_targets, prediction_mlc)
        else:
            conf_matrix_mlc = conf_matrix_mlc + confusion_matrix(test_targets, prediction_mlc)

        count += 1

conf_matrix_mlc = conf_matrix_mlc/count
conf_matrix_svc = conf_matrix_svc/count

print(tabulate(conf_matrix_svc, floatfmt=".0f", headers=('No ML Pred', 'ML Pred'), showindex=['No ML True', 'ML True']))
print()
print(tabulate(conf_matrix_mlc, floatfmt=".0f", headers=('No ML Pred', 'ML Pred'), showindex=['No ML True', 'ML True']))

print('Model Training starts.')
lr_model = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced')
lr_model.fit(all_features, all_targets)

svc_model = svm.SVC(kernel='linear')
svc_model.fit(all_features, all_targets)

print('Model Trained.')
joblib.dump(lr_model, path_to_mllr_model)
joblib.dump(svc_model, path_to_mlsvc_model)
print('Model Saved.')
