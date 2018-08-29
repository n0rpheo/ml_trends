from src.features.ap_features import build_feature_file
from src.features.ap_features import get_features_from_file
from src.modules.abstract_parser import train_models
from src.utils.functions import save_sk_model

print("Building Feature File")
build_feature_file(dtype='ssorc')

print("Feature File into fast readable Feature File")
targets, features = get_features_from_file(num_samples=200000,
                                           dtype='ssorc',
                                           feature_set='location,wordunigram,wordbigram,posunigram,posbigram,concreteness',
                                           file_suffix='all-200k-feat')

print("Start Training")
model = train_models([100, 200],  # Best @ 10000 ~ 80.6%
                     dtype="ssorc",
                     load_features=True,
                     load_suffix="all-200k-feat")

print("Save Model")
save_sk_model(model, 'svm_ap_all-200k-feat.model')