import os
from sklearn.decomposition import LatentDirichletAllocation
import scipy.sparse
import pickle

path_to_db = "/media/norpheo/mySQL/db/ssorc"


"""
 #
 # Infos
 #
"""

nlp_model = "en_wa_v2"
n_topics = 300
info_file_name = "merged_word_vec.info"

iterations = 100

"""
 #
 #
 #
"""

path_to_tm = os.path.join(path_to_db, "topic_modeling", nlp_model)
path_to_model = os.path.join(path_to_tm, "model")
if not os.path.isdir(path_to_model):
    print(f"Create Directory {path_to_model}")
    os.mkdir(path_to_model)

info_name_output = f"{info_file_name[:-5]}_{n_topics}.info"
model_file_name = f"{info_file_name[:-5]}_{n_topics}topics.model"
with open(os.path.join(path_to_tm, info_file_name), "rb") as handle:
    info = pickle.load(handle)

info['model_path'] = os.path.join(path_to_model, model_file_name)

tm_features = scipy.sparse.load_npz(info['feature_path'])

print(tm_features.shape)

lda_model = LatentDirichletAllocation(n_components=n_topics,
                                      random_state=0,
                                      learning_method='batch',
                                      verbose=1,
                                      max_iter=iterations)
print(f"Training Model. - {info_file_name[:-5]} - {n_topics}")
lda_model.fit(tm_features)

with open(info['model_path'], "wb") as handle:
    pickle.dump(lda_model, handle)

with open(os.path.join(path_to_tm, info_name_output), "wb") as handle:
    pickle.dump(info, handle)

