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

n_topics = 300
info_file_name = "word_lower_merged.info"

iterations = 200

"""
 #
 #
 #
"""

info_name_output = f"{info_file_name[:-5]}_{n_topics}.info"
model_file_name = f"{info_file_name[:-5]}_{n_topics}topics.model"
with open(os.path.join(path_to_db, "topic_modeling", info_file_name), "rb") as handle:
    info = pickle.load(handle)

info['model_path'] = os.path.join(path_to_db, "topic_modeling", "models", model_file_name)

tm_features = scipy.sparse.load_npz(info['feature_path'])

lda_model = LatentDirichletAllocation(n_components=n_topics,
                                      random_state=0,
                                      learning_method='batch',
                                      verbose=1,
                                      max_iter=iterations)
print(f"Training Model. - {info_file_name[:-5]} - {n_topics}")
lda_model.fit(tm_features)

with open(info['model_path'], "wb") as handle:
    pickle.dump(lda_model, handle)

with open(os.path.join(path_to_db, "topic_modeling", info_name_output), "wb") as handle:
    pickle.dump(info, handle)

