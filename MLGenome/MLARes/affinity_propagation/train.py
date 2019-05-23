import os
import pickle
import numpy as np
import sklearn.cluster

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_core_web_lg_mlalgo_v1"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_mlgenome_features = os.path.join(path_to_mlgenome, "features")

with open(os.path.join(path_to_mlgenome_features, "affinity_propagation_features.pickle"), "rb") as handle:
    features = pickle.load(handle)

words = features["words"]
string_similarity = features["string_sim"]

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.95, max_iter=1000, verbose=True)
print("Start Training")
affprop.fit(string_similarity)

for cluster_id in np.unique(affprop.labels_):
    top_word = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster_words = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
    cluster_str = ", ".join(cluster_words)
    print(f"*{top_word}*\t\t{cluster_str}")

print(f"Num Clusters: {len(np.unique(affprop.labels_))}")

with open(os.path.join(path_to_mlgenome, "affprop.pickle"), "wb") as handle:
    pickle.dump(affprop, handle)