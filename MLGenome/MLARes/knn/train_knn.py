import pickle
import os
import numpy as np
from sklearn.cluster import KMeans

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_core_web_lg_mlalgo_v1"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_mlgenome_features = os.path.join(path_to_mlgenome, "features")

with open(os.path.join(path_to_mlgenome, "unique_mentions.pickle"), "rb") as handle:
    mentions = pickle.load(handle)

feature_vector = []

lt_target = len(mentions)
for mention in mentions:
    m_string = mention["string"]
    m_is_acronym = mention['is_acronym']
    m_vec = mention['m_vec']

    #if m_is_acronym:
    #    continue

    feature_vector.append(m_vec)

X = np.array(feature_vector)
print(X.shape)

num_clusters = 100
print("Learning")
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
print(kmeans.labels_)

clusters = [set() for i in range(num_clusters)]

cluster_names = [None for i in range(num_clusters)]

print("Labeling")
for mention in mentions:
    m_string = mention["string"]
    m_is_acronym = mention['is_acronym']
    m_vec = mention['m_vec']

    if m_is_acronym:
        continue

    x = [np.array(m_vec)]
    cluster = kmeans.predict(x)[0]
    cluster_score = kmeans.score(x)

    clusters[cluster].add(m_string)

    if cluster_names[cluster] is None:
        cluster_names[cluster] = {"name": m_string, "score": cluster_score}
    else:
        old_score = cluster_names[cluster]["score"]
        if cluster_score > old_score:
            cluster_names[cluster] = {"name": m_string, "score": cluster_score}


for idx, cluster in enumerate(clusters):
    print(f"{cluster_names[idx]['name']}:\t\t\t{cluster}")