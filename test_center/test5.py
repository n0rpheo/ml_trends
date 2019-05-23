import pickle
import os
import numpy as np

"""
    Cluster Analysis
"""

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
with open(os.path.join(path_to_mlgenome, "occ_vec_ratio_p1.pickle"), "rb") as handle:
    occ_res = pickle.load(handle)


mentions = occ_res["mentions"]
cluster_matrix = occ_res["cluster_matrix"]
sim_matrix = occ_res["similarity_matrix"]

nc = cluster_matrix.shape[1]
print(np.count_nonzero([sum(cluster_matrix[:, cid_]) for cid_ in range(nc)]))

c_list = list()
c_set = set()

for cid in range(cluster_matrix.shape[1]):

    if sum(cluster_matrix[:, cid]) > 0:
        m_ids = np.where(cluster_matrix[:, cid] == 1)[0]
        top_rep_rating = 0
        top_rep = None
        for v in m_ids:
            distance = 0
            for u in m_ids:
                if v == u:
                    continue

                d = sim_matrix[v, u]
                distance += d
            if len(m_ids) > 1:
                distance = distance / (len(m_ids)-1)
            else:
                distance = 1
            if distance > top_rep_rating:
                top_rep = mentions[v]['string']
                top_rep_rating = distance
        if top_rep_rating > 0.7 and len(m_ids) > 10:
            c_list.append(top_rep)
            c_set.add(top_rep)
        print(f"Cluster {cid}: {top_rep} ({top_rep_rating}) {[mentions[i]['string'] for i in m_ids]}")
        #print(f"Cluster {cid}: {top_rep}")
        print("----------")
