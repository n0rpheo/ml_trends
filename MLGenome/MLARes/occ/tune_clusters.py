import pickle
import os
import numpy as np

"""
    Cluster Tuning
"""

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

occ_file = "occ_vec_ratio_p2.pickle"

"""
    ###########
"""

occ_file_out = f"{occ_file[:-7]}_tuned.pickle"
path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
with open(os.path.join(path_to_mlgenome, occ_file), "rb") as handle:
    occ_res = pickle.load(handle)


mentions = occ_res["mentions"]
cluster_matrix = occ_res["cluster_matrix"]
sim_matrix = occ_res["similarity_matrix"]


for cid in range(cluster_matrix.shape[1]):

    if sum(cluster_matrix[:, cid]) == 1:
        m_id = np.where(cluster_matrix[:, cid] == 1)[0][0]
        m_cid = np.where(cluster_matrix[m_id, :] == 1)[0]
        if len(m_cid) > 1:
            print(f"Set {m_id} MID and {cid} to 0")
            cluster_matrix[m_id, cid] = 0


with open(os.path.join(path_to_mlgenome, os.path.join(path_to_mlgenome, occ_file_out)), "wb") as handle:
    pickle.dump(occ_res, handle)