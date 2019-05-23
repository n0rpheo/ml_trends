import pickle
import os

from MLGenome.MLARes.occ.occ import OverlappingCorrelationClustering

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
with open(os.path.join(path_to_mlgenome, "unique_mentions.pickle"), "rb") as handle:
    mentions = pickle.load(handle)

cluster_save = os.path.join(path_to_mlgenome, "occ_vec_ratio_p1.pickle")
#cluster_save = None

print(cluster_save)
mentions = mentions
occ = OverlappingCorrelationClustering(num_clusters=len(mentions), p=1, mentions=mentions, nlp_model=nlp_model)

occ.optimize(tol=0.00000003, save_path=cluster_save)
occ.save(cluster_save)
#occ.print_result()