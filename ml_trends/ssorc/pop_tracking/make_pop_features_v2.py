import os
import pickle
import pandas as pd

import src.utils.ksc as ksc

from MLGenome.MLARes.occ.occ import occTopMention

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"
path_to_popularities = os.path.join(path_to_db, "popularities", nlp_model)
path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)


"""
    Infos
"""

pop_file_name = "pop_timeseries_lg.pickle"  # output Filename
occ_file = "occ_vec_ratio_p2_tuned.pickle"

year_start = 2000
year_end = 2017

predict = True
trend_set = ['up', 'down']

feature_file_name = "pop_feat.pandas"

rf_labels = ["Method", "Objective", "Background"]

"""
    ###################
"""

"""
    Loading OCC Information
"""
print("Loading Overlapping Correlation Clustering...")
otm = occTopMention(path=os.path.join(path_to_mlgenome, occ_file))

with open(os.path.join(path_to_popularities, pop_file_name), "rb") as handle:
    dataFrames = pickle.load(handle)

columns = list()

full_range = pd.date_range(start=f"{year_start}", end=f"{year_end}", freq='AS')
if predict:
    d_range = pd.date_range(start=f"{year_end-5}", end=f"{year_end}", freq='AS')
    start_range = d_range[:3]
    end_range = d_range[-3:]
    for label in rf_labels:
        columns.append(f"ldp-{label}")
        columns.append(f"ldd-{label}")
else:
    d_range = pd.date_range(start=f"{year_start}", end=f"{year_end}", freq='AS')
    start_range = d_range[:5]
    end_range = d_range[-5:]
    for label in rf_labels:
        columns.append(f"ldp-{label}")
        columns.append(f"ldr-{label}")
        columns.append(f"ldd-{label}")

columns.append('trend')
trends = list()

cluster_names = [cluster for cluster in dataFrames]

rf_columns = list()
sum_columns = list()
for rf_label in rf_labels:
    rf_columns.append(f"n-{rf_label}")
    sum_columns.append(f"sum-{rf_label}")

featureFrame = pd.DataFrame(0.0, index=cluster_names, columns=columns)

for cluster in dataFrames:
    df = dataFrames[cluster]

    df_features = df.copy()

    for column in rf_columns:
        df_features[column] = df[column] / df['sum-cluster']

    kspec = ksc.KSpectralCluster(time_interval=full_range, trend_types=trend_set)

    assignment = kspec.assign_cluster(df['sum-cluster'])

    if assignment is not None:
        featureFrame['trend'][cluster] = assignment

    for column in rf_columns:

        ldp = df[column][d_range].values.mean() #  * multi_all

        ldpStart = df[column][start_range].values.mean()  # * multi_start
        ldpEnd = df[column][end_range].values.mean()  # * multi_end

        ldd = ldpEnd - ldpStart

        featureFrame[f"ldp-{column[2:]}"][cluster] = ldp
        featureFrame[f"ldd-{column[2:]}"][cluster] = ldd
        if not predict:
            featureFrame[f"ldr-{column[2:]}"][cluster] = (ldpEnd / ldpStart) if ldpStart > 0 else 0

print(featureFrame)
featureFrame.to_pickle(os.path.join(path_to_popularities, feature_file_name))
