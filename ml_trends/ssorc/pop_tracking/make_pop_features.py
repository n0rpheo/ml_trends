import os
import pickle
import pandas as pd

import math

import src.utils.ksc as ksc

path_to_db = "/media/norpheo/mySQL/db/ssorc"
df_file_path = os.path.join(path_to_db, "popularities", "df_500topics_dist_pruned_notriggers_features_tm_lemma_pruned.pickle")

feature_file_path = os.path.join(path_to_db, 'popularities', 'pop_feat_late.pandas')  # output
predict = True
trend_set = ['up', 'down']


with open(df_file_path, "rb") as pif:
    dataFrames = pickle.load(pif)

num_topics = len(dataFrames)
labels = [label for label in dataFrames[0]][:5]
print(labels)
columns = list()


full_range = pd.date_range(start='2000', end='2017', freq='AS')

if predict:
    d_range = pd.date_range(start='2012', end='2017', freq='AS')
    start_range = d_range[:3]
    end_range = d_range[-3:]
    for label in labels:
        columns.append(f"ldp-{label}")
        columns.append(f"ldd-{label}")
else:
    d_range = pd.date_range(start='2000', end='2017', freq='AS')
    start_range = d_range[:5]
    end_range = d_range[-5:]
    for label in labels:
        columns.append(f"ldp-{label}")
        columns.append(f"ldr-{label}")
        columns.append(f"ldd-{label}")

columns.append('trend')
trends = list()

featureFrame = pd.DataFrame(0.0, index=range(num_topics), columns=columns)
for topic in range(num_topics):
    df = dataFrames[topic]

    kspec = ksc.KSpectralCluster(time_interval=full_range, trend_types=trend_set)

    assignment = kspec.assign_cluster(df['sum'])

    featureFrame['trend'][topic] = assignment

    df['lpy'] = 0.0
    for label in labels:
        df['lpy'] += df[label]


    # Multiplikator ist notwendig, damit die Mittelwerte auf 1 summieren
    # hierfür werden die Zeilen(jahre) rausgerechnet, in denen das Topic
    # nicht vorkommt.
    #multi_all = len(df['lpy'][d_range]) / df['lpy'][d_range].astype(bool).sum(axis=0) if df['lpy'][d_range].astype(bool).sum(axis=0) > 0 else 0
    #multi_start = len(df['lpy'][start_range]) / df['lpy'][start_range].astype(bool).sum(axis=0) if df['lpy'][start_range].astype(bool).sum(axis=0) > 0 else 0
    #multi_end = len(df['lpy'][end_range]) / df['lpy'][end_range].astype(bool).sum(axis=0) if df['lpy'][end_range].astype(bool).sum(axis=0) > 0 else 0

    for label in labels:
        #df[label] = df[label] / df['lpy']
        #df[label] = df[label].fillna(0)

        df[label][d_range] = df[label][d_range] / df['sum'][d_range]

        ldp = df[label][d_range].values.mean() #  * multi_all
        ldpStart = df[label][start_range].values.mean()  # * multi_start
        ldpEnd = df[label][end_range].values.mean()  # * multi_end

        ldd = ldpEnd - ldpStart

        featureFrame[f"ldp-{label}"][topic] = ldp
        featureFrame[f"ldd-{label}"][topic] = ldd
        if not predict:
            featureFrame[f"ldr-{label}"][topic] = (ldpEnd / ldpStart) if ldpStart > 0 else 0


print(featureFrame)
featureFrame.to_pickle(feature_file_path)
