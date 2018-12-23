import os
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import src.utils.ksc as ksc

path_to_db = "/media/norpheo/mySQL/db/ssorc"
df_file_path = os.path.join(path_to_db, "popularities", "df_500topics_dist.pickle")

feature_file_path = os.path.join(path_to_db, 'popularities', 'pop_feat.pandas')

with open(df_file_path, "rb") as pif:
    dataFrames = pickle.load(pif)

num_topics = len(dataFrames)
labels = [label for label in dataFrames[0]][:-2]
columns = list()
for label in labels:
    columns.append(f"ldp-start-{label}")
    columns.append(f"ldp-end-{label}")
    #columns.append(f"ldd-start-{label}")
    #columns.append(f"ldd-end-{label}")
    columns.append(f"ldr-{label}")
columns.append('trend')

d_range = pd.date_range(start='2000', end='2015', freq='AS')

start_range = d_range[:5]
end_range = d_range[-5:]

trends = list()

featureFrame = pd.DataFrame(0.0, index=range(num_topics), columns=columns)

for topic in range(num_topics):
    df = dataFrames[topic]

    df['lpy'] = 0.0

    df['sum'] = df['sum'] / df['spy']

    kspec = ksc.KSpectralCluster(time_interval=d_range)

    assignment = kspec.assign_cluster(df['sum'])

    ldp = dict()  # % of time topic is Label x
    ldd = dict()  # % of label x going up/down

    featureFrame['trend'][topic] = assignment

    for label in labels:
        df['lpy'] += df[label]

    # Multiplikator ist notwendig, damit die Mittelwerte auf 1 summieren
    # hierfür werden die Zeilen(jahre) rausgerechnet, in denen das Topic
    # nicht vorkommt.
    multipli = len(df['lpy']) / df['lpy'].astype(bool).sum(axis=0)

    for label in labels:
        df[label] = df[label] / df['lpy']
        df[label] = df[label].fillna(0)
        ldpStart = df[label][start_range].values.mean() * multipli
        ldpEnd = df[label][end_range].values.mean() * multipli
        featureFrame[f"ldp-start-{label}"][topic] = ldpStart
        featureFrame[f"ldp-end-{label}"][topic] = ldpEnd
        featureFrame[f"ldr-{label}"][topic] = (ldpEnd / ldpStart) if ldpStart > 0 else 0

print(featureFrame)
featureFrame.to_pickle(feature_file_path)
