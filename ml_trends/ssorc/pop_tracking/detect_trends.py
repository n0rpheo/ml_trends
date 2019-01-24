import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt

import src.utils.ksc as ksc

path_to_db = "/media/norpheo/mySQL/db/ssorc"
df_file_path = os.path.join(path_to_db, "popularities", "df_500topics_dist.pickle")

with open(df_file_path, "rb") as pif:
    dataFrames = pickle.load(pif)

num_topics = len(dataFrames)

d_range = pd.date_range(start='2000', end='2015', freq='AS')

trends = list()

for topic in range(num_topics):
    df = dataFrames[topic]

    df['sum'] = df['sum'] / df['spy']

    kspec = ksc.KSpectralCluster(time_interval=d_range)

    assignment = kspec.assign_cluster(df['sum'])

    trends.append(assignment)

    #if assignment == 2:
    #    df['sum'][d_range].plot()
    #    plt.show()

print(f"Ups: {trends.count(0)}")
print(f"Evens: {trends.count(1)}")
print(f"Downs: {trends.count(2)}")

