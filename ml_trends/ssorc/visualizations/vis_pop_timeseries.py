import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

from MLGenome.MLARes.occ.occ import occTopMention

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"
path_to_popularities = os.path.join(path_to_db, "popularities", nlp_model)
path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/presentation/mats"

"""
    Infos
"""

pop_file_name = "pop_timeseries_lg.pickle"  # output Filename
occ_file = "occ_vec_ratio_p2_tuned.pickle"

year_start = 2000
year_end = 2017

rf_labels = ["Method", "Objective", "Background"]

norm = True

"""
    ###################
"""
with open(os.path.join(path_to_popularities, pop_file_name), "rb") as handle:
    dataFrames = pickle.load(handle)


rf_columns = list()
sum_columns = list()
for rf_label in rf_labels:
    rf_columns.append(f"n-{rf_label}")
    sum_columns.append(f"sum-{rf_label}")

for cluster in dataFrames:
    df = dataFrames[cluster]

    if norm:
        for column in rf_columns:
            df[column] = df[column] / df['sum-cluster']
    else:
        for column in rf_columns:
            df[column] = df[column] / df['sum-all']

    full_range = pd.date_range(start='1980', end='2017', freq='AS')

    trend_graph = df[rf_columns].sum(axis=1).loc[full_range]

    bar_graph = df[rf_columns].loc[full_range]

    ax = bar_graph.plot(kind='bar',
                        stacked=True,)

    x_labels = [time.year for time in full_range]
    ax.set_xticklabels(x_labels)

    n = 5
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % n != 0:
            label.set_visible(False)

    #ax = trend_graph.plot(kind='line')

    plt.title(cluster)
    plt.xlabel("Time")
    plt.ylabel("Popularity")

    #plt.savefig(os.path.join(path_to_fig_save, "nn_stacked_norm.png"))
    plt.show()

