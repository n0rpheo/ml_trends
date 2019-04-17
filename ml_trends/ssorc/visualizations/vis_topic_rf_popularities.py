import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pickle
import pandas as pd

path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/presentation/mats"
path_to_db = "/media/norpheo/mySQL/db/ssorc"
df_file_path = os.path.join(path_to_db, "popularities",
                            "df_500topics_dist_pruned_notriggers_features_tm_lemma_pruned.pickle")

with open(df_file_path, "rb") as pif:
    dataFrames = pickle.load(pif)

num_topics = len(dataFrames)

fields = dict()
fields["nn_ids"] = [1, 95, 207, 268, 338]
fields["naive_bayes_ids"] = [22, 357]
fields["ql_ids"] = [31]
fields["adaboost_ids"] = [75]
fields["clustering_ids"] = [78, 497]
fields["dt_ids"] = [81, 392, 435]
fields["hmm_ids"] = [160]
fields["pca_ids"] = [168]
fields["svm_ids"] = [245, 393]
fields["nlp_ids"] = [271, 273, 492]
fields["lr_ids"] = [345]
fields["crf_ids"] = [410]

rf_labels = ["Design",
             "Background",
             "Objective",
             "Result",
             "Method"]

field = "nn_ids"
norm = False
sum_dt = dataFrames[fields[field][0]]

for topic_id in fields[field][1:]:
     sum_dt += dataFrames[topic_id]

if norm:
    for rf_label in rf_labels:
        sum_dt[rf_label] = sum_dt[rf_label] / sum_dt['sum']

full_range = pd.date_range(start='1950', end='2017', freq='AS')

trend_graph = sum_dt[rf_labels].sum(axis=1).loc[full_range]
bar_graph = sum_dt[rf_labels].loc[full_range]


ax = bar_graph.plot(kind='bar',
                    stacked=True,)

x_labels = [time.year for time in full_range]
ax.set_xticklabels(x_labels)

n = 5
for index, label in enumerate(ax.xaxis.get_ticklabels()):
    if index % n != 0:
        label.set_visible(False)

#ax = trend_graph.plot(kind='line')



plt.title(field)
plt.xlabel("Time")
plt.ylabel("Popularity")

#plt.savefig(os.path.join(path_to_fig_save, "nn_stacked_norm.png"))
plt.show()

