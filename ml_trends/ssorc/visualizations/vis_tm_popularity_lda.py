import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.modules.topic_modeling import TopicModelingLDA

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/Results/fig/topic_modeling"

tm_info_file_name = "word_lower_merged_300.info"

"""
track_topics = {"SVM": [212],
                "NeuralNetwork": [222, 233, 208, 106],
                "PCA": [189]}

track_topics = {"SVM": [66],
                "NeuralNetwork": [6, 292, 153, 65]
                }
"""

track_topics = {"SVM": [166],
                "NeuralNetwork": [50, 114, 140]
                }

tm_model = TopicModelingLDA(info_fn=tm_info_file_name)

wordDF = pd.read_pickle(os.path.join(path_to_db, "pandas", "aiml_ner_merged_word.pandas"))
infoDF = pd.read_pickle(os.path.join(path_to_db, "pandas", "ner_info_db.pandas"))
df = infoDF.join(wordDF)

timeseries = dict()
for topic in track_topics:
    timeseries[topic] = dict()
year_count = dict()
lc = LoopTimer(update_after=100, avg_length=5000, target=len(df))
for abstract_id, row in df.iterrows():

    year = row['year']
    text = row['merged_word'].replace("\t\t", "\t")
    tokens = row['merged_word'].split("\t")

    topic_dist = tm_model.get_topic_dist(tokens)

    top_n_topics = topic_dist.argsort()[::-1][:5]

    if year not in year_count:
        year_count[year] = 0
        for topic in track_topics:
            timeseries[topic][year] = 0
    year_count[year] += 1

    for topic in track_topics:
        for topic_n in track_topics[topic]:
            if topic_n in top_n_topics:
                timeseries[topic][year] += topic_dist[topic_n]

    breaker = lc.update("Model Topics")
print()

for topic in track_topics:
    years_list = [key for key in year_count.keys() if key < 2018]
    years_list.sort()
    x = np.array(years_list)
    y = np.array([timeseries[topic][year]/year_count[year] for year in years_list])

    fig, ax = plt.subplots()
    ax.plot(x, y)

    fig.suptitle(f"{topic} - LDA", fontsize=10, y=1.00)
    start, end = ax.get_xlim()
    start = int(start)
    end = int(end)
    ax.xaxis.set_ticks(np.arange(start, end, 2))

    # ax.set_yscale("log", nonposy='clip')
    plt.xticks(rotation=60)
    plt.xlim(start, 2020)
    # plt.ylim(0, 1200)
    plt.show()
    #plt.savefig(os.path.join(path_to_fig_save, f"topic_popularity_merged_lda_{topic}.png"))