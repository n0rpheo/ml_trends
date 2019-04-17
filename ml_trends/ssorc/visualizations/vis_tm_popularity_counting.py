import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/Results/fig/topic_modeling"

model_file_name = "aiml_tm_lda_500topics_merged_word.pickle"
dic_file_name = "aiml_tm_dictionary_merged.dict"

track_topics = {"SVM": ["svm", "support vector machine"],
                "NN": ["neural network", "deep learning"],
                "PCA": ["pca", "principal component analysis"]}

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
    text = row['merged_word'].replace("\t\t", " ").replace("\t", " ").lower()

    if year not in year_count:
        year_count[year] = 0
        for topic in track_topics:
            timeseries[topic][year] = 0
    year_count[year] += 1

    for topic in track_topics:
        for topic_word in track_topics[topic]:
            if topic_word in text:
                timeseries[topic][year] += 1

    breaker = lc.update("Model Topics")
print()

for topic in track_topics:
    years_list = [key for key in year_count.keys() if key < 2018]
    years_list.sort()
    x = years_list
    y = [timeseries[topic][year]/year_count[year] for year in years_list]

    fig, ax = plt.subplots()
    ax.plot(x, y)

    fig.suptitle(f"{topic} - Counting", fontsize=10, y=1.00)
    start, end = ax.get_xlim()
    start = int(start)
    end = int(end)
    ax.xaxis.set_ticks(np.arange(start, end, 2))

    # ax.set_yscale("log", nonposy='clip')
    plt.xticks(rotation=60)
    plt.xlim(start, 2020)
    # plt.ylim(0, 1200)
    #plt.show()
    plt.savefig(os.path.join(path_to_fig_save, f"topic_popularity_counting_{topic}.png"))