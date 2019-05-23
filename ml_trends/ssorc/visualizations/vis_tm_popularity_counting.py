import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from spacy.vocab import Vocab
from spacy.tokens import Doc

from src.utils.LoopTimer import LoopTimer

nlp_model = "en_wa_v2"

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/Results/fig/topic_modeling"
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)

track_topics = {"SVM": ["svm", "support vector machine"],
                "NN": ["neural network", "deep learning"],
                "PCA": ["pca", "principal component analysis"]}

timeseries = dict()
for topic in track_topics:
    timeseries[topic] = dict()
year_count = dict()

print("Loading Vocab")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

lt = LoopTimer(update_after=200, avg_length=1000, target=len(infoDF))
for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    year = row['year']
    text = doc.text.lower()

    if year not in year_count:
        year_count[year] = 0
        for topic in track_topics:
            timeseries[topic][year] = 0
    year_count[year] += 1

    for topic in track_topics:
        for topic_word in track_topics[topic]:
            if topic_word in text:
                timeseries[topic][year] += 1

    breaker = lt.update("Model Topics")
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