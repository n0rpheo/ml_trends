import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from spacy.vocab import Vocab
from spacy.tokens import Doc

from src.modules.topic_modeling import TopicModelingLDA
from src.utils.LoopTimer import LoopTimer


nlp_model = "en_wa_v2"
info_name = "merged_word_vec_300.info"

track_topics_lda = {"SVM": [45],
                    "Neural Networks": [135, 198, 201, 206, 293],
                    #"PCA": [143],
                    #"KNN": [223]
                    }

track_clusters = {"SVM": [864],
                  "Neural Networks": [49, 127, 1529],
                  }

track_topics_counting = {"SVM": ["svm", "support vector machine"],
                         "Neural Networks": ["neural network", "deep learning", "deep network"],
                         #"PCA": ["pca", "principal component analysis"],
                         #"KNN": ["k-nearest neighbor", "k nearest neighbor", "knn"]
                         }

output_prefix = "vector_v2_"

"""
==========================================
"""

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/Results/fig/topic_modeling"
path_to_tm = os.path.join(path_to_db, "topic_modeling", nlp_model)
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)

tm_path_to_info = os.path.join(path_to_tm, info_name)
tm_model = TopicModelingLDA(path_to_info=tm_path_to_info)

"""
    Init timeseries
"""
timeseries_lda = dict()
timeseries_counting = dict()
for topic in track_topics_counting:
    timeseries_lda[topic] = dict()
    timeseries_counting[topic] = dict()
year_count = dict()

print("Loading Vocab")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

print("Starting...")

lt = LoopTimer(update_after=1, avg_length=1000, target=len(infoDF))
for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    year = row['year']
    text = doc.text.lower()

    topic_dist = tm_model.get_topic_dist(doc)

    n = 5
    top_n_topics = topic_dist.argsort()[::-1][:n]
    top_n_clusters = [tm_model.topic2cluster[tid] if tid in tm_model.topic2cluster else None for tid in top_n_topics]

    if year not in year_count:
        year_count[year] = 0
        for topic in track_topics_lda:
            timeseries_lda[topic][year] = 0
            timeseries_counting[topic][year] = 0
    year_count[year] += 1

    """
        LDA / Counting Approach
    """
    for topic in track_topics_lda:
        """
            Gerneral Counting
        for topic_word in track_topics_counting[topic]:
            if topic_word in text:
                timeseries_counting[topic][year] += 1
                
            General LDA
        
        for topic_n in track_topics_lda[topic]:
            if topic_n in top_n_topics:
                timeseries_lda[topic][year] += topic_dist[topic_n]
        """

        """
            Cluster Counting
        """
        cluster_words = list()
        for cluster_id in track_clusters[topic]:
            cluster_words.extend([tm_model.otm.mentions[mention_id]["string"] for mention_id in np.where(tm_model.otm.clusters[:, cluster_id] == 1)[0]])

        for c_word in cluster_words:
            if c_word in text:
                timeseries_counting[topic][year] += 1

        """
            Cluster LDA
        """
        for cluster_id in track_clusters[topic]:
            if cluster_id in top_n_clusters:
                top_cluster_topics = np.where(np.array(top_n_clusters) == cluster_id)[0]
                for topic_n in top_cluster_topics:
                    timeseries_lda[topic][year] += topic_dist[topic_n]

    breaker = lt.update("Model Topics")
    if breaker % 100 == 0 or breaker > lt.target - 10:
        for topic in track_topics_lda:
            """
                Collecting Information
            """
            years_list = [key for key in year_count.keys() if key < 2018]
            years_list.sort()
            x = np.array(years_list)
            y_lda = np.array([timeseries_lda[topic][year] / year_count[year] for year in years_list])
            y_counting = np.array([timeseries_counting[topic][year] / year_count[year] for year in years_list])

            norm_lda = max(y_lda) if max(y_lda) > 0 else 1
            norm_counting = max(y_counting) if max(y_counting) > 0 else 1

            y_lda = y_lda / norm_lda
            y_counting = y_counting / norm_counting

            """
                Individual Plots
            """
            topic_fig, topic_ax = plt.subplots()
            topic_ax.plot(x, y_lda, label="LDA")
            topic_ax.plot(x, y_counting, label="Counting")

            topic_fig.suptitle(f"{topic} - LDA+Counting", fontsize=10, y=1.00)
            start, end = topic_ax.get_xlim()
            start = int(start)
            end = int(end)
            topic_ax.xaxis.set_ticks(np.arange(start, end, 2))

            # ax.set_yscale("log", nonposy='clip')
            plt.xticks(rotation=60)
            plt.xlim(start, 2020)
            # plt.ylim(0, 1200)
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(os.path.join(path_to_fig_save, f"{output_prefix}topop_ldacounting_{topic}.png"))
            plt.close(topic_fig)

        counting_fig, counting_ax = plt.subplots()
        lda_fig, lda_ax = plt.subplots()

        counting_lines = list()
        count_topics = list()
        lda_lines = list()

        for topic in track_topics_lda:
            """
                Collecting Information
            """
            years_list = [key for key in year_count.keys() if key < 2018]
            years_list.sort()
            x = np.array(years_list)
            y_lda = np.array([timeseries_lda[topic][year] / year_count[year] for year in years_list])
            y_counting = np.array([timeseries_counting[topic][year] / year_count[year] for year in years_list])

            counting_ax.plot(x, y_counting, label=f"{topic}")
            count_topics.append(topic)

            lda_ax.plot(x, y_lda, label=f"{topic}")

        """
            Counting Fig
        """
        counting_fig.suptitle(f"All Topics - Counting", fontsize=10, y=1.00)
        start, end = counting_ax.get_xlim()
        start = int(start)
        end = int(end)
        counting_ax.set_xticks(np.arange(start, end, 2))
        for tick in counting_ax.get_xticklabels():
            tick.set_rotation(60)
        counting_fig.legend(loc='upper right')
        counting_fig.savefig(os.path.join(path_to_fig_save, f"{output_prefix}topic_popularity_counting.png"))
        plt.close(counting_fig)

        """
            LDA Fig
        """
        lda_fig.suptitle(f"All Topics - LDA", fontsize=10, y=1.00)
        start, end = lda_ax.get_xlim()
        start = int(start)
        end = int(end)
        lda_ax.set_xticks(np.arange(start, end, 2))
        for tick in lda_ax.get_xticklabels():
            tick.set_rotation(60)

        lda_fig.legend(loc='upper right')
        lda_fig.savefig(os.path.join(path_to_fig_save, f"{output_prefix}topic_popularity_lda.png"))
        plt.close(lda_fig)