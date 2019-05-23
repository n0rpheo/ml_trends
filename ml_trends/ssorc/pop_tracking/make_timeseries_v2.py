import os
import pickle
import pandas as pd
import numpy as np

from spacy.vocab import Vocab
from spacy.tokens import Doc

from MLGenome.MLARes.occ.occ import occTopMention

from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)
path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
path_to_popularities = os.path.join(path_to_db, "popularities", nlp_model)
path_to_rfl = os.path.join(path_to_db, "rhet_func_labeling", nlp_model)

"""
    Information Input
"""

occ_file = "occ_vec_ratio_p1.pickle"
model_file_name = f"svm_lin_feature_info.pickle.pickle"
pop_file_name = "pop_timeseries_lg_occ1.pickle"  # output Filename

year_start = 1980
year_end = 2020

rf_labels = ["Method", "Background", "Objective"]

"""
    ###########
"""

print(pop_file_name)

print("Loading Vocab...")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

"""
    Loading OCC Information
"""
print("Loading Overlapping Correlation Clustering...")
otm = occTopMention(path=os.path.join(path_to_mlgenome, occ_file))

cluster_dict = dict()
for cluster_id in range(otm.clusters.shape[1]):
    cluster_words = set([otm.mentions[mention_id]["string"] for mention_id in np.where(otm.clusters[:, cluster_id] == 1)[0]])
    if len(cluster_words) > 0:
        c_rep = otm.cid2rep[cluster_id]
        if c_rep not in cluster_dict:
            cluster_dict[c_rep] = set()
        cluster_dict[c_rep] = cluster_dict[c_rep].union(cluster_words)

"""
    Loading Rhetorical Function Classifier
"""
with open(os.path.join(path_to_rfl, model_file_name), "rb") as handle:
    rf_info = pickle.load(handle)
rf_clf = rf_info["model"]

"""
    Prepare popularity dict
"""

popularity_dict = dict()
for year in range(year_start, year_end):
    popularity_dict[year] = dict()
    popularity_dict[year]['sum'] = 0  # Total Count for all Mentions in all RF

    for rf_label in rf_labels:
        popularity_dict[year][rf_label] = dict()
        popularity_dict[year][rf_label]["sum"] = 0  # Total Count for all Mentions in one RF
        for cluster in cluster_dict:
            popularity_dict[year][rf_label][cluster] = 0  # Total Count for all Mentions of a specific Cluster in one RF
    for cluster in cluster_dict:
        popularity_dict[year][cluster] = 0  # Total Count of all Mentions of a specific Cluster


lc = LoopTimer(update_after=50, avg_length=2000, target=len(infoDF))

"""
    Iterate over the span of the years
"""

for year in range(year_start, year_end):
    yearDF = infoDF[infoDF['year'] == year]

    """
        Iterate over all abstracts in a year
    """

    for idx, (abstract_id, df_row) in enumerate(yearDF.iterrows()):
        file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
        doc = Doc(vocab).from_disk(file_path)

        n_sents = sum([1 for sent in doc.sents])

        if n_sents == 0:
            continue

        """
            Iterate over all sentences in an abstract
        """

        for sid, sentence in enumerate(doc.sents):
            word_vec = sentence.vector
            location = sid / n_sents

            # Features for the rhetorical Function labeling
            features = np.append(word_vec, [location])

            sent_label = rf_clf.predict([features])[0]
            # Get Senctenc String to Count Cluster-Words
            sent_string = sentence.text

            for topic_cluster in cluster_dict:
                cluster_words = cluster_dict[topic_cluster]

                word_count = 0

                for cluster_word in cluster_words:
                    word_count += sent_string.count(cluster_word)


                popularity_dict[year][sent_label][topic_cluster] += word_count
                popularity_dict[year][sent_label]["sum"] += word_count
                popularity_dict[year][topic_cluster] += word_count
                popularity_dict[year]['sum'] += word_count

                if word_count > 0 and topic_cluster == "naive bayes":
                    print(word_count, sent_label, year, popularity_dict[year][sent_label][topic_cluster], popularity_dict[year]['sum'])
        #breaker = lc.update(f"Make Timeseries | Year {year}")
        if idx > 50:
            break

print()
dates = pd.date_range(f"{year_start}", f"{year_end}", freq='AS')

columns = ["sum-cluster", "sum-all"]
for rf_label in rf_labels:
    columns.append(f"n-{rf_label}")
    columns.append(f"sum-{rf_label}")

dataFrames = dict()
for cluster in cluster_dict:
    df = pd.DataFrame(0.0, index=dates, columns=columns)
    for date in dates:
        year = date.year
        if year in popularity_dict:
            for rf_label in rf_labels:
                if rf_label in popularity_dict[year] and cluster in popularity_dict[year][rf_label]:
                    df[f"n-{rf_label}"][date] = popularity_dict[year][rf_label][cluster]
                    df[f"sum-{rf_label}"][date] = popularity_dict[year][rf_label]["sum"]
            df["sum-cluster"][date] = popularity_dict[year][cluster]
            df["sum-all"][date] = popularity_dict[year]["sum"]

    if cluster in dataFrames:
        dataFrames[cluster] += df
    else:
        dataFrames[cluster] = df

if not os.path.isdir(path_to_popularities):
    print(f"Create Directory {path_to_popularities}")
    os.mkdir(path_to_popularities)

with open(os.path.join(path_to_popularities, pop_file_name), 'wb') as handle:
    pickle.dump(dataFrames, handle)
