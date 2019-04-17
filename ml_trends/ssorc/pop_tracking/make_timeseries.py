import os
import pickle
import pandas as pd

from src.modules.topic_modeling import TopicModelingGLDA
from src.modules.topic_modeling import TopicModelingLDA
from src.modules.abstract_parser import AbstractParser
from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, 'dictionaries')
path_to_models = os.path.join(path_to_db, 'models')
worddb_path = os.path.join(path_to_db, 'pandas', 'ml_word.pandas')
lemmadb_path = os.path.join(path_to_db, 'pandas', 'ml_lemma.pandas')
otdb_path = os.path.join(path_to_db, 'pandas', 'ml_ot.pandas')
posdb_path = os.path.join(path_to_db, 'pandas', 'ml_pos.pandas')
yeardb_path = os.path.join(path_to_db, 'pandas', 'ml_year.pandas')
popularity_data_frame_dist = os.path.join(path_to_db, "popularities",
                                          "df_500topics_svmlin_dist_pruned_features_tm_lemma_pruned.pickle")  # output
popularity_data_frame_top = os.path.join(path_to_db, "popularities",
                                         "df_500topics_svmlin_top_pruned_features_tm_lemma_pruned.pickle")  # output

print(popularity_data_frame_dist)
print("Loading Panda DB")
wordDF = pd.read_pickle(worddb_path)
lemmaDF = pd.read_pickle(lemmadb_path)
otDF = pd.read_pickle(otdb_path)
posDF = pd.read_pickle(posdb_path)
yearDF = pd.read_pickle(yeardb_path)
print("Done Loading")

print('Initialize Model')

tm_model = TopicModelingLDA(dic_path=os.path.join(path_to_dictionaries, "pruned_lemma_lower_pd.dict"),
                            model_path=os.path.join(path_to_models, "tm_lda_500topics.pickle"))
ap_model = AbstractParser(model_name='svm_lin_rf_pruned_features')

num_topics = tm_model.num_topics()

popularity_top = dict()
popularity_dist = dict()
segments_per_year = dict()
segments_per_label_per_year = dict()
rf_labels = set()
df = wordDF.join(otDF).join(posDF).join(yearDF).join(lemmaDF)

lc = LoopTimer(update_after=50, avg_length=2000, target=len(df))
for count, (abstract_id, row) in enumerate(df.iterrows()):
    lc.update("Pops")

    year = row['year']
    ot_tokens = row['originalText'].split()
    ot_sentence_tokens = [sentence.split(" ") for sentence in row['originalText'].split("\t")]

    num_sentences = len(ot_sentence_tokens)

    # Filter for abstract length 10 to 50 sentences
    if not (5 < num_sentences < 50):
        continue

    word_tokens = row['word'].split()
    word_sentence_tokens = [sentence.split(" ") for sentence in row['word'].split("\t")]

    lemma_tokens = row['lemma'].split()
    lemma_sentence_tokens = [sentence.split(" ") for sentence in row['lemma'].split("\t")]

    pos_tokens = row['pos'].split()
    pos_sentence_tokens = [sentence.split(" ") for sentence in row['pos'].split("\t")]

    results = ap_model.predict(word_sentence_tokens, pos_sentence_tokens)

    if results is None:
        continue

    segments = dict()
    for idx, rhetfunc_label in enumerate(results):
        rf_labels.add(rhetfunc_label)
        if rhetfunc_label not in segments:
            segments[rhetfunc_label] = list()
        segments[rhetfunc_label].append(idx)

    num_segments = len(segments)

    if year not in segments_per_year:
        segments_per_year[year] = 0
    segments_per_year[year] += num_segments

    for segment_label in segments:
        # Calculate how many segments occur for a specific label in a certain year
        skey = (segment_label, year)
        if skey not in segments_per_label_per_year:
            segments_per_label_per_year[skey] = 0
        segments_per_label_per_year[skey] += 1

        segment_sentence_tokens = [lemma_sentence_tokens[i] for i in segments[segment_label]]
        segment_tokens = [token for sentence in segment_sentence_tokens for token in sentence]
        # segment_tokens = sum(segment_sentence_tokens, []) # maybe slower
        topic_dist = tm_model.get_topic_dist(segment_tokens)
        top_topic = topic_dist.argmax()

        # Top Topic Popularity
        pop_key = (top_topic, segment_label, year)
        if pop_key not in popularity_top:
            popularity_top[pop_key] = 0
        popularity_top[pop_key] += 1

        # Topic Distribution Popularity
        for topic_id in range(len(topic_dist)):
            pop_key = (topic_id, segment_label, year)
            if pop_key not in popularity_dist:
                popularity_dist[pop_key] = 0
            popularity_dist[pop_key] += topic_dist[topic_id]


dates = pd.date_range('1950', '2020', freq='AS')

columns = list(rf_labels)
for label in rf_labels:
    columns.append(f"sp-{label}-py")
columns.append("spy")
columns.append("sum")

dataFrames_dist = list()
dataFrames_top = list()
for topic in range(num_topics):
    df_dist = pd.DataFrame(0.0, index=dates, columns=columns)
    df_top = pd.DataFrame(0.0, index=dates, columns=columns)
    for date in dates:
        year = date.year
        if year in segments_per_year:
            df_dist['spy'][date] = segments_per_year[year]
            df_top['spy'][date] = segments_per_year[year]

            for rf_label in rf_labels:
                skey = (rf_label, year)
                key = (topic, rf_label, year)

                if skey in segments_per_label_per_year:
                    df_dist[f"sp-{rf_label}-py"][date] = segments_per_label_per_year[skey]
                    df_top[f"sp-{rf_label}-py"][date] = segments_per_label_per_year[skey]

                if key in popularity_dist:
                    df_dist[rf_label][date] = popularity_dist[key] / segments_per_year[year]
                    df_dist['sum'][date] += df_dist[rf_label][date]
                if key in popularity_top:
                    df_top[rf_label][date] = popularity_top[key] / segments_per_label_per_year[skey]
                    df_top['sum'][date] += popularity_top[key] / segments_per_label_per_year[skey]
    dataFrames_dist.append(df_dist)
    dataFrames_top.append(df_top)

print()

for date in dates:
    year_sum = 0.0
    for topic in range(num_topics):
        year_sum += dataFrames_dist[topic]['sum'][date]

    print(f"{date}: {year_sum}")

exit()

with open(popularity_data_frame_dist, 'wb') as pop_file_dist:
    pickle.dump(dataFrames_dist, pop_file_dist)

with open(popularity_data_frame_top, 'wb') as pop_file_top:
    pickle.dump(dataFrames_top, pop_file_top)




