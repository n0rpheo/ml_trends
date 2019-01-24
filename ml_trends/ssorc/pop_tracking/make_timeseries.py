import os
import pickle
import pandas as pd

from src.modules.topic_modeling import TopicModelingGLDA
from src.modules.abstract_parser import AbstractParser
from src.utils.LoopTimer import LoopTimer

from src.utils.selector import select_path_from_dir

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, 'dictionaries')
path_to_models = os.path.join(path_to_db, 'models')
otdb_path = os.path.join(path_to_db, 'pandas', 'ml_ot.pandas')
posdb_path = os.path.join(path_to_db, 'pandas', 'ml_pos.pandas')
yeardb_path = os.path.join(path_to_db, 'pandas', 'ml_year.pandas')
popularity_data_frame_dist = os.path.join(path_to_db, "popularities", "df_500topics_dist_hl.pickle")  # output
popularity_data_frame_top = os.path.join(path_to_db, "popularities", "df_500topics_top_hl.pickle")  # output


print("Loading Panda DB")
otDF = pd.read_pickle(otdb_path)
posDF = pd.read_pickle(posdb_path)
yearDF = pd.read_pickle(yeardb_path)
print("Done Loading")

dic_path = select_path_from_dir(path_to_dictionaries,
                                phrase="Select Dictionary: ",
                                suffix=".dict",
                                preselection="pruned_ot_ml.dict")
model_path = select_path_from_dir(path_to_models,
                                  phrase="Select Topic Model: ",
                                  suffix=".pickle",
                                  preselection="tm_glda_model_500topics.pickle")

print('Initialize Model')
tm_model = TopicModelingGLDA(dic_path=dic_path,
                             model_path=model_path)
ap_model = AbstractParser(model_name='svm_rf_lcpupbwuwb_hl.pickle',
                          word_dic=os.path.join(path_to_dictionaries, "pruned_word_ml.dict"),
                          wordbigram_dic=os.path.join(path_to_dictionaries, "pruned_wordbigramm_ml.dict"),
                          pos_dic=os.path.join(path_to_dictionaries, "full_pos_ml.dict"),
                          posbigram_dic=os.path.join(path_to_dictionaries, "full_posbigramm_ml.dict"),
                          word_tfidf=os.path.join(path_to_models, "pruned_word_ml.tfidf"),
                          wordbigram_tfidf=os.path.join(path_to_models, "pruned_wordbigramm_ml.tfidf"),
                          pos_tfidf=os.path.join(path_to_models, "full_pos_ml.tfidf"),
                          posbigram_tfidf=os.path.join(path_to_models, "full_posbigramm_ml.tfidf"))

num_topics = tm_model.num_topics()
popularity_top = dict()
popularity_dist = dict()
segments_per_year = dict()
rf_labels = set()
df = otDF.join(posDF).join(yearDF)

lc = LoopTimer(update_after=50, avg_length=2000, target=len(df))
for abstract_id, row in df.iterrows():
    year = row['year']
    ot_tokens = row['originalText'].split()
    ot_sentence_tokens = [sentence.split(" ") for sentence in row['originalText'].split("\t")]

    pos_tokens = row['pos'].split()
    pos_sentence_tokens = [sentence.split(" ") for sentence in row['pos'].split("\t")]

    results = ap_model.predict(ot_sentence_tokens, pos_sentence_tokens)

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
        topic_dist = tm_model.get_topic_dist(ot_tokens)
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
    lc.update("Pops")

dates = pd.date_range('1990', '2020', freq='AS')

columns = list(rf_labels)
columns.append("spy")
columns.append("sum")

dataFrames_dist = list()
dataFrames_top = list()
for topic in range(num_topics):
    df_dist = pd.DataFrame(0, index=dates, columns=columns)
    df_top = pd.DataFrame(0, index=dates, columns=columns)
    for date in dates:
        year = date.year
        if year in segments_per_year:
            df_dist['spy'][date] = segments_per_year[year]
            df_top['spy'][date] = segments_per_year[year]
        for rf_label in rf_labels:
            key = (topic, rf_label, year)
            if key in popularity_dist:
                df_dist[rf_label][date] += popularity_dist[key]
                df_dist['sum'][date] += popularity_dist[key]
            if key in popularity_top:
                df_top[rf_label][date] += popularity_top[key]
                df_top['sum'][date] += popularity_top[key]
    dataFrames_dist.append(df_dist)
    dataFrames_top.append(df_top)

with open(popularity_data_frame_dist, 'wb') as pop_file_dist:
    pickle.dump(dataFrames_dist, pop_file_dist)

with open(popularity_data_frame_top, 'wb') as pop_file_top:
    pickle.dump(dataFrames_top, pop_file_top)




