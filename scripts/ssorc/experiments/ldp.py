import os
import pickle
import numpy as np
import operator

path_to_db = "/media/norpheo/mySQL/db/ssorc"
popularity_info_file = os.path.join(path_to_db, "visualization", "pop_info_file.pickle")

with open(popularity_info_file, "rb") as pif:
    pop_infos = pickle.load(pif)
    num_topics = pop_infos["num_topics"]
    popularity = pop_infos["popularitiy"]
    segments_per_year = pop_infos["segments_per_year"]
    rf_labels = pop_infos["rf_labels"]

segments_per_year = dict(sorted(segments_per_year.items(), key=operator.itemgetter(0)))
rf_labels = list(rf_labels)
year_from = 1990
year_to = 2020


for topic in range(0, num_topics):
    before_data_set = None
    ld_matrix = list()
    for segment_label in rf_labels:
        pop = list()
        for year in segments_per_year:
            if year_from <= year <= year_to:
                key = (topic, segment_label, year)

                if key not in popularity:
                    pop.append(0)
                else:
                    pop.append(popularity[key] / segments_per_year[year])

        ld_matrix.append(pop)
    ld_matrix = np.array(ld_matrix)

    sum_years = np.sum(ld_matrix, axis=1)
    sum_all = np.sum(sum_years)
    rf_percentage = sum_years / sum_all
    for idx in range(len(rf_labels)):
        print(f"{rf_labels[idx]}: {rf_percentage[idx]}")
    print("-----")

