import json
import pickle
import os
import operator

import matplotlib.pyplot as plt
import numpy as np
import math

import random

from src.utils.corpora import lemma_doc_stream
from src.modules.topic_modeling import TopicModeling
from src.utils.LoopTimer import LoopTimer

from src.utils.ksc import ksc

create_db = False
create_pop = False
dtype = 'ssorc'

pop_path = os.path.join('../../data/processed', dtype, 'popularities')
info_db_path = os.path.join(pop_path, 'info_db.p')
pop_db_path = os.path.join(pop_path, 'pop_db.p')
year_db_path = os.path.join(pop_path, 'year_db.p')

if create_db:
    info_db = dict()
    year_db = dict()

    lc = LoopTimer(update_after=1000)
    with open('../../data/processed/ssorc/json/s2-corpus-00.json') as corpus:
        for idx, line in enumerate(corpus):
            data = json.loads(line)

            year = int(data['year'])

            info_db[data['id']] = year

            if year not in year_db:
                year_db[year] = 0
            year_db[year] += 1

            lc.update("Build DB")
    pickle.dump(info_db, open(info_db_path, "wb"))
    pickle.dump(year_db, open(year_db_path, "wb"))
else:
    info_db = pickle.load(open(info_db_path, "rb"))
    year_db = pickle.load(open(year_db_path, "rb"))


ls = lemma_doc_stream('ssorc')
tm = TopicModeling('ssorc')
tm.load_model("lda_500_topics.model")

if create_pop:
    popularity = dict()
    lc = LoopTimer(update_after=10)
    for idx, (did, lemmas) in enumerate(ls):
        topic = tm.get_topic(lemmas)
        year = info_db[did]
        if (topic, year) not in popularity:
            popularity[(topic, year)] = 0

        popularity[(topic, year)] += 1

        lc.update("Pops")
        if idx % 10000 == 0:
            pickle.dump(popularity, open(pop_db_path, "wb"))

    pickle.dump(popularity, open(pop_db_path, "wb"))
else:
    popularity = pickle.load(open(pop_db_path, "rb"))


year_db = dict(sorted(year_db.items(), key=operator.itemgetter(0)))

x_series = list()
for topic in range(0, 499):
    pop = dict()
    for year in year_db:

        if year >= 1990:
            key = (topic, year)

            if key not in popularity:
                pop[year] = 0
            else:
                pop[year] = popularity[key] / year_db[year]

    x_serie = list()

    for year in range(1990, 2010):
        x_serie.append(pop[year])

    if sum(x_serie) > 0:
        x_serie = np.array(x_serie)
        x_series.append(x_serie)

    #plt.plot(list(pop.keys()), list(pop.values()), 'b-')


init_clusters = list()
init_clusters.append(list())
init_clusters.append(list())
init_clusters.append(list())

r_scale = 2000

#upwards trend
for i in range(0, 20):
    assign = 0.5*math.exp(i-19)+0.25 + random.randint(0, 100)/r_scale
    init_clusters[0].append(assign)

#downwards_trend
for i in range(0, 20):
    assign = 0.5*(-math.exp(i-19)+1)+0.25 + random.randint(0, 100)/r_scale
    init_clusters[1].append(assign)

#even_trend

for i in range(0, 20):
    assign = 0.5 + random.randint(0, 100)/r_scale
    init_clusters[2].append(assign)

kspec = ksc(max_iter=10000)
cluster, mu = kspec.train(x_series, init_clusters)

print()
print(cluster[0])
print(cluster[1])
print(cluster[2])


plt.plot(mu[0])
plt.plot(mu[1])
plt.plot(mu[2])
plt.axis([0, 20, -1, 1])
plt.ylabel('some numbers')
plt.show()

