import operator

import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
import math

import random

from src.utils.corpora import TokenDocStream
from src.modules.topic_modeling import TopicModeling
from src.utils.LoopTimer import LoopTimer

from src.utils.ksc import ksc

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

print("Fetching db entries")
cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id, year FROM abstracts WHERE isML=1"
cursor.execute(sq1)

info_db = dict()
year_db = dict()
abstracts = list()

for result in cursor:
    abstract_id = result[0]
    year = result[1]

    abstracts.append(abstract_id)
    info_db[abstract_id] = year
    if year not in year_db:
        year_db[year] = 0
    year_db[year] += 1

connection.close()

print('Initialize Model')
tm = TopicModeling(dictionary_name='full_lemma.dict',
                   tfidf_name='lemma_model.tfidf',
                   model_name='tm_lda_lemma.model')

num_topics = tm.lda_model.num_topics

corpus = TokenDocStream(abstracts=abstracts, token_type='lemma', print_status=False, output='all')

popularity = dict()
lc = LoopTimer(update_after=50)
for abstract_id, tokens in corpus:
    topic = tm.get_topic(tokens)
    year = info_db[abstract_id]
    if (topic, year) not in popularity:
        popularity[(topic, year)] = 0

    popularity[(topic, year)] += 1

    lc.update("Pops")

year_db = dict(sorted(year_db.items(), key=operator.itemgetter(0)))

x_series = list()
for topic in range(0, num_topics):
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
        if year in pop:
            x_serie.append(pop[year])
        elif len(x_serie) > 0:
            x_serie.append(x_serie[-1])
        else:
            x_serie.append(0)

    if sum(x_serie) > 0:
        x_serie = np.array(x_serie)
        x_series.append(x_serie)

    #plt.plot(list(pop.keys()), list(pop.values()), 'b-')


init_clusters = list()
init_clusters.append(list())
init_clusters.append(list())
init_clusters.append(list())

r_scale = 2000

t_from = 1990
t_to = 2010

#upwards trend
for i in range(t_from, t_to):
    assign = 0.5*math.exp((i-(t_to-1))/(t_to - t_from))+0.25 + random.randint(0, 100)/r_scale
    init_clusters[0].append(assign)

#downwards_trend
for i in range(t_from, t_to):
    assign = 0.5*(-math.exp((i-(t_to-1))/(t_to - t_from))+1)+0.25 + random.randint(0, 100)/r_scale
    init_clusters[1].append(assign)

#even_trend
for i in range(t_from, t_to):
    assign = 0.5 + random.randint(0, 100)/r_scale
    init_clusters[2].append(assign)

kspec = ksc(max_iter=100)
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

