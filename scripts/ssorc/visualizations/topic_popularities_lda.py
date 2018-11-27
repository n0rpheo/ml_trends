import operator
import mysql.connector
import matplotlib.pyplot as plt

from src.utils.corpora import TokenDocStream
from src.modules.topic_modeling import TopicModelingGLDA, TopicModelingLDA
from src.utils.LoopTimer import LoopTimer

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
# tm = TopicModelingLDA(dictionary_name='full_lemma.dict',
#                       tfidf_name='lemma_model.tfidf',
#                       model_name='tm_lda_lemma.model')

tm = TopicModelingGLDA(dictionary_name='pruned_originalText_isML.dict',
                       model_name='tm_glda_model.pickle')

num_topics = tm.num_topics()

corpus = TokenDocStream(abstracts=abstracts, token_type='originalText', print_status=False, output='all', lower=True)

popularity = dict()
lc = LoopTimer(update_after=50, avg_length=100, target=len(abstracts))
for abstract_id, tokens in corpus:
    topic = tm.get_topic(tokens)
    year = info_db[abstract_id]
    if (topic, year) not in popularity:
        popularity[(topic, year)] = 0

    popularity[(topic, year)] += 1

    lc.update("Pops")

year_db = dict(sorted(year_db.items(), key=operator.itemgetter(0)))

year_from = 2000
year_to = 2017

x_series = list()
topic = 2
pop = dict()

for topic in range(0, num_topics):
    peak = 0
    for year in year_db:

        if year_from <= year <= year_to:
            key = (topic, year)

            if key not in popularity:
                pop[year] = 0
            else:
                popu = popularity[key] / year_db[year]
                if popu > peak:
                    peak = popu
                pop[year] = popu
                #print(f"{year} - {popu} - {year_db[year]}")

    x_serie = list()

    plt.plot(list(pop.keys()), list(pop.values()), 'b-')


    plt.axis([year_from, year_to, 0, peak*1.2])
    plt.ylabel(f'Topic-No {topic}')
    plt.show()

print()
print(f"Topics: {num_topics}")

