import os
import pickle

import mysql.connector

from nltk.tag import StanfordNERTagger
from src.utils.corpora import TokenDocStream
from src.utils.LoopTimer import LoopTimer

st = StanfordNERTagger('/media/norpheo/mySQL/stanford-ner-2018-02-27/classifiers/ner-mlalgo-model.ser.gz',
                       '/media/norpheo/mySQL/stanford-ner-2018-02-27/stanford-ner.jar',
                       encoding='utf-8')

path_to_ner = "/media/norpheo/mySQL/db/ssorc/NER"
path_to_ml_algos_save = os.path.join(path_to_ner, "ml_algos.dict")
path_to_ml_algo_abstract_save = os.path.join(path_to_ner, "ml_algo_abstract.pickle")
path_to_skip_list = os.path.join(path_to_ner, "skip_list.pickle")

if os.path.isfile(path_to_skip_list):
    with open(path_to_skip_list, "rb") as skip_list_file:
        skip_list = pickle.load(skip_list_file)
else:
    skip_list = set()

if os.path.isfile(path_to_ml_algo_abstract_save):
    with open(path_to_ml_algo_abstract_save, "rb") as algo_abstract_file:
        ml_algo_abstract = pickle.load(algo_abstract_file)
else:
    ml_algo_abstract = set()

if os.path.isfile(path_to_ml_algos_save):
    with open(path_to_ml_algos_save, "rb") as ml_algos_file:
        ml_algos = pickle.load(ml_algos_file)
else:
    ml_algos = dict()

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id FROM abstracts WHERE annotated=1 and dictionaried=1"
cursor.execute(sq1)

abstracts = set()
lc = LoopTimer(update_after=10000, avg_length=200000)
for row in cursor:
    abstracts.add(row[0])
    lc.update("Collecting")
connection.close()

abstracts = abstracts.difference(skip_list)
abstracts = set(list(abstracts)[:2000])

docstream = TokenDocStream(token_type="originalText",
                           abstracts=abstracts,
                           token_cleaned=False,
                           print_status=True,
                           output='all')
print()
for doc in docstream:
    abstract_id = doc[0]
    abstract = doc[1]
    classified_text = st.tag(abstract)

    algo_name = None
    for item in classified_text:
        if item[1] == "MLALGO":
            if algo_name is None:
                algo_name = list()
            algo_name.append(item[0])
        elif algo_name is not None:
            algo_name = " ".join(algo_name)

            ml_algo_abstract.add(abstract_id)
            if algo_name not in ml_algos:
                ml_algos[algo_name] = 0
            ml_algos[algo_name] += 1

            with open(path_to_ml_algos_save, "wb") as algo_save_file:
                pickle.dump(ml_algos, algo_save_file, protocol=pickle.HIGHEST_PROTOCOL)

            with open(path_to_ml_algo_abstract_save, "wb") as algo_save_file:
                pickle.dump(ml_algo_abstract, algo_save_file, protocol=pickle.HIGHEST_PROTOCOL)

            algo_name = None

    skip_list.add(abstract_id)
    with open(path_to_skip_list, "wb") as skip_list_file:
        pickle.dump(skip_list, skip_list_file, protocol=pickle.HIGHEST_PROTOCOL)

print()
print()
for algo in ml_algos:
    print(f"{algo}: {ml_algos[algo]}")
print()
print(f"{len(ml_algos)} Machine Learning Algorithms found.")
print(f"{len(ml_algo_abstract)} Number of Abstracts with potential ML Content")

