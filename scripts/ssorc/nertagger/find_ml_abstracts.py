import os
import pickle
import threading

import mysql.connector

from nltk.tag import StanfordNERTagger
from src.utils.corpora import TokenDocStream
from src.utils.LoopTimer import LoopTimer
from src.utils.functions import lookahead


def worker(abstract_id_, abstract_):
    st = StanfordNERTagger(path_to_ner_model,
                           '/media/norpheo/mySQL/stanford-ner-2018-02-27/stanford-ner.jar',
                           encoding='utf-8')
    classified_text = st.tag(abstract_)

    algo_name = None
    for item in classified_text:
        if item[1] == "MLALGO":
            if algo_name is None:
                algo_name = list()
            algo_name.append(item[0])
        elif algo_name is not None:
            algo_name = " ".join(algo_name)

            ml_algo_abstract.add(abstract_id_)
            if algo_name not in ml_algos:
                ml_algos[algo_name] = 0
            ml_algos[algo_name] += 1
            algo_name = None


NUM_WORKERS = 12
limit_abstracts = 20000

path_to_ner = "/media/norpheo/mySQL/db/ssorc/NER"

# Model 1 Files
path_to_ner_model1 = '/media/norpheo/mySQL/stanford-ner-2018-02-27/classifiers/ner-mlalgo-model.ser.gz'
path_to_ml_algos_save1 = os.path.join(path_to_ner, "ml_algos.dict")
path_to_ml_algo_abstract_save1 = os.path.join(path_to_ner, "ml_algo_abstract.pickle")
path_to_skip_list1 = os.path.join(path_to_ner, "skip_list.pickle")

# Model 2 Files - USE THIS
path_to_ner_model2 = os.path.join(path_to_ner, 'new_ner-model.ser.gz')
path_to_ml_algos_save2 = os.path.join(path_to_ner, "ml_algos_new.dict")
path_to_ml_algo_abstract_save2 = os.path.join(path_to_ner, "ml_algo_abstract_new.pickle")
path_to_skip_list2 = os.path.join(path_to_ner, "skip_list_new.pickle")

# Model Test Files
path_to_ner_model_t = os.path.join(path_to_ner, 'new_ner-model.ser.gz')
path_to_ml_algos_save_t = os.path.join(path_to_ner, "ml_algos_test.dict")
path_to_ml_algo_abstract_save_t = os.path.join(path_to_ner, "ml_algo_abstract_test.pickle")
path_to_skip_list_t = os.path.join(path_to_ner, "skip_list_test.pickle")

# Assign Files
path_to_ner_model = path_to_ner_model2
path_to_ml_algos_save = path_to_ml_algos_save2
path_to_ml_algo_abstract_save = path_to_ml_algo_abstract_save2
path_to_skip_list = path_to_skip_list2

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

print("Collecting Abstracts")
abstracts = set()
for row in cursor:
    abstracts.add(row[0])
connection.close()

abstracts = abstracts.difference(skip_list)
abstracts = set(list(abstracts)[:limit_abstracts])

docstream = TokenDocStream(token_type="originalText",
                           abstracts=abstracts,
                           token_cleaned=False,
                           print_status=False,
                           output='all')

lc = LoopTimer(update_after=NUM_WORKERS, avg_length=NUM_WORKERS*10, target=len(abstracts))
doc_bag = list()
starting_ml_algos = len(ml_algos)
starting_ml_abstracts = len(ml_algo_abstract)
for doc, has_next in lookahead(docstream):

    doc_bag.append(doc)
    if len(doc_bag) == NUM_WORKERS or not has_next:
        threads = list()

        for document in doc_bag:
            abstract_id = document[0]
            abstract = document[1]
            t = threading.Thread(target=worker, args=(abstract_id, abstract,))
            threads.append(t)
            t.start()
            skip_list.add(abstract_id)

        for thread in threads:
            thread.join()

        doc_bag = list()
        with open(path_to_ml_algos_save, "wb") as algo_save_file:
            pickle.dump(ml_algos, algo_save_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path_to_ml_algo_abstract_save, "wb") as algo_save_file:
            pickle.dump(ml_algo_abstract, algo_save_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path_to_skip_list, "wb") as skip_list_file:
            pickle.dump(skip_list, skip_list_file, protocol=pickle.HIGHEST_PROTOCOL)
    num_ml_abstracts = len(ml_algo_abstract)
    diff_ml_abstracts = num_ml_abstracts - starting_ml_abstracts
    lc.update(f"Find ML - Abstracts ({diff_ml_abstracts})")

print()
print()
num_ml_algos = len(ml_algos)
num_ml_abstracts = len(ml_algo_abstract)
diff_ml_algos = num_ml_algos - starting_ml_algos
diff_ml_abstracts = num_ml_abstracts - starting_ml_abstracts

for algo in ml_algos:
    print(f"{algo}: {ml_algos[algo]}")
print()

print(f"{diff_ml_algos} new Machine Learning Algorithms found. Total: {num_ml_algos}")
print(f"{diff_ml_abstracts} new Abstracts with potential ML Content found. Total {num_ml_abstracts}")

