import os
import _thread

import gensim
import mysql.connector
import pickle

from src.utils.corpora import nlp_to_doc_token
from src.utils.corpora import nlp_to_doc_tokenbigrams

from src.utils.LoopTimer import LoopTimer


def input_thread(a_list):
    input()
    a_list.append(True)


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations")
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id FROM abstracts WHERE annotated=1 and dictionaried=0"

print("Request Abstracts")
cursor.execute(sq1)

print("Collect Abstracts")
abstracts_to_process = cursor.fetchall()

print("Abstract to process: " + str(len(abstracts_to_process)))

dic_names = set(["word", "wordbigram", "pos", "posbigram", "lemma", "lemmabigram"])

dictionaries = dict()

for dictionary in dic_names:
    dic_path = os.path.join(path_to_dictionaries, "full_" + dictionary + ".dict")
    if os.path.isfile(dic_path):
        dictionaries[dictionary] = gensim.corpora.Dictionary.load(dic_path)
    else:
        dictionaries[dictionary] = gensim.corpora.Dictionary()

a_list = list()
_thread.start_new_thread(input_thread, (a_list,))
lt = LoopTimer(update_after=10, avg_length=500)
for row in abstracts_to_process:
    abstract_id = row[0]

    path_to_annotation_file = os.path.join(path_to_annotations, abstract_id + ".antn")

    if not os.path.isfile(path_to_annotation_file):
        print()
        print(abstract_id + " in db but missing file.")
        print()
        continue

    with open(path_to_annotation_file, "rb") as anno_file:
        annotation = pickle.load(anno_file)

    for dictionary in dic_names:
        if "bigram" in dictionary:
            token_type = dictionary[0:len(dictionary)-6]
            tokens = nlp_to_doc_tokenbigrams(annotation, token_type=token_type)
        else:
            tokens = nlp_to_doc_token(annotation, token_type=dictionary)

        dictionaries[dictionary].add_documents([tokens], prune_at=20000000)

    sq1 = f"UPDATE abstracts SET dictionaried=1 WHERE abstract_id='{abstract_id}';"
    cursor.execute(sq1)

    lt.update("Build Dictionaries")

    if a_list:
        break

connection.commit()

print()
for dictionary in dic_names:
    dic_path = os.path.join(path_to_dictionaries, "full_" + dictionary + ".dict")
    dictionaries[dictionary].save(dic_path)

    print(dictionary)
    print(dictionaries[dictionary])

connection.close()
