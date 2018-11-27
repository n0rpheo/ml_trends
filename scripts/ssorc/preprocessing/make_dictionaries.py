import os

import gensim
import mysql.connector
import pickle

from src.utils.corpora import nlp_to_doc_token
from src.utils.corpora import nlp_to_doc_tokenbigrams

from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations")
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")

# Possible Types:
# "word", "wordbigram", "pos", "posbigram", "lemma", "lemmabigram", "originalText"
dic_type = "originalText"
dic_filename = "full_originalText_potML.dict"
all_lower = True

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
# sq1 = "SELECT abstract_id FROM abstracts WHERE isML=1"
sq1 = "SELECT abstract_id FROM mlabstracts"
# sq1 = "SELECT abstract_id FROM abstracts WHERE annotated=1 and dictionaried=0"

print("Request Abstracts")
cursor.execute(sq1)

print("Collect Abstracts")
abstracts = set()
for row in cursor:
    abstracts.add(row[0])

print("Abstract to process: " + str(len(abstracts)))


dic_path = os.path.join(path_to_dictionaries, dic_filename)
if os.path.isfile(dic_path):
    dictionary = gensim.corpora.Dictionary.load(dic_path)
else:
    dictionary = gensim.corpora.Dictionary()

lt = LoopTimer(update_after=10, avg_length=500, target=len(abstracts))
for abstract_id in abstracts:
    path_to_annotation_file = os.path.join(path_to_annotations, abstract_id + ".antn")

    if not os.path.isfile(path_to_annotation_file):
        print()
        print(abstract_id + " in db but missing file.")
        print()
        continue

    with open(path_to_annotation_file, "rb") as anno_file:
        annotation = pickle.load(anno_file)

    if "bigram" in dic_type:
        token_type = dic_type[0:len(dic_type)-6]
        tokens = nlp_to_doc_tokenbigrams(annotation, token_type=token_type)
    else:
        tokens = nlp_to_doc_token(annotation, token_type=dic_type)

    if all_lower:
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

    dictionary.add_documents([tokens], prune_at=None)

    # sq1 = f"UPDATE abstracts SET dictionaried=1 WHERE abstract_id='{abstract_id}';"
    # cursor.execute(sq1)

    lt.update("Build Dictionaries")

print("Building has finished. Saving and Committing.")

connection.commit()
connection.close()

print()

dictionary.save(dic_path)
print(dictionary)


