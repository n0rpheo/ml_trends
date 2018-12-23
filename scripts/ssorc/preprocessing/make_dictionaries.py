import os

import gensim
import mysql.connector

from src.utils.corpora import TokenDocStream
import src.utils.selector as selector


path_to_db = "/media/norpheo/mySQL/db/ssorc"

token_types = list()
dic_paths = list()

while True:
    # Possible Types:
    valid_types = ["word", "wordbigram", "pos", "posbigram", "lemma", "lemmabigram", "originalText", "originalTextbigram"]
    token_type = selector.select_item_from_list(list(set(valid_types)-set(token_types)), "Select Dictionary Type: ")
    dic_filename = input("Dictionary File Name: ")
    dic_path = os.path.join(path_to_db, "dictionaries", dic_filename)

    token_types.append(token_type)
    dic_paths.append(dic_path)

    if input("Add more Dictionaries? (y/n) ") != 'y':
        break

lower = True

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
# sq1 = "SELECT abstract_id FROM abstracts WHERE isML=1"
sq1 = "SELECT abstract_id FROM abstracts_ml WHERE entities LIKE '%machine learning%' AND annotated=1"
# sq1 = "SELECT abstract_id FROM abstracts WHERE annotated=1 and dictionaried=0"
cursor.execute(sq1)

print("Collect Abstracts")
abstracts = set()
for row in cursor:
    abstracts.add(row[0])
connection.close()

for i in range(len(token_types)):
    token_type = token_types[i]
    dic_path = dic_paths[i]

    corpus = TokenDocStream(abstracts=abstracts,
                            token_type=token_type,
                            print_status=True,
                            token_cleaned=False,
                            output=None,
                            lower=lower)

    dictionary = gensim.corpora.Dictionary()

    for tokens in corpus:
        dictionary.add_documents([tokens], prune_at=None)

    print()

    dictionary.save(dic_path)
    print(dictionary)


