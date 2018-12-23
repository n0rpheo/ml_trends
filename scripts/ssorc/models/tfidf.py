import os
import gensim
import mysql.connector

from src.utils.corpora import TokenDocStream
from src.utils.LoopTimer import LoopTimer
import src.utils.selector as selector

token_types = list()
dic_paths = list()
model_paths = list()
path_to_db = "/media/norpheo/mySQL/db/ssorc"
valid_types = ["word", "wordbigram", "pos", "posbigram", "lemma", "lemmabigram", "originalText", "originalTextbigram"]

while True:
    token_type = selector.select_item_from_list(list(set(valid_types) - set(token_types)), "Select Token Type: ")
    tfidf_model_name = input("Enter tfidf Modelname: ")
    dic_path = selector.select_path_from_dir(os.path.join(path_to_db, "dictionaries"), "Select Dictionary: ")
    model_path = os.path.join(path_to_db, "models", tfidf_model_name)

    token_types.append(token_type)
    dic_paths.append(dic_path)
    model_paths.append(model_path)

    if input("Add more Models? (y/n) ") != 'y':
        break


connection = mysql.connector.connect(host="localhost",
                                     user="root",
                                     passwd="thesis")

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id FROM abstracts_ml WHERE entities LIKE '%machine learning%' AND annotated=1"
cursor.execute(sq1)

lc = LoopTimer(update_after=1000)
abstracts = set()
for row in cursor:
    abstracts.add(row[0])
    lc.update("Collect Abstracts to Process")
connection.close()


class Corpus(object):
    def __init__(self, token_type_, dictionary_):
        self.dictionary = dictionary_
        self.corpus = TokenDocStream(abstracts,
                                     token_type=token_type_,
                                     token_cleaned=False,
                                     print_status=True,
                                     lower=True)

    def __iter__(self):
        for document in self.corpus:
            yield self.dictionary.doc2bow(document)


for i in range(len(token_types)):
    dictionary = gensim.corpora.Dictionary.load(dic_paths[i])

    corpus = Corpus(token_type_=token_types[i], dictionary_=dictionary)
    print()
    tfidf = gensim.models.TfidfModel(corpus)
    tfidf.save(model_paths[i])
