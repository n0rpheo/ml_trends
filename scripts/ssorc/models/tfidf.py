import os
import pickle
import gensim
import mysql.connector

from src.utils.corpora import nlp_to_doc_token
from src.utils.corpora import nlp_to_doc_tokenbigrams
from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")
path_to_annotations = os.path.join(path_to_db, 'annotations')
path_to_models = os.path.join(path_to_db, 'models')

dic_names = set(["word", "wordbigram", "pos", "posbigram", "lemma", "lemmabigram"])
dictionaries = dict()
token_type = 'word'
dic_path = os.path.join(path_to_dictionaries, "full_" + token_type + ".dict")
dictionaries[token_type] = gensim.corpora.Dictionary.load(dic_path)


class Corpus(object):
    def __init__(self, token_type):
        self.token_type = token_type

        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

        cursor = connection.cursor()
        cursor.execute("USE ssorc;")
        cursor.execute("SELECT abstract_id FROM abstracts WHERE annotated=1 and dictionaried=1")

        lc = LoopTimer(update_after=1000)
        self.abstracts = set()
        for idx, row in enumerate(cursor):
            self.abstracts.add(row[0])
            lc.update("Collect Abstracts to Process")
        connection.close()

    def __iter__(self):
        lc = LoopTimer(update_after=1, avg_length=1000)
        for abstract_id in self.abstracts:
            path_to_annotation_file = os.path.join(path_to_annotations, abstract_id + ".antn")
            if not os.path.isfile(path_to_annotation_file):
                print()
                print(abstract_id + " in db but missing file.")
                print()
                continue

            with open(path_to_annotation_file, "rb") as annotation_file:
                annotation = pickle.load(annotation_file)
            document = nlp_to_doc_token(annotation, self.token_type)
            lc.update("Yield Abstract")
            yield dictionaries[token_type].doc2bow(document)


corpus = Corpus(token_type)
print()
tfidf = gensim.models.TfidfModel(corpus)
tfidf.save(os.path.join(path_to_models, token_type + '_model.tfidf'))
