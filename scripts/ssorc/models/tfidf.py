import os
import gensim
import mysql.connector

from src.utils.corpora import TokenDocStream
from src.utils.LoopTimer import LoopTimer


token_type = 'originalText'

tfdf_model_name = "pruned_oT_potML.tfidf"
dic_file_name = "pruned_originalText_potML.dict"

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionary = os.path.join(path_to_db, "dictionaries",  dic_file_name)

dictionary = gensim.corpora.Dictionary.load(path_to_dictionary)

connection = mysql.connector.connect(host="localhost",
                                     user="root",
                                     passwd="thesis")

cursor = connection.cursor()
cursor.execute("USE ssorc;")
#sq1 = "SELECT abstract_id FROM abstracts WHERE annotated=1 and dictionaried=1"
sq1 = "SELECT abstract_id FROM mlabstracts"
cursor.execute(sq1)

lc = LoopTimer(update_after=1000)
abstracts = set()
for row in cursor:
    abstracts.add(row[0])
    lc.update("Collect Abstracts to Process")
connection.close()


class Corpus(object):
    def __init__(self):
        self.corpus = TokenDocStream(abstracts, token_type, token_cleaned=False, print_status=True, lower=True)

    def __iter__(self):
        for document in self.corpus:
            yield dictionary.doc2bow(document)


corpus = Corpus()
print()
tfidf = gensim.models.TfidfModel(corpus)
tfidf.save(os.path.join(path_to_db, "models", tfdf_model_name))
