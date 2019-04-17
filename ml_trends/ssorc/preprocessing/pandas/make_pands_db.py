import os
import mysql.connector
import pandas as pd

from src.utils.corpora import TokenDocStream
from src.utils.LoopTimer import LoopTimer

db_path = "/media/norpheo/mySQL/db/ssorc/pandas"

connection = mysql.connector.connect(host="localhost",
                                     user="root",
                                     passwd="thesis")

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id, year FROM abstracts_ml WHERE entities LIKE '%artificial intelligence%' AND annotated=1"
cursor.execute(sq1)

print("Collect Abstracts from DB")
abstracts = list()
year_db = dict()
for row in cursor:
    abs_id = row[0]
    abstracts.append(abs_id)
    year_db[abs_id] = row[1]
connection.close()


ot_corpus = TokenDocStream(abstracts=abstracts,
                           token_type='originalText',
                           token_cleaned=0,
                           output='all',
                           lower=False,
                           split_sentences=True)
pos_corpus = TokenDocStream(abstracts=abstracts,
                            token_type='pos',
                            token_cleaned=0,
                            output='all',
                            lower=True,
                            split_sentences=True)
word_corpus = TokenDocStream(abstracts=abstracts,
                             token_type='word',
                             token_cleaned=0,
                             output='all',
                             lower=False,
                             split_sentences=True)
lemma_corpus = TokenDocStream(abstracts=abstracts,
                              token_type='lemma',
                              token_cleaned=0,
                              output='all',
                              lower=False,
                              split_sentences=True)

abstract_id_list = list()
word_list = list()
word_lower_list = list()
lemma_list = list()
lemma_lower_list = list()
pos_list = list()
ot_list = list()
ot_lower_list = list()
year_list = list()

lc = LoopTimer(update_after=100, avg_length=5000, target=len(abstracts))
for line in zip(ot_corpus, pos_corpus, word_corpus, lemma_corpus):
    abstract_id = line[0][0]
    ot_tokens = line[0][1]
    pos_tokens = line[1][1]
    word_tokens = line[2][1]
    lemma_tokens = line[3][1]
    year = year_db[abstract_id]

    ot_list.append("\t".join([" ".join(sentence) for sentence in ot_tokens]))
    ot_lower_list.append("\t".join([" ".join(sentence).lower() for sentence in ot_tokens]))
    word_list.append("\t".join([" ".join(sentence) for sentence in word_tokens]))
    word_lower_list.append("\t".join([" ".join(sentence).lower() for sentence in word_tokens]))
    lemma_list.append("\t".join([" ".join(sentence) for sentence in lemma_tokens]))
    lemma_lower_list.append("\t".join([" ".join(sentence).lower() for sentence in lemma_tokens]))

    pos_list.append("\t".join([" ".join(sentence) for sentence in pos_tokens]))

    year_list.append(year)
    abstract_id_list.append(abstract_id)
    lc.update("Read Abstract")

wordDF = pd.DataFrame(word_list, index=abstract_id_list, columns=["word"])
wordlowerDF = pd.DataFrame(word_lower_list, index=abstract_id_list, columns=["word"])

lemmaDF = pd.DataFrame(lemma_list, index=abstract_id_list, columns=["lemma"])
lemmalowerDF = pd.DataFrame(lemma_lower_list, index=abstract_id_list, columns=["lemma"])

otDF = pd.DataFrame(ot_list, index=abstract_id_list, columns=["originalText"])
otlowerDF = pd.DataFrame(ot_lower_list, index=abstract_id_list, columns=["originalText"])

posDF = pd.DataFrame(pos_list, index=abstract_id_list, columns=["pos"])
yearDF = pd.DataFrame(year_list, index=abstract_id_list, columns=['year'])


otDF.to_pickle(os.path.join(db_path, 'ai_ot.pandas'))
otlowerDF.to_pickle(os.path.join(db_path, 'ai_ot_lower.pandas'))

wordDF.to_pickle(os.path.join(db_path, 'ai_word.pandas'))
wordlowerDF.to_pickle(os.path.join(db_path, 'ai_word_lower.pandas'))

lemmaDF.to_pickle(os.path.join(db_path, 'ai_lemma.pandas'))
lemmalowerDF.to_pickle(os.path.join(db_path, 'ai_lemma_lower.pandas'))
posDF.to_pickle(os.path.join(db_path, 'ai_pos.pandas'))
yearDF.to_pickle(os.path.join(db_path, 'ai_year.pandas'))

