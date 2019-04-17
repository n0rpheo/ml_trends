import pickle
import os
import mysql.connector

from src.utils.LoopTimer import LoopTimer

path_to_ner = "/media/norpheo/mySQL/db/ssorc/NER"
path_to_ml_algo_abstract_save = os.path.join(path_to_ner, "ml_algo_abstract_new.pickle")


with open(path_to_ml_algo_abstract_save, "rb") as algo_abstract_file:
    ml_algo_abstract = pickle.load(algo_abstract_file)

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")


lc = LoopTimer(update_after=100, avg_length=10000)
for idx, abstract_id in enumerate(ml_algo_abstract):
    sq1 = f'INSERT INTO mlabstracts (abstract_id) VALUES("{abstract_id}") ON DUPLICATE KEY UPDATE abstract_id = ("{abstract_id}")'
    cursor.execute(sq1)

    lc.update("Insert Into")

connection.commit()
connection.close()