import os
import pickle
import mysql.connector

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )
cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = f"SELECT abstract_id, author_ids FROM abstracts_ml"
cursor.execute(sq1)

abstract_to_author_db = dict()

lt = LoopTimer(update_after=5000, avg_length=20000)
for result in cursor:
    abstract_id = result[0]
    if len(result[1]) == 0:
        continue

    abstract_to_author_db[abstract_id] = result[1]
    lt.update('Build DB')

with open(os.path.join(path_to_db, 'csrankings', 'abs2auth.dict'), 'wb') as a2afile:
    pickle.dump(abstract_to_author_db, a2afile)