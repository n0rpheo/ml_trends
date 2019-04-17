import os
import pandas as pd
import mysql.connector
from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
worddb_path = os.path.join(path_to_db, 'pandas', 'ml_word.pandas')

wordDF = pd.read_pickle(worddb_path)

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )
cursor = connection.cursor()
cursor.execute("USE ssorc;")

author_list = list()
author_id_list = list()
check_list = set()

lc = LoopTimer(update_after=10, avg_length=1000, target=len(wordDF))
for index, row in wordDF.iterrows():
    abstract_id = index
    sq1 = f"SELECT author_ids FROM abstracts_ml WHERE abstract_id='{abstract_id}'"
    cursor.execute(sq1)

    author_ids = cursor.fetchall()[0][0].split(',')

    for author_id in author_ids:
        if len(author_id) == 0:
            continue
        author_id = int(author_id)
        sq1 = f"SELECT author FROM authors WHERE author_id='{author_id}'"
        cursor.execute(sq1)
        author = cursor.fetchall()[0][0]

        if author_id not in check_list:
            author_id_list.append(author_id)
            author_list.append(author)
            check_list.add(author_id)

    lc.update("Build Author DB")

author_db = pd.DataFrame(author_list, index=author_id_list, columns=["authors"])
author_db.to_pickle(os.path.join(path_to_db, 'pandas', 'author_db.pandas'))
