import mysql.connector
import os
from src.utils.LoopTimer import LoopTimer

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")

sq1 = "SELECT abstract_id FROM abstracts WHERE year>2000 and annotated=1 LIMIT 1000"

print("Start request")
cursor.execute(sq1)
print("Start fetching")
result = cursor.fetchall()
print("Start printing")
print(len(result))
connection.close()

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_raw = os.path.join(path_to_db, "raw")
path_to_big_file = os.path.join(path_to_db, "patterns", "bigfile1k.txt")

with open(path_to_big_file, "w") as bigfile:

    lc = LoopTimer(update_after=10, avg_length=500, target=len(result))
    for row in result:
        abstract_id = row[0]

        abstract_file_path = os.path.join(path_to_raw, abstract_id + ".rawtxt")

        if os.path.isfile(abstract_file_path):
            with open(abstract_file_path, "r") as file:
                abstract = file.readlines()
            abstract = " ".join([x.strip() for x in abstract])

            bigfile.write(abstract + "\n")
        else:
            print()
            print(f"Missing {abstract_id}")
            print()

        lc.update("Writing Big File")