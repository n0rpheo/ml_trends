import mysql.connector
from src.utils.LoopTimer import LoopTimer

from collections import Counter

db_path = "/media/norpheo/mySQL/db/ssorc/pandas"

connection = mysql.connector.connect(host="localhost",
                                     user="root",
                                     passwd="thesis")

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT entities FROM abstracts_ml WHERE entities LIKE '%machine learning%' OR entities LIKE '%artificial intelligence%' AND annotated=1"
cursor.execute(sq1)

print("Collect Abstracts from DB")

occurrences = Counter()

lt = LoopTimer(update_after=1000, avg_length=5000)
for row in cursor:
    entities = (row[0].split(','))
    occ = Counter(entities)

    occurrences += occ

    lt.update("Colleting")
connection.close()

for item in occurrences.most_common(100):
    entity = item[0]
    occ = item[1]

    print(f"{occ}:\t{entity}")
#print(occ2)

