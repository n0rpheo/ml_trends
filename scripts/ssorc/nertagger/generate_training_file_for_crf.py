import mysql.connector
import os

from src.utils.corpora import TokenDocStream
from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_training_file = os.path.join(path_to_db, "NER", "new_training.txt")
path_to_mlalgo_list_file = os.path.join(path_to_db, "NER", "new_ml_algos.txt")

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")

sq1 = "SELECT abstract_id FROM abstracts WHERE year>1990 and annotated=1 LIMIT 500000"
print("Start request")
cursor.execute(sq1)

abstracts = set()
lc = LoopTimer(update_after=10000, avg_length=200000)
for row in cursor:
    abstracts.add(row[0])
    lc.update("Collecting")
connection.close()

docstream = TokenDocStream(token_type="originalText",
                           abstracts=abstracts,
                           token_cleaned=False,
                           print_status=True,
                           output=None,
                           print_settings={"update_after": 50, "avg_length": 10000})

mlalgo_seed = list()
with open(path_to_mlalgo_list_file) as algo_list_file:
    for line in algo_list_file:
        mlalgo_seed.append(line.split())

with open(path_to_training_file, "w") as training_file:
    for document in docstream:
        abstract = " ".join(document)
        has_algo = False
        for algo in mlalgo_seed:
            algo = " ".join(algo)
            if algo in abstract:
                has_algo = True
                break

        if has_algo:
            idx = 0
            length = len(document)
            while idx < length:
                for ml2check in mlalgo_seed:
                    in_idx = 0
                    is_mlalgo = True
                    for word2check in ml2check:
                        if (idx + in_idx >= length) or (word2check != document[idx+in_idx]):
                            is_mlalgo = False
                            break
                        in_idx += 1

                    if is_mlalgo:
                        break
                if is_mlalgo:
                    for i in range(0, in_idx):
                        line = f"{document[idx]}\tMLALGO\n"
                        training_file.write(line)
                        idx += 1
                else:
                    line = f"{document[idx]}\tO\n"
                    training_file.write(line)
                    idx += 1
            training_file.write("\n")