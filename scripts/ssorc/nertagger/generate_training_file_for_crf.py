import mysql.connector
import os
import pickle

from src.utils.corpora import nlp_to_doc_token
from src.utils.LoopTimer import LoopTimer

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")

sq1 = "SELECT abstract_id FROM abstracts WHERE year>2000 and annotated=1 LIMIT 50000"

print("Start request")
cursor.execute(sq1)
print("Start fetching")
result = cursor.fetchall()
print("Start printing")
print(len(result))
connection.close()

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations")
path_to_training_file = os.path.join(path_to_db, "NER", "training.txt")
path_to_mlalgo_list_file = os.path.join(path_to_db, "NER", "ml_algo_list.txt")

mlalgo_seed = list()
with open(path_to_mlalgo_list_file) as algo_list_file:
    for line in algo_list_file:
        mlalgo_seed.append(line.split())

with open(path_to_training_file, "w") as training_file:
    lc = LoopTimer(update_after=10, avg_length=500, target=len(result))
    for row in result:
        abstract_id = row[0]

        annotation_file_path = os.path.join(path_to_annotations, abstract_id + ".antn")

        if os.path.isfile(annotation_file_path):
            with open(annotation_file_path, "rb") as annotation_file:
                annotation = pickle.load(annotation_file)
            document = nlp_to_doc_token(annotation, "word", clean=False)

            abstract = " ".join(document)
            has_algo = False
            for algo in mlalgo_seed:
                algo = " ".join(algo).lower()
                if algo in abstract.lower():
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
                            if (idx + in_idx >= length) or (word2check.lower() != document[idx+in_idx].lower()):
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
        else:
            print()
            print(f"Missing {abstract_id}")
            print()

        lc.update("Writing Big File")