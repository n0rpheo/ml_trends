import mysql.connector

from src.utils.corpora import TokenDocStream

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id FROM abstracts WHERE isML=1 LIMIT 10"
cursor.execute(sq1)

abstracts = set()
for row in cursor:
    abstracts.add(row[0])
connection.close()

corpus = TokenDocStream(abstracts=abstracts, token_type="originalText", token_cleaned=False, print_status=False)

for doc in corpus:
    words = list()
    for word in doc:
        words.append(word)
        if len(words) == 10:
            print(" ".join(words))
            words = list()
    if len(words) > 0:
        print(" ".join(words))
    print("--------------------------")
