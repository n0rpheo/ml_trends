import mysql.connector
import string

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()

#cursor.execute("DROP DATABASE testing")
#cursor.execute("CREATE DATABASE IF NOT EXISTS testing")
cursor.execute("USE testing;")
cursor.execute("create table IF NOT EXISTS abstracts ("
                            "abstractID VARCHAR(64),"
                            "title TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,"
                            "authorID TEXT,"
                            "year INT,"
                            "inCitations MEDIUMTEXT,"
                            "outCitations MEDIUMTEXT,"
                            "PRIMARY KEY (abstractID));")


abstract_id = "a"
title = u'May-Äzerov Algorithm for Nearest-Neighbor Problem over 𝔽q and Its Application to Information Set Decoding'

#title = bytes(title, 'utf-8')
print(title)

author_ids = "Kevin"
year = 2001
inCitations = "keine"
outCitations = "auch keine"

sq1 = "INSERT INTO abstracts (abstractID, title, authorID, year, inCitations, outCitations) VALUES(%s, %s, %s, %s, %s, %s)"
try:
    cursor.execute(sq1, (abstract_id, title, author_ids, year, inCitations, outCitations))
except Exception as err:
    title = "".join([x for x in title if x in string.printable])
    print(title)
    cursor.execute(sq1, (abstract_id, title, author_ids, year, inCitations, outCitations))

