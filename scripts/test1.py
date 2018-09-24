import mysql.connector
import pickle

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")

title = "May-Ozerov Algorithm for Nearest-Neighbor Problem over 𝔽q and Its Application to Information Set Decoding"
title = "Èöó×ô Blockinø× Óö ¹ööý Ëøùùùù× Óó Ððüý Ðù×øøö× Ûûøø ×øöó¹¹¾»»êë"

title = bytes(title, 'latin1', 'ignore').decode('utf-8', 'ignore')
title = bytes(title, 'latin1', 'ignore').decode('utf-8', 'ignore')

sq1 = "SELECT abstract_id FROM abstracts WHERE annotated=1"

cursor.execute(sq1)

result = cursor.fetchall()

print(len(result))

connection.close()
