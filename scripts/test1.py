import mysql.connector

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE test;")

title = "May-Ozerov Algorithm for Nearest-Neighbor Problem over 𝔽q and Its Application to Information Set Decoding"
title = "Blockin        󹹾"

title = bytes(title, 'latin1', 'ignore').decode('utf-8', 'ignore')

print(title)

try:
    sq1 = f"INSERT INTO testing (title) VALUES('{title}')"
    cursor.execute(sq1)
    connection.commit()
except Exception as err:
    print("'" + title + "'")
    print("Error {0}".format(err))
    raise

connection.close()
