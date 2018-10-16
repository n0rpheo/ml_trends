import mysql.connector

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")

sql = "UPDATE abstracts SET isML=NULL WHERE isML IS NOT NULL"
cursor.execute(sql)
connection.close()