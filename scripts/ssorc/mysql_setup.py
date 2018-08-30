import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="masterarbeit",
    database="ssorc"
)

cursor = mydb.cursor()

# mycursor.execute("CREATE DATABASE ssorc")

cursor.execute("")

cursor.execute("SHOW DATABASES")

for x in cursor:
    print(x)

#test