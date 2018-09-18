import mysql.connector

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()

cursor.execute("DROP DATABASE ssorc")

cursor.execute("CREATE DATABASE IF NOT EXISTS ssorc")

cursor.execute("USE ssorc")
cursor.execute("create table IF NOT EXISTS abstracts ("
               "abstract_id VARCHAR(64),"
               "title TEXT,"
               "author_ids TEXT,"
               "year INT,"
               "inCitations MEDIUMTEXT,"
               "outCitations MEDIUMTEXT,"
               "annotated Boolean DEFAULT 0,"
               "PRIMARY KEY (abstract_id));")

cursor.execute("create table IF NOT EXISTS authors ("
               "author_id BIGINT UNSIGNED,"
               "author TEXT,"
               "abstract_ids MEDIUMTEXT,"
               "PRIMARY KEY (author_id));")

cursor.execute("create table IF NOT EXISTS rfLabels ("
               "abstract_id VARCHAR(64),"
               "labelinfo MEDIUMTEXT,"
               "PRIMARY KEY (abstract_id));")
