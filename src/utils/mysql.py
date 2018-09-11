import mysql.connector
import string


class DBConnector:
    def __init__(self, db=None):
        self.connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

        self.cursor = self.connection.cursor()

        if db is not None:
            self.cursor.execute(f"USE {db}")

    def delete_db(self):
        self.cursor.execute("DROP DATABASE ssorc")

    def setup(self):
        self.cursor.execute("CREATE DATABASE IF NOT EXISTS ssorc")
        self.cursor.execute("USE ssorc")
        self.cursor.execute("create table IF NOT EXISTS abstracts ("
                            "abstractID VARCHAR(64),"
                            "title TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,"
                            "authorID TEXT,"
                            "year INT,"
                            "inCitations MEDIUMTEXT,"
                            "outCitations MEDIUMTEXT,"
                            "PRIMARY KEY (abstractID));")

        self.cursor.execute("create table IF NOT EXISTS authors ("
                            "authorID BIGINT UNSIGNED,"
                            "author TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,"
                            "PRIMARY KEY (authorID));")

    def add_abstract(self, abstract_id, title, authors, year, inCitations, outCitations):
        self.cursor.execute(f"SELECT EXISTS(SELECT * FROM abstracts WHERE abstractID='{abstract_id}');")
        result = self.cursor.fetchall()

        if result[0][0] == 0:
            title = bytes(title, 'utf-8').decode('utf-8', 'ignore')

            author_ids = []
            for author in authors:
                author_name = author['name']

                if len(author['ids']) != 1:
                    continue

                author_id = int(author['ids'][0])

                self.cursor.execute(f"SELECT EXISTS(SELECT * FROM authors WHERE authorID={author_id});")
                result = self.cursor.fetchall()

                if result[0][0] == 0:
                    sq1 = "INSERT INTO authors (authorID, author) VALUES(%s, %s)"
                    try:
                        self.cursor.execute(sq1, (author_id, author_name))
                    except:
                        author_name = "".join([x for x in author_name if x in string.printable])
                        self.cursor.execute(sq1, (author_id, author_name))

                author_ids.append(author['ids'][0])

            author_ids = ",".join(author_ids)

            sq1 = "INSERT INTO abstracts (abstractID, title, authorID, year, inCitations, outCitations) VALUES(%s, %s, %s, %s, %s, %s)"

            try:
                self.cursor.execute(sq1, (abstract_id, title, author_ids, year, inCitations, outCitations))
            except:
                title = "".join([x for x in title if x in string.printable])
                self.cursor.execute(sq1, (abstract_id, title, author_ids, year, inCitations, outCitations))

    def commit(self):
        self.connection.commit()

    def print_db(self):
        self.cursor.execute("SELECT * FROM abstracts")

        for entry in self.cursor:
            print(entry)

        self.cursor.execute("SELECT * FROM authors")

        for entry in self.cursor:
            print(entry)
