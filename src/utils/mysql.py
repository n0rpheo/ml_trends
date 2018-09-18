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

    def add_abstract(self, abstract_id, title, authors, year, inCitations, outCitations):
        self.cursor.execute(f"SELECT abstract_id FROM abstracts WHERE abstract_id='{abstract_id}';")
        result = self.cursor.fetchall()

        if len(result) == 0:
            author_ids = []
            for author in authors:
                author_name = author['name']

                if len(author['ids']) != 1:
                    continue

                author_id = int(author['ids'][0])

                self.cursor.execute(f"SELECT abstract_ids FROM authors WHERE author_id={author_id};")
                result = self.cursor.fetchall()

                if len(result) == 0:
                    sq1 = "INSERT INTO authors (author_id, author, abstract_ids) VALUES(%s, %s, %s)"
                    self.cursor.execute(sq1, (author_id, author_name, abstract_id))
                else:
                    abstract_ids = set(result[0][0].split(','))
                    abstract_ids.add("6")
                    abstract_ids = ",".join(abstract_ids)

                    sq1 = "UPDATE authors SET abstract_ids=%s WHERE author_id=%s;"
                    self.cursor.execute(sq1, (abstract_ids, author_id))

                author_ids.append(author['ids'][0])

            author_ids = ",".join(author_ids)

            try:
                sq1 = "INSERT INTO abstracts (abstract_id, title, author_ids, year, inCitations, outCitations) VALUES(%s, %s, %s, %s, %s, %s)"
                self.cursor.execute(sq1, (abstract_id, title, author_ids, year, inCitations, outCitations))
            except Exception as err:
                print("'" + title + "'")
                print("Error {0}".format(err))
                raise

    def add_rflabel_info(self, abstract_id, labelinfo):
        sq1 = "INSERT INTO rfLabels (abstract_id, labelinfo) VALUES(%s, %s)"
        try:
            self.cursor.execute(sq1, (abstract_id, labelinfo))
        except:
            print("ERROR")
            exit()

    def annotate(self, abstract_id):
        sq1 = f"UPDATE abstracts SET annotated=1 WHERE abstract_id='{abstract_id}';"
        self.cursor.execute(sq1)

    def commit(self):
        self.connection.commit()
