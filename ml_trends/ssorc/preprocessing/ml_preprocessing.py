import os
import json
import mysql.connector

from src.utils.LoopTimer import LoopTimer
from src.utils.functions import check_string_for_english


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_raw_text = "/media/norpheo/mySQL/db/ssorc/raw"
raw_dir = "/media/norpheo/mySQL/db/raw/ssorc"
file_list = sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )
cursor = connection.cursor()
cursor.execute("USE ssorc;")

req_keys = ['title',
            'authors',
            'inCitations',
            'outCitations',
            'year',
            'paperAbstract',
            'id',
            'entities']
categories = ['machine learning',
              'artificial intelligence']

count = 0

for filename in file_list[0:]:
    cur_path = os.path.join(raw_dir, filename)

    print()
    print(filename)

    lt = LoopTimer(update_after=10000, avg_length=50000, target=1000000)
    with open(cur_path) as file:
        for idx, file_line in enumerate(file):
            data = json.loads(file_line)

            if all(key in data for key in req_keys):
                title = data['title']
                abstract = data['paperAbstract']
                abstract_id = data['id']
                year = data['year']
                authors = data['authors']
                inCitations = data['inCitations']
                outCitations = data['outCitations']
                entities = data['entities']


                elist = set()
                for entity in entities:
                    elist.add(entity.lower())

                if (any(category in elist for category in categories) and
                        len(elist) <= 9999 and
                        abstract_id != '' and
                        year != '' and
                        len(abstract.split()) > 50 and
                        check_string_for_english(abstract)):
                    count += 1
                    in_cit = ",".join(inCitations)
                    out_cit = ",".join(outCitations)
                    entities_string = ",".join(elist)

                    # remove all non-utf-8 characters
                    title = bytes(title, 'latin1', 'ignore').decode('utf-8', 'ignore')
                    title = bytes(title, 'latin1', 'ignore').decode('utf-8', 'ignore')

                    author_ids = []
                    for author in authors:
                        author_name = author['name']

                        if len(author['ids']) != 1:
                            continue

                        author_id = int(author['ids'][0])

                        cursor.execute(f"SELECT abstract_ids FROM authors WHERE author_id={author_id};")
                        result = cursor.fetchall()

                        if len(result) == 0:
                            sq1 = "INSERT INTO authors (author_id, author, abstract_ids) VALUES(%s, %s, %s)"
                            cursor.execute(sq1, (author_id, author_name, abstract_id))
                        else:
                            abstract_ids = set(result[0][0].split(','))
                            abstract_ids.add(abstract_id)
                            abstract_ids = ",".join(abstract_ids)

                            sq1 = "UPDATE authors SET abstract_ids=%s WHERE author_id=%s;"
                            cursor.execute(sq1, (abstract_ids, author_id))

                        author_ids.append(author['ids'][0])

                    author_ids = ",".join(author_ids)

                    try:
                        sq1 = "REPLACE INTO abstracts_ml (abstract_id, title, author_ids, year, inCitations, outCitations, entities) VALUES(%s, %s, %s, %s, %s, %s, %s)"
                        cursor.execute(sq1, (abstract_id, title, author_ids, year, in_cit, out_cit, entities_string))
                    except Exception as err:
                        print()
                        print("Original:  '" + title + "'")
                        title = bytes(title, 'latin1', 'ignore').decode('utf-8', 'ignore')
                        print("After:     '" + title + "'")
                        print("Error {0}".format(err))
                        raise

                    with open(os.path.join(path_to_raw_text, abstract_id + ".rawtxt"), "w") as abstract_file:
                        abstract_file.write(abstract)
            lt.update(f"Make Data - {count}")

            if idx % 2000 == 0:
                connection.commit()
    connection.commit()

connection.close()
