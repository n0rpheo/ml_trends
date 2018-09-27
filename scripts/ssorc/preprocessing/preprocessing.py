import os
import json
import mysql.connector

from src.utils.mysql import DBConnector
from src.utils.LoopTimer import LoopTimer
from src.utils.functions import check_string_for_english


path_to_db = "/media/norpheo/mySQL/db/ssorc/raw"

dbcon = DBConnector(db="ssorc")

raw_dir = '../../../data/raw/ssorc'
file_list = sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )
cursor = connection.cursor()
cursor.execute("USE ssorc;")
cursor.execute("SELECT abstract_id FROM abstracts;")
lc = LoopTimer(update_after=1000)
processed = set()
for idx, row in enumerate(cursor):
    processed.add(row[0])
    lc.update("Collect Processed Abstracts")
connection.close()


for filename in file_list[0:3]:
    cur_path = os.path.join(raw_dir, filename)

    print()
    print(filename)

    lt = LoopTimer()
    with open(cur_path) as file:
        for idx, file_line in enumerate(file):
            data = json.loads(file_line)

            if (('title' in data) and
                    ('authors' in data) and
                    ('inCitations' in data) and
                    ('outCitations' in data) and
                    ('year' in data) and
                    ('paperAbstract' in data) and
                    ('id' in data)):
                title = data['title']
                abstract = data['paperAbstract']
                abstract_id = data['id']
                year = data['year']
                authors = data['authors']
                inCitations = data['inCitations']
                outCitations = data['outCitations']

                if abstract_id not in processed:
                    if ((abstract_id != '') and
                            (year != '') and
                            (len(abstract.split()) > 50) and
                            check_string_for_english(abstract)):

                        in_cit = ",".join(inCitations)
                        out_cit = ",".join(outCitations)

                        # remove all non-utf-8 characters
                        title = bytes(title, 'latin1', 'ignore').decode('utf-8', 'ignore')
                        title = bytes(title, 'latin1', 'ignore').decode('utf-8', 'ignore')

                        dbcon.add_abstract(abstract_id, title, authors, year, in_cit, out_cit)

                        with open(os.path.join(path_to_db, abstract_id + ".rawtxt"), "w") as abstract_file:
                            abstract_file.write(abstract)
                else:
                    processed.remove(abstract_id)
            lt.update("Make Data")

            if idx % 2000 == 0:
                dbcon.commit()
    dbcon.commit()
