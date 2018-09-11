import os
import json

from src.utils.mysql import DBConnector
from src.utils.LoopTimer import LoopTimer
from src.utils.functions import check_string_for_english


path_to_db = "/media/norpheo/mySQL/db/ssorc/raw"

dbcon = DBConnector(db="ssorc")

#dbcon.delete_db()
#dbcon.setup()

raw_dir = '../../data/raw/ssorc'
file_list = sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])

for filename in file_list[1:2]:
    cur_path = os.path.join(raw_dir, filename)

    print(filename)

    lt = LoopTimer()
    with open(cur_path) as file:
        for idx, file_line in enumerate(file):
            data = json.loads(file_line)

            if ('title' in data) and ('authors' in data) and ('inCitations' in data) and ('outCitations' in data) and ('year' in data) and ('paperAbstract' in data) and ('id' in data):
                title = data['title']
                abstract = data['paperAbstract']
                abstract_id = data['id']
                year = data['year']
                authors = data['authors']
                inCitations = data['inCitations']
                outCitations = data['outCitations']

                if (year != '') and (len(abstract.split()) > 50) and (abstract_id != ''):
                    if check_string_for_english(abstract):
                        in_cit = ",".join(inCitations)
                        out_cit = ",".join(outCitations)
                        dbcon.add_abstract(abstract_id, title, authors, year, in_cit, out_cit)

                        with open(os.path.join(path_to_db, abstract_id + ".rawtxt"), "w") as abstract_file:
                            abstract_file.write(abstract)
                lt.update(f"Make Data")

            if idx % 2000 == 0:
                dbcon.commit()
    dbcon.commit()
    print()
    print()

#dbcon.print_db()