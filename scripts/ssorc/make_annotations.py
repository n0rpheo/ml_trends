import os
import json
import pickle

import mysql.connector
from stanfordcorenlp import StanfordCoreNLP

from src.utils.mysql import DBConnector
from src.utils.LoopTimer import LoopTimer


def paragraph_splitter(split_abstract):
    max_para_title_words = 3    # Maximum Number of words for a line to be considered as a paragraph_title
    min_para_words = 10         # Minimum Number of words for a line to be considered as a paragraph

    lines = split_abstract.split('\n')
    paragraph_title = ''
    empty_line_allowed = False

    paras = list()

    for line in lines:
        # Check if line matches a Paragraph-Title-Pattern
        if len(line) > 0 and (len(line.split()) <= max_para_title_words or line == line.upper()):
            paragraph_title = line
            empty_line_allowed = True
        elif len(line) > 0 and (len(line.split()) >= min_para_words):
            para_data = dict()
            para_data['paragraphContent'] = line
            para_data['paragraphTitle'] = paragraph_title

            paras.append(para_data)

            empty_line_allowed = False
        elif len(line) == 0 and not empty_line_allowed:
            paragraph_title = ''
    return paras


dbcon = DBConnector(db="ssorc")
path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations")
path_to_raw = os.path.join(path_to_db, "raw")

connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="thesis",
        )

cursor = connection.cursor()
cursor.execute("USE ssorc;")
cursor.execute("SELECT abstract_id FROM abstracts WHERE annotated=0")

lc = LoopTimer(update_after=1000)
abstracts_to_process = set()
for idx, row in enumerate(cursor):
    abstracts_to_process.add(row[0])
    lc.update("Collect Abstracts to Process")
connection.close()
print()

print("There are " + str(len(abstracts_to_process)) + " files to process")

annotators = 'tokenize,ssplit,pos,lemma,depparse'
splitter_annotators = 'ssplit'
nlp = StanfordCoreNLP('../../stanford-corenlp-full-2018-02-27')
props = {'annotators': annotators, 'pipelineLanguage': 'en', 'outputFormat': 'json'}
split_props = {'annotators': splitter_annotators, 'pipelineLanguage': 'en', 'outputFormat': 'json'}

lc = LoopTimer(update_after=1)
for idx, abstract_id in enumerate(abstracts_to_process):
    lc.update("Annotate Abstract " + abstract_id)

    with open(os.path.join(path_to_raw, abstract_id + ".rawtxt")) as rawfile:
        abstract = rawfile.read()

    paragraphs = paragraph_splitter(abstract)

    last_id = 0
    para_info = dict()
    para_info['sentences'] = list()
    for paragraph in paragraphs:
        para_text = paragraph['paragraphContent']
        para_title = paragraph['paragraphTitle']
        para_anno = json.loads(nlp.annotate(para_text, properties=split_props))

        number_of_sentences = len(para_anno['sentences'])

        for i in range(last_id, last_id + number_of_sentences):
            infos = dict()
            infos['label'] = para_title
            infos['index'] = i

            para_info['sentences'].append(infos)

        last_id += number_of_sentences

    para_info_text = json.dumps(para_info)
    new_abstract = " ".join([x['paragraphContent'] for x in paragraphs])
    annotation = nlp.annotate(new_abstract, properties=props)
    data = json.loads(annotation)
    with open(os.path.join(path_to_annotations, abstract_id + ".antn"), "wb") as anno_file:
        pickle.dump(data, anno_file, protocol=pickle.HIGHEST_PROTOCOL)

    dbcon.add_rflabel_info(abstract_id, para_info_text)
    dbcon.annotate(abstract_id)

    dbcon.commit()

    if idx == 20000:
        break

nlp.close()
