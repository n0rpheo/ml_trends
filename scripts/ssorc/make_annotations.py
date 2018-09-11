import os
import json
import pickle

from stanfordcorenlp import StanfordCoreNLP

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"

path_to_annotations = os.path.join(path_to_db, "annotations")

path_to_raw = os.path.join(path_to_db, "raw")

raw_list = set([f[0:(len(f)-7)] for f in os.listdir(path_to_raw) if os.path.isfile(os.path.join(path_to_raw, f))])

annotation_list = set([f[0:(len(f)-7)] for f in os.listdir(path_to_annotations)
                       if os.path.isfile(os.path.join(path_to_annotations, f))])

to_process_list = raw_list - annotation_list

print("There are " + str(len(to_process_list)) + " files to process")

annotators = 'tokenize,ssplit,pos,lemma,depparse'
nlp = StanfordCoreNLP('../../stanford-corenlp-full-2018-02-27')
props = {'annotators': annotators, 'pipelineLanguage': 'en', 'outputFormat': 'json'}

lc = LoopTimer(update_after=1)
for idx, filename in enumerate(to_process_list):
    lc.update("Annotate Abstract " + filename)
    with open(os.path.join(path_to_raw, filename + ".rawtxt")) as rawfile:
        abstract = rawfile.read()

        annotation = nlp.annotate(abstract, properties=props)

        data = json.loads(annotation)

        with open(os.path.join(path_to_annotations, filename + ".antn"), "wb") as anno_file:
            pickle.dump(data, anno_file, protocol=pickle.HIGHEST_PROTOCOL)

nlp.close()


def paragraph_splitter(split_abstract):
    max_para_title_words = 2    # Maximum Number of words for a line to be considered as a paragraph_title
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

