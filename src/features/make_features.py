import os
import json
import xml.etree.ElementTree as ET
from time import time
from collections import deque
from stanfordcorenlp import StanfordCoreNLP


def nlp_annotation(dtype, annotators='tokenize,ssplit,pos,lemma,depparse'):
    nlp = StanfordCoreNLP('../stanford-corenlp-full-2018-02-27')
    props = {'annotators': annotators, 'pipelineLanguage': 'en', 'outputFormat': 'xml'}

    dirname = os.path.dirname(__file__)
    root_dir = os.path.join(dirname, '../../')
    dir = os.path.join(root_dir, 'data/processed/' + dtype + '/json')

    annotation_dir = os.path.join(root_dir, 'data/processed/' + dtype + '/annotation')

    file_list = sorted(
        [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith('.para')])

    for filename in file_list[0:1]:
        cur_path = os.path.join(dir, filename)
        line_number = 0

        subfilename = filename[0:filename.find('.')]
        annotation_file = os.path.join(annotation_dir, subfilename + '_annotation.json')
        noskip = True
        if os.path.isfile(annotation_file):
            print("File " + subfilename + " found. Checking last line information")
            with open(annotation_file) as anno_file:
                lc = 0
                for line in anno_file:
                    curline = line
                    print(lc, end="\r")
                    lc += 1
            cur_data = json.loads(curline)
            id = cur_data['id']
            pid = cur_data['paragraphID']
            noskip = False
            print()
            print("Information gathered!")

        with open(cur_path) as file:
            avg_time = deque(100 * [float(0)], maxlen=100)
            for file_line in file:
                t0 = time()
                data = json.loads(file_line)

                if noskip:
                    abstract = data['paragraphContent']

                    annotation = nlp.annotate(abstract, properties=props)

                    annotation_data = {}
                    annotation_data['id'] = data['id']
                    annotation_data['paragraphID'] = data['paragraphID']
                    annotation_data['annotation'] = annotation

                    json_string = json.JSONEncoder().encode(annotation_data)

                    with open(annotation_file, 'a') as wfile:
                        wfile.write(json_string + '\n')
                else:
                    if id == data['id'] and pid == data['paragraphID']:
                        noskip = True

                t1 = time()
                avg_time.append(t1 - t0)
                it_per_second = 100 / sum(avg_time)
                print("Annotate " + filename + " | " + str(line_number) + ' | ' + str(it_per_second) + ' / s     ', end='\r')
                line_number += 1

        nlp.close()


def xml2words(xml_string):
    document = []

    if xml_string.startswith('<?xml'):
        root = ET.fromstring(xml_string)
        for sentence in root.iter('sentence'):
            tokens = []
            for token in sentence.iter('token'):
                tokens.append(token.find('word').text)
            document.append(tokens)
    return document


def xml2lemmas(xml_string):
    document = []

    if xml_string.startswith('<?xml'):
        root = ET.fromstring(xml_string)
        for sentence in root.iter('sentence'):
            tokens = []
            for token in sentence.iter('token'):
                tokens.append(token.find('lemma').text)
            document.append(tokens)
    return document


def xml2pos(xml_string):
    document = []

    if xml_string.startswith('<?xml'):
        root = ET.fromstring(xml_string)
        for sentence in root.iter('sentence'):
            tokens = []
            for token in sentence.iter('token'):
                tokens.append(token.find('POS').text)
            document.append(tokens)
    return document


def xml2depparse(xml_string):
    root_ids = []
    childs = []

    if xml_string.startswith('<?xml'):
        root = ET.fromstring(xml_string)
        for sentence in root.iter('sentence'):
            for deps in sentence.iter('dependencies'):
                if deps.attrib['type'] == 'basic-dependencies':
                    root_id = None
                    child_ids = []
                    for dep in deps.iter('dep'):
                        if dep.attrib['type'] == 'root':
                            root_id = int(dep.find('dependent').attrib['idx'])-1
                        if root_id is not None:
                            if int(dep.find('governor').attrib['idx']) == root_id+1:
                                child_ids.append(int(dep.find('dependent').attrib['idx']))
                    root_ids.append(root_id)
                    childs.append(child_ids)
    return root_ids, childs

