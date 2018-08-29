import json
import os

from src.utils.functions import check_string_for_english
from src.utils.LoopTimer import LoopTimer

def make_dblp_data():
    dirname = os.path.dirname(__file__)
    raw_dir = os.path.join(dirname, '../../data/raw/dblp')
    data_abstract_file = os.path.join(raw_dir, 'data_abstracts.tsv')
    data_information_file = os.path.join(raw_dir, 'data_information.tsv')

    json_dir = os.path.join(dirname, '../../data/processed/dblp/json')
    json_file = os.path.join(json_dir, 'dblp.json')
    if os.path.isfile(json_file):
        os.remove(json_file)

    with open(data_information_file) as infofile:
        with open(data_abstract_file) as abstractfile:
            with open(json_file, 'a') as jfile:
                lt = LoopTimer()
                count = 0
                for infoline, abstractline in zip(infofile, abstractfile):
                    infodata = infoline.split('\t')
                    abstractdata = abstractline.split('\t')

                    infoID = infodata[0]
                    infoDOI = infodata[1]
                    infoTitle = infodata[2]
                    infoAuthors = infodata[3]
                    infoYear = int(infodata[4])

                    abstractID = abstractdata[0]
                    abstractContent = abstractdata[1]

                    if check_string_for_english(abstractContent):
                        new_data = {}
                        new_data['year'] = infoYear
                        new_data['paperAbstract'] = abstractContent
                        new_data['id'] = infoDOI
                        jsonstring = json.JSONEncoder().encode(new_data)
                        jfile.write(jsonstring + '\n')
                        count += 1
                    lt.update(str(count) + " Abstracts added")


def make_ssorc_data():
    dirname = os.path.dirname(__file__)
    raw_dir = os.path.join(dirname, '../../data/raw/ssorc')
    file_list = sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])
    json_dir = os.path.join(dirname, '../../data/processed/ssorc/json')

    for filename in file_list[1:2]:
        cur_path = os.path.join(raw_dir, filename)

        json_file = os.path.join(json_dir, filename + '.json')
        if os.path.isfile(json_file):
            os.remove(json_file)

        print(filename)


        lt = LoopTimer()
        with open(json_file, 'a') as wfile:
            with open(cur_path) as file:
                for idx, file_line in enumerate(file):
                    data = json.loads(file_line)

                    if ('year' in data) and ('paperAbstract' in data) and ('doi' in data):
                        if (data['year'] != '') and (len(data['paperAbstract'].split()) > 50) and (data['doi'] != ''):
                            if check_string_for_english(data['paperAbstract']):
                                new_data = {}
                                new_data['year'] = data['year']
                                new_data['paperAbstract'] = data['paperAbstract']
                                new_data['id'] = data['doi']
                                jsonstring = json.JSONEncoder().encode(new_data)
                                wfile.write(jsonstring + '\n')
                    lt.update("Make Data")


def paragraph_splitter(dtype):
    dirname = os.path.dirname(__file__)
    json_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/json')
    file_list = sorted([f for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, f)) and f.endswith('.json')])

    max_para_title_words = 2    # Maximum Number of words for a line to be considered as a paragraph_title
    min_para_words = 10         # Minimum Number of words for a line to be considered as a paragraph

    for filename in file_list:
        cur_path = os.path.join(json_dir, filename)

        paragraph_file = os.path.join(json_dir, filename + '.para')
        if os.path.isfile(paragraph_file):
            os.remove(paragraph_file)

        print(filename)

        lt = LoopTimer()
        with open(paragraph_file, 'a') as wfile:
            with open(cur_path) as file:
                for file_line in file:
                    data = json.loads(file_line)
                    lines = data['paperAbstract'].split('\n')
                    paragraph_title = ''
                    paragraph_id = 0
                    new_data = {}
                    empty_line_allowed = False
                    for line in lines:
                        # Check if line machtes a Paragraph-Title-Pattern
                        if len(line) > 0 and (len(line.split()) <= max_para_title_words or line == line.upper()):
                            paragraph_title = line
                            empty_line_allowed = True
                        elif len(line) > 0 and (len(line.split()) >= min_para_words):
                            new_data['paragraphContent'] = line
                            new_data['year'] = data['year']
                            new_data['id'] = data['id']
                            new_data['paragraphTitle'] = paragraph_title
                            new_data['paragraphID'] = paragraph_id

                            json_string = json.JSONEncoder().encode(new_data)
                            wfile.write(json_string + '\n')

                            empty_line_allowed = False
                            paragraph_id += 1
                        elif len(line) == 0 and not empty_line_allowed:
                            paragraph_title = ''
                    lt.update_after("Para-Split")


def rf_label_ssorc_paragraphs():
    dirname = os.path.dirname(__file__)
    json_dir = os.path.join(dirname, '../../data/processed/ssorc/json')
    file_list = sorted([f for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, f)) and f.endswith('.para')])

    target_file_path = os.path.join(dirname, '../../data/processed/ssorc/rf_targets/targets.json')
    if os.path.isfile(target_file_path):
        os.remove(target_file_path)

    labels = dict()
    with open(os.path.join(dirname, '../../data/definitions/rf_labels.txt')) as rfdef:
        for line in rfdef:
            linesplit = line.split('\t')
            label = linesplit[0]
            candidates = linesplit[1].rstrip().split(',')
            labels[label] = []
            for candidate in candidates:
                if len(candidate) > 0:
                    candidate = candidate.lower()
                    labels[label].append(candidate)
                    labels[label].append(candidate + ":")

    with open(target_file_path, 'a') as target_file:
        for filename in file_list:
            cur_path = os.path.join(json_dir, filename)

            lt = LoopTimer()
            with open(cur_path) as file:
                for idx, file_line in enumerate(file):
                    data = json.loads(file_line)
                    title = data['paragraphTitle'].lower()

                    rf_label = -1
                    for key in labels:
                        if title in labels[key]:
                            rf_label = key
                            break

                    if rf_label != -1:
                        target_data = {}
                        target_data['id'] = data['id']
                        target_data['paragraphID'] = data['paragraphID']
                        target_data['rflabel'] = rf_label

                        json_string = json.JSONEncoder().encode(target_data)
                        target_file.write(json_string + '\n')
                    lt.update("RF Labeling")

