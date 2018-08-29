import os
import json

import src.features.make_features as mf
import src.utils.functions as utils

from src.utils.LoopTimer import LoopTimer


class word_doc_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)
                abs_list = []

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Word Doc Stream")

                    data = json.loads(line)

                    doc_id = data['id']
                    xml = data['annotation']

                    if lastid != doc_id and len(abs_list) > 0:
                        # Yield Stuff
                        yield lastid, abs_list
                        abs_list = []

                    lastid = doc_id
                    token_list = mf.xml2words(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(token_cleaned) > 0:
                            for j in range(0, len(token_cleaned)):
                                abs_list.append(token_cleaned[j])
                if len(abs_list) > 0:
                    # Yield Stuff
                    yield doc_id, abs_list


class wordbigram_doc_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)
                abs_list = []

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Wordbigram Doc Stream")

                    data = json.loads(line)

                    doc_id = data['id']
                    xml = data['annotation']

                    if lastid != doc_id and len(abs_list) > 0:
                        # Yield Stuff
                        yield lastid, abs_list
                        abs_list = []

                    lastid = doc_id
                    token_list = mf.xml2words(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        token_cleaned = utils.makeBigrams(token_cleaned)

                        if len(token_cleaned) > 0:
                            for j in range(0, len(token_cleaned)):
                                abs_list.append(token_cleaned[j])
                if len(abs_list) > 0:
                    # Yield Stuff
                    yield doc_id, abs_list


class pos_doc_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)
                abs_list = []

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Pos Doc Stream")

                    data = json.loads(line)

                    doc_id = data['id']
                    xml = data['annotation']

                    if lastid != doc_id and len(abs_list) > 0:
                        # Yield Stuff
                        yield lastid, abs_list
                        abs_list = []

                    lastid = doc_id
                    token_list = mf.xml2words(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(pos_cleaned) > 0:
                            for j in range(0, len(pos_cleaned)):
                                abs_list.append(pos_cleaned[j])
                if len(abs_list) > 0:
                    # Yield Stuff
                    yield doc_id, abs_list


class posbigram_doc_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)
                abs_list = []

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Posbigram Doc Stream")

                    data = json.loads(line)

                    doc_id = data['id']
                    xml = data['annotation']

                    if lastid != doc_id and len(abs_list) > 0:
                        # Yield Stuff
                        yield lastid, abs_list
                        abs_list = []

                    lastid = doc_id
                    token_list = mf.xml2words(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        pos_cleaned = utils.makeBigrams(pos_cleaned)

                        if len(pos_cleaned) > 0:
                            for j in range(0, len(pos_cleaned)):
                                abs_list.append(pos_cleaned[j])
                if len(abs_list) > 0:
                    # Yield Stuff
                    yield doc_id, abs_list


class lemma_doc_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)
                abs_list = []

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Lemma Doc Stream")

                    data = json.loads(line)

                    doc_id = data['id']
                    xml = data['annotation']

                    if lastid != doc_id and len(abs_list) > 0:
                        # Yield Stuff
                        yield lastid, abs_list
                        abs_list = []

                    lastid = doc_id
                    token_list = mf.xml2lemmas(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(token_cleaned) > 0:
                            for j in range(0, len(token_cleaned)):
                                abs_list.append(token_cleaned[j])
                if len(abs_list) > 0:
                    # Yield Stuff
                    yield lastid, abs_list


class lemma_para_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)

                for line in file:
                    if self.print_status:
                        lc.update("Lemma Para Stream")

                    data = json.loads(line)

                    doc_id = data['id']
                    para_id = data['paragraphID']
                    xml = data['annotation']

                    token_list = mf.xml2lemmas(xml)
                    pos_list = mf.xml2pos(xml)

                    para_list = []
                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(token_cleaned) > 0:
                            for j in range(0, len(token_cleaned)):
                                para_list.append(token_cleaned[j])
                    yield doc_id, para_id, para_list


class word_sent_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Word Sent Stream")

                    data = json.loads(line)

                    xml = data['annotation']
                    id = data['id']
                    if lastid != id:
                        para_num = 0
                    else:
                        para_num += 1
                    lastid = id

                    token_list = mf.xml2words(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(token_cleaned) > 0:
                            yield id, para_num, token_cleaned


class wordbigram_sent_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Wordbigram Sent Stream")

                    data = json.loads(line)

                    xml = data['annotation']
                    id = data['id']
                    if lastid != id:
                        para_num = 0
                    else:
                        para_num += 1
                    lastid = id

                    token_list = mf.xml2words(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(token_cleaned) > 0:
                            yield id, para_num, utils.makeBigrams(token_cleaned)


class pos_sent_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Pos Sent Stream")

                    data = json.loads(line)

                    xml = data['annotation']
                    id = data['id']
                    if lastid != id:
                        para_num = 0
                    else:
                        para_num += 1
                    lastid = id

                    token_list = mf.xml2words(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(pos_cleaned) > 0:
                            yield id, para_num, pos_cleaned


class posbigram_sent_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)

                lastid = None
                for line in file:
                    if self.print_status:
                        lc.update("Posbigram Sent Stream")

                    data = json.loads(line)

                    xml = data['annotation']
                    id = data['id']
                    if lastid != id:
                        para_num = 0
                    else:
                        para_num += 1
                    lastid = id

                    token_list = mf.xml2words(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(token_cleaned) > 0:
                            yield id, para_num, utils.makeBigrams(pos_cleaned)


class lemma_sent_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)

                for line in file:
                    if self.print_status:
                        lc.update("Lemma Sent Stream")

                    data = json.loads(line)

                    xml = data['annotation']
                    doc_id = data['id']
                    para_num = data['paragraphID']

                    token_list = mf.xml2lemmas(xml)
                    pos_list = mf.xml2pos(xml)

                    for i in range(0, len(token_list)):
                        token_cleaned, pos_cleaned = utils.posFilterString(token_list[i], pos_list[i])

                        if len(token_cleaned) > 0:
                            yield doc_id, para_num, token_cleaned


class xml_para_stream(object):
    def __init__(self, dtype, print_status=False):
        self.print_status = print_status
        self.dirname = os.path.dirname(__file__)
        self.annotation_dir = os.path.join(self.dirname, '../../data/processed/' + dtype + '/annotation')

        self.file_list = sorted(
            [f for f in os.listdir(self.annotation_dir) if os.path.isfile(os.path.join(self.annotation_dir, f))])

    def __iter__(self):
        for filename in self.file_list[0:1]:
            sent_file = os.path.join(self.annotation_dir, filename)
            with open(sent_file) as file:
                lc = LoopTimer(update_after=100)

                for line in file:
                    if self.print_status:
                        lc.update("XML Para Stream")

                    data = json.loads(line)

                    xml = data['annotation']
                    doc_id = data['id']
                    para_num = data['paragraphID']

                    if xml.startswith('<?xml'):
                        yield doc_id, para_num, xml
