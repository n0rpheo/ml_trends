import os
import json
import pickle
import spacy
import pandas as pd

from src.utils.LoopTimer import LoopTimer
from src.utils.functions import check_string_for_english
from ml_trends.ssorc.statistics import jourven_list


path_to_db = "/media/norpheo/mySQL/db/ssorc"

"""
NLP STANDARD
==================================
"""
nlp_model_standard = 'en_core_web_sm'
"""
==================================
"""

"""
NLP with Trained NER
==================================
"""
nlp_model_trained = "en_wa_v2"
nlp_model_trained_path = os.path.join(path_to_db, "models", nlp_model_trained)
#component = nlp_ner.create_pipe("merge_entities")
#nlp_ner.add_pipe(component)
"""
==================================
"""

nlp_path = nlp_model_trained_path
nlp_model = nlp_model_trained
nlp = spacy.load(nlp_path)

raw_dir = "/media/norpheo/Daten/Masterarbeit/python_server/ml_trends/data/raw/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)

if not os.path.isdir(path_to_annotations):
    print(f"Create Directory {path_to_annotations}")
    os.mkdir(path_to_annotations)
    count = 0
    author_dic = dict()

    abstract_id_list = list()

    year_list = list()
    author_list = list()
    entity_list = list()
    journal_list = list()
    venue_list = list()
else:
    nlp.vocab = nlp.vocab.from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
    with open(os.path.join(path_to_annotations, "author.dic"), "rb") as handle:
        author_dic = pickle.load(handle)
    infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))
    year_list = infoDF['year'].tolist()
    author_list = infoDF['authors'].tolist()
    entity_list = infoDF['entities'].tolist()
    journal_list = infoDF['journal'].tolist()
    venue_list = infoDF['venue'].tolist()

    abstract_id_list = infoDF.index.tolist()

    count = len(infoDF)


file_list = sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])

req_keys = ['title',
            'authors',
            'inCitations',
            'outCitations',
            'year',
            'paperAbstract',
            'id',
            'entities',
            'journalName',
            'venue']

journals = list()
venues = list()

for region in jourven_list.venues:
    vr = jourven_list.venues[region]
    for mag in vr:
        venues.append(mag)
for region in jourven_list.journals:
    vr = jourven_list.journals[region]
    for mag in vr:
        journals.append(mag)

filerange = [0, 39]
filerange[1] = min(filerange[1], 39)
blastfile = 1 if filerange[1] == 39 else 0
target = min(39, (filerange[1]-filerange[0]))*1000000+blastfile*219709

key_error = 0
mass_error = 0
prune_error = 0

lt = LoopTimer(update_after=500, avg_length=1000000, target=target)
for filename in file_list[filerange[0]:filerange[1]]:
    cur_path = os.path.join(raw_dir, filename)
    with open(cur_path) as file:
        for idx, file_line in enumerate(file):
            update_string = f"Prep  - Count:{count} |  key: {key_error} - different: {mass_error} - One Char: {prune_error}"
            break_p = lt.update(update_string)
            data = json.loads(file_line)
            if not all(key in data for key in req_keys):
                key_error += 1
                continue
            title = data['title']
            abstract = data['paperAbstract']
            abstract_id = data['id']

            year = data['year']
            authors = data['authors']
            inCitations = data['inCitations']
            outCitations = data['outCitations']

            entities = data['entities']
            journal = data['journalName'].lower()
            venue = data['venue'].lower()

            elist = set()
            for entity in entities:
                elist.add(entity.lower())

            if ((journal not in journals and venue not in venues) or
                    abstract_id == '' or
                    len(abstract.split()) <= 50 or
                    not check_string_for_english(abstract)):
                mass_error += 1
                continue

            """
                Check for too many single characters
            """
            alist = abstract.split(" ")
            n_token = len(alist)
            n_ones = 0
            for token in alist:
                if len(token) == 1 and token.isalpha():
                    n_ones += 1
            result = n_ones / n_token
            if result > 0.15:
                prune_error += 1
                continue

            if abstract_id in abstract_id_list:
                continue

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

                author_dic[author_id] = author_name

                author_ids.append(str(author_id))

            abstract_id_list.append(abstract_id)
            author_list.append(",".join(author_ids))
            journal_list.append(journal)
            venue_list.append(venue)
            year_list.append(year)
            entity_list.append(",".join(elist))

            doc = nlp(abstract)

            doc.to_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))

    nlp.vocab.to_disk(os.path.join(path_to_annotations, "spacy.vocab"))
    infoDF = pd.DataFrame({'year': year_list,
                           'entities': entity_list,
                           'journal': journal_list,
                           'venue': venue_list,
                           'authors': author_list},
                          index=abstract_id_list)
    infoDF.to_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))
    with open(os.path.join(path_to_annotations, "author.dic"), "wb") as handle:
        pickle.dump(author_dic, handle)

print()
print(f"Number of Publications: {count}")
print(f"Vocab Size: {len(nlp.vocab)}")

nlp.vocab.to_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.DataFrame({'year': year_list,
                       'entities': entity_list,
                       'journal': journal_list,
                       'venue': venue_list,
                       'authors': author_list},
                      index=abstract_id_list)
infoDF.to_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))
with open(os.path.join(path_to_annotations, "author.dic"), "wb") as handle:
    pickle.dump(author_dic, handle)