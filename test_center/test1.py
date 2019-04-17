import os
import pandas as pd
import spacy
from spacy.tokens import Doc

from src.utils.LoopTimer import LoopTimer


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations_ner")
pandas_path = os.path.join(path_to_db, "pandas")

nlp = spacy.load(os.path.join(path_to_db, "models", "en_core_web_sm_nertrained"))
vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "ner_spacy.vocab"))

infoDF = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'ner_info_db.pandas'))

targ = len(infoDF)
lt = LoopTimer(update_after=100, avg_length=10000, target=targ)

year_df = dict()
ml_algo = "neural network"
years = set()
for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)
    abstract = doc.text.lower()
    year = row['year']

    if ml_algo in abstract:
        if year not in year_df:
            years.add(year)
            year_df[year] = 0
        year_df[year] += 1
    breaker = lt.update(f"Create MLalgo")
    if breaker > 100:
        break

year_sort = sorted(list(years), reverse=False)
for year in year_sort:
    print(f"{year}: {year_df[year]}")