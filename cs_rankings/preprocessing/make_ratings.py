import os
import pickle
import pandas as pd

from src.modules.topic_modeling import TopicModelingLDA
from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, 'dictionaries')
path_to_models = os.path.join(path_to_db, 'models')

lemmaDF = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'ml_lemma.pandas'))

with open(os.path.join(path_to_db, 'csrankings', 'abs2auth.dict'), 'rb') as a2afile:
    abstract_to_author_db = pickle.load(a2afile)

tm_model = TopicModelingLDA(dic_path=os.path.join(path_to_dictionaries, "pruned_lemma_lower_pd.dict"),
                            model_path=os.path.join(path_to_models, "tm_lda_500topics.pickle"))

summe = None

author_ratings = dict()

lt = LoopTimer(update_after=500, avg_length=5000, target=len(lemmaDF))
for count, (abstract_id, row) in enumerate(lemmaDF.iterrows()):
    tokens = row['lemma'].split()

    if abstract_id not in abstract_to_author_db:
        continue

    author_ids = [int(id_) for id_ in abstract_to_author_db[abstract_id].split(',')]
    num_authors = len(author_ids)

    topic_dist = tm_model.get_topic_dist(tokens) / num_authors

    for author_id in author_ids:
        if author_id not in author_ratings:
            author_ratings[author_id] = topic_dist
        else:
            author_ratings[author_id] += topic_dist

    lt.update("Ranking")

author_ids = list()
ratings = list()
for author_id in author_ratings:
    author_ids.append(author_id)
    ratings.append(author_ratings[author_id])

df = pd.DataFrame(ratings, index=author_ids)

df.to_pickle(os.path.join(path_to_db, "csrankings", 'ratings.pd'))