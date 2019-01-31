import os
import scipy.sparse
import numpy as np
import gensim
import pandas as pd

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
feature_file_name = 'tm_features.npz'
dic_temp_file_name = "tm_dictionary.dict"

print("Loading Panda DB")
tokenDF = pd.read_pickle(os.path.join(path_to_db, "pandas", "ml_lemma.pandas"))

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(os.path.join(path_to_db, "dictionaries", "pruned_lemma_lower_pd.dict"))
print(f"Length of Dic: {len(dictionary)}")

num_samples = 100000

row = []
col = []
data = []

lt = LoopTimer(update_after=100, avg_length=2000, target=len(tokenDF))
for idx, (abstract_id, df_row) in enumerate(tokenDF.iterrows()):
    tokens = df_row['lemma'].split()

    if num_samples != -1 and idx == num_samples:
        break

    bow = dictionary.doc2bow(tokens)

    for entry in bow:
        row.append(idx)
        col.append(entry[0])
        data.append(entry[1])
    lt.update("Build Features")

m = idx + 1
n = len(dictionary)

row = np.array(row)
col = np.array(col)
data = np.array(data)

feature_vector = scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n))

scipy.sparse.save_npz(os.path.join(path_to_db, "features", feature_file_name), feature_vector)
dictionary.save(os.path.join(path_to_db, "dictionaries", dic_temp_file_name))
