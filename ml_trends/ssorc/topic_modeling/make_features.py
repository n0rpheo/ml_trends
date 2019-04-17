import os
import scipy.sparse
import numpy as np
import gensim
import pandas as pd
import pickle

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_tm = os.path.join(path_to_db, "topic_modeling")

"""
 #
 # Infos
 #
"""
feature_file_name = 'aiml_tm_features_word_lower_merged.npz'  # output
info_name = "word_lower_merged.info"  # output

dic_name = "aiml_full_ner_word_lower_merged.dict"
pandas_name = "aiml_word_lower_merged.pandas"

col_name = "word_lower_merged"
"""
 #
 #
 #
"""

tokenDF = pd.read_pickle(os.path.join(path_to_db, "pandas", pandas_name))
dictionary = gensim.corpora.Dictionary.load(os.path.join(path_to_db, "dictionaries", dic_name))
print(f"Length of Dic: {len(dictionary)}")

num_samples = -1

row = []
col = []
data = []

lt = LoopTimer(update_after=100, avg_length=2000, target=len(tokenDF))
for idx, (abstract_id, df_row) in enumerate(tokenDF.iterrows()):
    tokens = df_row[col_name].replace("\t\t", "\t").split("\t")

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

scipy.sparse.save_npz(os.path.join(path_to_tm, "features", feature_file_name), feature_vector)

info = dict()
info['dic_path'] = os.path.join(path_to_db, "dictionaries", dic_name)
info['feature_path'] = os.path.join(path_to_tm, "features", feature_file_name)
info['pandas_path'] = os.path.join(path_to_db, "pandas", pandas_name)

with open(os.path.join(path_to_db, "topic_modeling", info_name), "wb") as handle:
    pickle.dump(info, handle)