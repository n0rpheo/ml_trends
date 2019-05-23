import os
import scipy.sparse
import numpy as np
import gensim
import pandas as pd
import pickle
from spacy.vocab import Vocab
from spacy.tokens import Doc

from src.utils.LoopTimer import LoopTimer
from MLGenome.MLARes.occ.occ import occTopMention

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

path_to_tm = os.path.join(path_to_db, "topic_modeling", nlp_model)
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)
path_to_features = os.path.join(path_to_tm, "features")
path_to_dictionaries = os.path.join(path_to_db, "dictionaries", nlp_model)
path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)

if not os.path.isdir(path_to_tm):
    print(f"Create Directory {path_to_tm}")
    os.mkdir(path_to_tm)
    print(f"Create Directory {path_to_features}")
    os.mkdir(path_to_features)

"""
 #
 # Infos
 #
"""
training_size = 25000
is_merged = True
otm_name = "occ_vec_ratio_p2.pickle"

merged_string = "merged_" if is_merged else ""
feature_file_name = f"features_{merged_string}word_vec.npz"  # output
info_name = f"{merged_string}word_vec.info"  # output

dic_name = f"full_{merged_string}word.dict"

otm_path = os.path.join(path_to_mlgenome, otm_name)

otm = occTopMention(path=otm_path)
"""
 #
 #
 #
"""

dictionary = gensim.corpora.Dictionary.load(os.path.join(path_to_dictionaries, dic_name))
print(f"Length of Dic: {len(dictionary)}")

num_samples = -1

row = []
col = []
data = []

print("Loading Vocab...")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

lt = LoopTimer(update_after=5, avg_length=1000, target=min(training_size, len(infoDF)))
for idx, (abstract_id, df_row) in enumerate(infoDF.iterrows()):

    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    if is_merged:
        for ent in doc.ents:
            ent.merge(ent.root.tag_, ent.orth_, ent.label_)

        tokens = []
        for token in doc:
            append_tokens = otm.get(token.text.lower())

            if 'svm' in append_tokens:
                print(append_tokens)
                exit()
            for append_token in append_tokens:
                tokens.append(append_token)
    else:
        tokens = [token.text.lower() for token in doc]

    if num_samples != -1 and idx == num_samples:
        break

    bow = dictionary.doc2bow(tokens)

    for entry in bow:
        row.append(idx)
        col.append(entry[0])
        data.append(entry[1])
    breaker = lt.update("Build Features")
    if breaker >= training_size:
        break


m = idx + 1
n = len(dictionary)

row = np.array(row)
col = np.array(col)
data = np.array(data)

feature_vector = scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n))

scipy.sparse.save_npz(os.path.join(path_to_features, feature_file_name), feature_vector)

info = dict()
info['dic_path'] = os.path.join(path_to_dictionaries, dic_name)
info['feature_path'] = os.path.join(path_to_features, feature_file_name)
info['is_merged'] = is_merged
if is_merged:
    info['otm_path'] = otm_path

with open(os.path.join(path_to_tm, info_name), "wb") as handle:
    pickle.dump(info, handle)