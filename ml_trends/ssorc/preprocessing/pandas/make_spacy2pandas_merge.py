import os
import pickle
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

import pandas as pd
from src.utils.LoopTimer import LoopTimer


def is_sublist(ls1, ls2):
    for word_b_idx in range(len(ls2) - len(ls1) + 1):
        for word_a_idx in range(len(ls1)):
            word_a = ls1[word_a_idx]
            word_b = ls2[word_b_idx + word_a_idx]

            if word_a != word_b:
                break
        else:
            return True
    return False


def token_conditions(token_):
    if token_.ent_iob == 3 or token_.ent_iob == 1:
        return True
    if token_.is_punct:
        return False
    if token_.is_stop:
        return False
    if len(token_.orth_) < 3:
        return False

    return True

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations_ner")
pandas_path = os.path.join(path_to_db, "pandas")
path_to_ner = os.path.join(path_to_db, "NER")

nlp = spacy.load(os.path.join(path_to_db, "models", "en_core_web_sm_nertrained"))
vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "spacy.vocab"))
matcher = Matcher(vocab)

with open(os.path.join(path_to_ner, "ml_algos.txt"), "r") as handle:
    ml_algos = set()
    ml_algos_list = list()
    for line in handle:
        algo = line.strip().lower()
        if algo not in ml_algos:
            ml_algos.add(algo)
            ml_algos_list.append(algo.split(" "))

for i in range(len(ml_algos_list)):
    for j in range(len(ml_algos_list)):
        if i != j:
            algo1 = ml_algos_list[i]
            algo2 = ml_algos_list[j]

            if is_sublist(algo1, algo2):
                algo1 = ' '.join(algo1)
                algo2 = ' '.join(algo2)
                ml_algos.discard(algo2)


for ml_algo in ml_algos:
    doc = nlp(ml_algo)
    pattern = [{"LOWER": token.lower_} for token in doc]
    pattern_name = "".join([entity["LOWER"] for entity in pattern])
    matcher.add(pattern_name, None, pattern)


infoDF = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'ner_info_db.pandas'))

abstract_id_list = list()
word_list = list()

targ = len(infoDF)
lt = LoopTimer(update_after=10, avg_length=10000, target=targ)
merged = 0
for abstract_id, row in infoDF.iterrows():
    doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))

    matches = matcher(doc)

    skip_matches = set()
    n_matches = len(matches)
    for i in range(n_matches):
        match = matches[i]

        match_id = match[0]
        start = match[1]
        end = match[2]

        for j in range(i + 1, n_matches):
            c_match = matches[j]
            c_match_id = c_match[0]
            c_start = c_match[1]
            c_end = c_match[2]

            if ((c_start <= end and start <= c_end) or
                    (start <= c_end and c_start <= end)):
                ent_size = end - start
                c_ent_size = c_end - c_start

                if ent_size < c_ent_size:
                    skip_matches.add(i)
                elif c_ent_size < ent_size or start < c_start:
                    skip_matches.add(j)
                else:
                    skip_matches.add(i)

    with doc.retokenize() as retokenizer:
        for i, match in enumerate(matches):
            match_id = match[0]
            if i in skip_matches:
                continue
            start = match[1]
            end = match[2]
            merged += 1
            retokenizer.merge(doc[start:end])

    abstract_id_list.append(abstract_id)

    token_string = "\t\t".join(["\t".join([token.text.lower() for token in sentence if token_conditions(token)])
                                for sentence in doc.sents])

    word_list.append(token_string)
    lt.update(f"Merging - {merged}")

wordDF = pd.DataFrame(word_list, index=abstract_id_list, columns=["word_lower_merged"])
wordDF.to_pickle(os.path.join(pandas_path, 'aiml_word_lower_merged.pandas'))





