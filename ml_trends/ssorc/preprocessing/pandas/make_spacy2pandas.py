import os
import pandas as pd
import spacy
from spacy.tokens import Doc

from src.utils.LoopTimer import LoopTimer


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
pandas_path = os.path.join(path_to_db, "pandas")

#nlp = spacy.load('en_core_web_sm')
#vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "spacy.vocab"))

nlp_model = "en_core_web_sm_nertrained_v3"
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)

nlp = spacy.load(os.path.join(path_to_db, "models", nlp_model))
#component = nlp.create_pipe("merge_entities")
#nlp.add_pipe(component)
vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "ner_spacy.vocab"))

infoDF = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'ner_info_db.pandas'))

abstract_id_list = list()
word_list = list()
lemma_list = list()
coarse_pos_list = list()
fine_pos_list = list()
ent_type_list = list()

merged_word_list = list()
merged_ent_type_list = list()

targ = len(infoDF)
not_found = 0
lt = LoopTimer(update_after=100, avg_length=10000, target=targ)
for abstract_id, row in infoDF.iterrows():
    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    if not os.path.isfile(file_path):
        not_found += 1
        lt.update(f"Create Pandas - {len(abstract_id_list)}")
        continue
    doc = Doc(vocab).from_disk(file_path)

    abstract_id_list.append(abstract_id)

    word_list.append("\t\t".join(["\t".join([token.text for token in sentence if token_conditions(token)])
                                  for sentence in doc.sents]))
    lemma_list.append("\t\t".join(["\t".join([token.lemma_ for token in sentence if token_conditions(token)])
                                   for sentence in doc.sents]))
    coarse_pos_list.append("\t\t".join(["\t".join([token.pos_ for token in sentence if token_conditions(token)])
                                        for sentence in doc.sents]))
    fine_pos_list.append("\t\t".join(["\t".join([token.tag_ for token in sentence if token_conditions(token)])
                                      for sentence in doc.sents]))

    ent_type_list.append("\t\t".join(["\t".join([token.ent_type_ if len(token.ent_type_) > 0 else "."
                                                 for token in sentence if token_conditions(token)])
                                      for sentence in doc.sents]))

    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.orth_, ent.label_)

    merged_word_list.append("\t\t".join(["\t".join([token.text for token in sentence if token_conditions(token)])
                                         for sentence in doc.sents]))
    merged_ent_type_list.append("\t\t".join(["\t".join([token.ent_type_ if len(token.ent_type_) > 0 else "."
                                                        for token in sentence if token_conditions(token)])
                                             for sentence in doc.sents]))
    lt.update(f"Create Pandas - {len(abstract_id_list)}")

print()
print(f"{not_found} nicht gefunden.")

wordDF = pd.DataFrame(word_list, index=abstract_id_list, columns=["word"])
lemmaDF = pd.DataFrame(lemma_list, index=abstract_id_list, columns=["lemma"])
fineposDF = pd.DataFrame(fine_pos_list, index=abstract_id_list, columns=["pos"])
coarseposDF = pd.DataFrame(coarse_pos_list, index=abstract_id_list, columns=["coarse_pos"])
ent_typeDF = pd.DataFrame(ent_type_list, index=abstract_id_list, columns=["ent_type"])
merged_ent_typeDF = pd.DataFrame(merged_ent_type_list, index=abstract_id_list, columns=["merged_ent_type"])
merged_wordDF = pd.DataFrame(merged_word_list, index=abstract_id_list, columns=["merged_word"])


wordDF.to_pickle(os.path.join(pandas_path, 'aiml_ner_word.pandas'))
lemmaDF.to_pickle(os.path.join(pandas_path, 'aiml_ner_lemma.pandas'))
fineposDF.to_pickle(os.path.join(pandas_path, 'aiml_ner_finepos.pandas'))
coarseposDF.to_pickle(os.path.join(pandas_path, 'aiml_ner_coarsepos.pandas'))
ent_typeDF.to_pickle(os.path.join(pandas_path, 'aiml_ner_ent_type.pandas'))
merged_ent_typeDF.to_pickle(os.path.join(pandas_path, 'aiml_ner_merged_ent_type.pandas'))
merged_wordDF.to_pickle(os.path.join(pandas_path, 'aiml_ner_merged_word.pandas'))