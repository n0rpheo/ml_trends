import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

import spacy
from spacy.util import minibatch, compounding

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/Results/fig"

# training data
with open(os.path.join(path_to_db, "NER", "spacy_ner_mlalgo_traindata_th1.pickle"), "rb") as handle:
    TRAIN_DATA = pickle.load(handle)

n_train_data = len(TRAIN_DATA)


model = "en_core_web_sm"
new_model_name = "en_core_web_sm_mlalgo_test"
input_dir = os.path.join(path_to_db, "models", "en_core_web_sm_nertrained_v3")
output_dir = os.path.join(path_to_db, "models", "en_core_web_sm_nertrained_v3")
n_iter = 1000
random.seed(0)

nlp = spacy.load(input_dir)  # load existing spaCy model
#nlp = spacy.load(model)
print(f"Loaded model new")

if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe("ner")

ner.add_label("MLALGO")

print(f"Trainsize: {n_train_data}")

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

y_list = list()
x_list = list()

y_top_list = list()
x_top_list = list()

disabled = nlp.disable_pipes(*other_pipes)  # only train NER

optimizer = nlp.resume_training()
lt = LoopTimer(update_after=50, avg_length=5000, target=n_iter*n_train_data)
sizes = compounding(4.0, 32.0, 1.005)
for itn in range(n_iter):
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=sizes)
    n_batches = 0
    batch_count = 0
    for batch in batches:
        n_batch = len(batch)
        texts, annotations = zip(*batch)
        nlp.update(
            texts,  # batch of texts
            annotations,  # batch of annotations
            drop=0.25,  # dropout - make it harder to memorise data
            sgd=optimizer,
            losses=losses,
        )

        y_list.append(losses['ner'])
        batch_count += 1

        update_text = f"Train iteration {itn} - Losses: {round(losses['ner'], 2)}"
        lt.update(update_text=update_text, update_len=n_batch)

    y_top_list.append(losses['ner'])
    x_top_list.append(itn)

    step_size = 1 / batch_count
    for i in range(batch_count):
        x_list.append(itn+(i+1)*step_size)

    disabled.restore()
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    disabled = nlp.disable_pipes(*other_pipes)

    if True:
        # X = np.array(x_list)
        # Y = np.array(y_list)

        X = np.array(x_top_list)
        Y = np.array(y_top_list)

        plt.plot(X, Y)
        plt.savefig(os.path.join(path_to_fig_save, "ner_training_los.png"))


X = np.array(x_list)
Y = np.array(y_list)

plt.plot(X, Y)
plt.show()

exit()


# save model to output directory
disabled.restore()
nlp.meta["name"] = new_model_name
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
for text, _ in TRAIN_DATA[:20]:
    doc = nlp2(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob, t.pos_, t.tag_, t.lemma_) for t in doc])