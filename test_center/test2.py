import os
import spacy
from spacy.matcher import Matcher
from spacy.vocab import Vocab


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)

nlp = spacy.load(os.path.join(path_to_db, "models", nlp_model))
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))

#nlp = spacy.load('en_core_web_lg')


matcher = Matcher(vocab)
term = nlp(u"Barack Obama")
# Only run nlp.make_doc to speed things up
pattern = [{"ORTH": token.orth_} for token in term]
matcher.add("BO", None, pattern)

doc = nlp(u"German Chancellor Angela Merkel and US President Barack Obama "
          u"converse in the Oval Office inside the White House in Washington, D.C.")
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print(span.text)