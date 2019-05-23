import os
from spacy.vocab import Vocab
from spacy.tokens import Doc
import pandas as pd
import pickle

from src.utils.LoopTimer import LoopTimer

definition_excludes = [',', '"', ')', '(', ';', '?', '!']


def is_acronym_candidate(c_string):
    if len(c_string) < 2 or len(c_string) > 10:
        return False

    is_upper = [character == character.upper() for character in c_string if character.isalpha()]
    first_character_valid = (c_string[0] == c_string[0].lower() and c_string[0].isalpha()) or c_string[0].isdigit()
    last_character_valid = (c_string[-1] == c_string[-1].lower() and c_string[-1].isalpha()) or c_string[-1].isdigit()
    if not (any(is_upper) or
            (first_character_valid or
             last_character_valid)):
        return False
    return True


def find_best_definition(acronym_, definition_):
    aidx = len(acronym_) - 1  # Set aidx at the end of the short form | The index on the short form
    didx = len(definition_) - 1  # Set didx at the end of the long form | The index on the long form

    while aidx >= 0:
        # from end to start
        # Store the next character to match. Ignore case
        currChar = acronym_[aidx].lower()
        # ignore non alphanumeric characters
        if not (currChar.isalpha() or currChar.isdigit()):
            aidx -= 1
            continue
        # Decrease didx while current character in the long form
        # does not match the current character in the short form.
        # If the current character is the first character in the
        # short form, decrement didx until a matching character
        # is found at the beginning of a word in the long form.
        while (((didx >= 0) and
                (definition_[didx].lower() != currChar)) or
               ((aidx == 0) and (didx > 0) and
                (definition_[didx - 1].isalpha() or definition_[didx - 1].isdigit()))):
            didx -= 1

        # If no match was found in the long form for the current
        # character, return null (no match).
        if didx < 0:
            return None

        # A match was found for the current character. Move to the
        # next character in the long form.
        didx -= 1

        aidx -= 1
    # Find the beginning of the first word (in case the first
    # character matches the beginning of a hyphenated word).

    didx = definition_.rfind(" ", 0, didx + 1) + 1
    # Return the best long form, the substring of the original
    # long form, starting from didx up to the end of the original
    # long form.
    return definition_[didx:]


def find_best_definition_span(acronym_span, definition_span):
    ac_string = acronym_span.text
    aidx = len(ac_string) - 1  # Set aidx at the end of the short form | The index on the short form
    didx = len(definition_span) - 1  # Set didx at the end of the long form | The index on the long form
    d_string = definition_span[didx].orth_.lower()
    dtidx = len(d_string) - 1

    while True:
        # from end to start
        # Store the next character to match. Ignore case
        currChar = ac_string[aidx].lower()
        # ignore non alphanumeric characters
        if not (currChar.isalpha() or currChar.isdigit()):
            if aidx == 0:
                break
            else:
                aidx -= 1
            continue
        # Decrease didx while current character in the long form
        # does not match the current character in the short form.
        # If the current character is the first character in the
        # short form, decrement didx until a matching character
        # is found at the beginning of a word in the long form.
        while ((d_string[dtidx] != currChar) or
               ((aidx == 0) and (dtidx > 0) and
                (d_string[dtidx - 1].isalpha() or d_string[dtidx - 1].isdigit()))):
            # If no match was found in the long form for the current
            # character, return null (no match).
            if dtidx == 0 and didx == 0:
                return None
            elif dtidx == 0:
                didx -= 1
                d_string = definition_span[didx].orth_.lower()
                dtidx = len(d_string) - 1
            else:
                dtidx -= 1

        if aidx == 0:
            break
        else:
            aidx -= 1

        if dtidx == 0 and didx == 0:
            return None
        elif dtidx == 0:
            didx -= 1
            d_string = definition_span[didx].orth_.lower()
            dtidx = len(d_string) - 1
        else:
            dtidx -= 1

    # Find the beginning of the first word (in case the first
    # character matches the beginning of a hyphenated word).

    while didx >= 0 and definition_span[didx - 1].whitespace_ == '':
        didx -= 1

    if didx < 0:
        return None

    # Check if acronym is in definition
    if ac_string in definition_span[didx:].text:
        return None

    # Return the best long form, the substring of the original
    # long form, starting from didx up to the end of the original
    # long form.
    return definition_span[didx:]


def find_definition_candidate(sent_span, acronym_span):

    # id of the first token of the acronym_candidate
    candidate_id = acronym_span[0].i - sent_span.start
    acronym_candidate = acronym_span.text

    if not is_acronym_candidate(acronym_candidate):
        return None

    # Maximum Length of Tokens a candidate is allowed to have
    max_len = min(len(acronym_candidate) + 5, len(acronym_candidate) * 2)

    # if candidate is in parenthesis
    # calculate from where to where the definition-span should be
    if (0 < candidate_id < len(sent_span) - 1 and
            sent_span[candidate_id - 1].orth_ == '('
            and sent_span[candidate_id + len(acronym_span)].orth_ == ')'):
        before_span = sent_span[max(0, candidate_id - max_len - 1):candidate_id - 1]
    else:
        before_span = sent_span[max(0, candidate_id - max_len):candidate_id]

    # definition_candidate = find_best_definition(acronym_candidate, before_span.text)
    definition_candidate = find_best_definition_span(acronym_span, before_span)

    if definition_candidate is not None and all(
            [forbidden_character not in definition_candidate.text.lower()
             for forbidden_character in definition_excludes]):
        return definition_candidate

    return None


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)

if not os.path.isdir(path_to_mlgenome):
    print(f"Create Directory {path_to_mlgenome}")
    os.mkdir(path_to_mlgenome)

path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)
path_to_pandas = os.path.join(path_to_db, "pandas", nlp_model)

print("Loading Vocab...")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

acronym_dictionary = dict()

lt = LoopTimer(update_after=1, avg_length=1000, target=len(infoDF))
for abstract_id, row in infoDF.iterrows():

    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    for sentence in doc.sents:
        for ent in sentence.ents:
            definition_span = find_definition_candidate(sentence, ent)
            if definition_span is not None:
                acronym_string = ent.text.lower()
                acronym_orth = [token.orth_ for token in ent]

                d_string = definition_span.text.lower()
                d_orth = [token.orth_.lower() for token in definition_span]
                d_orth_with_ws = [token.text_with_ws.lower() for token in definition_span]
                d_lemma = [token.lemma_.lower() for token in definition_span]
                d_lemma_with_ws = [f"{token.lemma_.lower()}{token.whitespace_}" for token in definition_span]

                def_dic = {"string": d_string,
                           "orth": d_orth,
                           "orth_with_ws": d_orth_with_ws,
                           'lemma': d_lemma,
                           'lemma_with_ws': d_lemma_with_ws}

                if acronym_string not in acronym_dictionary:
                    acronym_dictionary[acronym_string] = list()
                acronym_dictionary[acronym_string].append(def_dic)
    lt.update(f"Find Acronym Definitions - {len(acronym_dictionary)}")
    if len(acronym_dictionary) > 2000:
        break

print()
ad = dict()
for acronym in acronym_dictionary:
    definitions = [dd['string'] for dd in acronym_dictionary[acronym]]
    n_defs = len(definitions)
    n_unique_defs = len(set(definitions))

    def_avg = n_defs / n_unique_defs
    def_set = set()
    def_list = list()

    for def_dic in acronym_dictionary[acronym]:
        definition = def_dic["string"]

        n_def = definitions.count(definition)
        if n_def >= def_avg:
            if definition not in def_set:
                def_set.add(definition)
                def_list.append(def_dic)

    if len(def_set) > 0:
        ad[acronym] = def_list
        print(f"{acronym}: {def_set}")


with open(os.path.join(path_to_mlgenome, "ml_acronyms.pickle"), "wb") as handle:
    pickle.dump(ad, handle)