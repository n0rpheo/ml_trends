import os
from spacy.vocab import Vocab
from spacy.tokens import Doc
import pandas as pd
import pickle

from src.utils.LoopTimer import LoopTimer


acronym_excludes = ['ii', 'iii']
definition_excludes = [',', '"', ')', '(', ';', '?', '!']


def token_is_acronym_candidate(candidate):
    c_string = candidate.orth_

    if len(c_string) < 2:
        return False
    if c_string in acronym_excludes:
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


def find_definition_candidate(span, acronym_cand):
    candidate_id = acronym_cand.i - span.start

    """
        Case: definition '(' acronym ')'
    """

    if (0 < candidate_id < len(span) - 1 and
            span[candidate_id-1].orth_ == '('
            and span[candidate_id+1].orth_ == ')'):
        max_len = min(len(acronym_cand.orth_) + 5, len(acronym_cand.orth_)*2)
        before_span = span[max(0, candidate_id-max_len-1):candidate_id-1]

        definition_candidate = find_best_definition(acronym_cand.orth_, before_span.text)

        if definition_candidate is not None and all([forbidden_character not in definition_candidate for forbidden_character in definition_excludes]):
            return definition_candidate

    """
        Non Parentheses Matching
    

    max_len = min(len(acronym_cand.orth_) + 5, len(acronym_cand.orth_) * 2)
    before_span = span[max(0, candidate_id - max_len - 1):candidate_id]
    after_span = span[(candidate_id + 1):min(candidate_id + max_len + 1, len(span))]

    before_candidate = find_best_definition(acronym_cand.orth_, before_span.text)
    after_candidate = find_best_definition(acronym_cand.orth_, after_span.text)

    if before_candidate is not None and after_candidate is not None:
        return None

    if before_candidate is not None:
        print(f"{acronym_cand.orth_}: {before_candidate}")
    if after_candidate is not None:
        print(f"{acronym_cand.orth_}: {after_candidate}")

    """
    return None


path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)

if not os.path.isdir(path_to_mlgenome):
    print(f"Create Directory {path_to_mlgenome}")
    os.mkdir(path_to_mlgenome)

path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)

vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

acronym_dictionary = dict()

lt = LoopTimer(update_after=1, avg_length=1000, target=len(infoDF))
for abstract_id, row in infoDF.iterrows():

    file_path = os.path.join(path_to_annotations, f"{abstract_id}.spacy")
    doc = Doc(vocab).from_disk(file_path)

    for sentence in doc.sents:
        for token in sentence:
            if token_is_acronym_candidate(token):
                definition = find_definition_candidate(sentence, token)
                if definition is not None:
                    acronym = token.orth_.lower()
                    if acronym not in acronym_dictionary:
                        acronym_dictionary[acronym] = set()
                    acronym_dictionary[acronym].add(definition.lower())
    lt.update(f"Find Acronym Definitions - {len(acronym_dictionary)}")


with open(os.path.join(path_to_mlgenome, "acronyms.pickle"), "wb") as handle:
    pickle.dump(acronym_dictionary, handle)