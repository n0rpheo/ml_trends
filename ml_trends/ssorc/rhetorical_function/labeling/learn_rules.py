import operator
import os
import json
import pandas as pd

from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.matcher import Matcher

import src.modules.pattern_matching as pm
from src.utils.LoopTimer import LoopTimer


"""
    Infos
"""
cat_set = ['Background',
           # 'Design',
           'Method',
           # 'Result',
           'Objective']

iterations = 3
select_top_n = 2
phrase_boundaries = (3, 10)  # Length of a Phrase

seed_rules_filename = "seed_rules.json"
learned_rules_filename = "learned_rules.json"

"""
====================
"""

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"
path_to_annotations = os.path.join(path_to_db, "annotations_version", nlp_model)
path_to_patterns = os.path.join(path_to_db, "pattern_matching")
path_to_learned_patterns = os.path.join(path_to_patterns, nlp_model)

if not os.path.isdir(path_to_learned_patterns):
    print(f"Create Directory {path_to_learned_patterns}")
    os.mkdir(path_to_learned_patterns)


# Loading Rules
rules = pm.load_rules(os.path.join(path_to_patterns, seed_rules_filename))

print("Loading Vocab...")
vocab = Vocab().from_disk(os.path.join(path_to_annotations, "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_annotations, 'info_db.pandas'))

db_size = len(infoDF)


def find_phrases_by_rule(doc_, rules_, phrase_boundaries_):
    min_len = phrase_boundaries_[0]
    max_len = phrase_boundaries_[1]
    phrases_ = list()

    for rule_ in rules_:
        trigger_word = rule_[0].lower()
        dependency = rule_[1]

        for token in doc_:
            if token.lower_ == trigger_word:
                for child in token.children:
                    if child.dep_ == dependency:
                        p_list = [{"LOWER": subtoken.orth_.lower()} for subtoken in child.subtree]
                        if min_len <= len(p_list) < max_len-1:
                            phrases_.append(p_list)
    return phrases_


def find_rule_by_phrase(doc_, matcher_, rule_by_phrase_counter_, phrase_counter_, ignore_rules_):
    matches = matcher_(doc_)

    for match_id, start, end in matches:
        if match_id not in phrase_counter_:
            phrase_counter_[match_id] = 0
        phrase_counter_[match_id] += 1

        phrase_tokens = [t for t in doc_[start:end]]
        head_set = set()
        lasthead_set = set()
        head = None
        last_head = None
        for phrase_token in phrase_tokens:
            head = phrase_token.head
            last_head = phrase_token
            while head in phrase_tokens:
                last_head = head
                head = head.head
                check_dep = last_head.dep_
                if check_dep == 'ROOT':
                    break

            head_set.add(head)
            lasthead_set.add(last_head)

        if not (len(head_set) == 1 and len(lasthead_set) == 1):
            # phrase is not within a unique subtree
            continue

        if head.is_stop:
            continue

        trigger_rule_word = head.lower_
        trigger_rule_dependency = last_head.dep_

        if trigger_rule_dependency == 'ROOT':
            # There is no trigger_word
            continue

        rule_ = (trigger_rule_word, trigger_rule_dependency)

        if rule_ in ignore_rules_:
            continue

        if rule_ not in rule_by_phrase_counter_:
            rule_by_phrase_counter_[rule_] = dict()
            rule_by_phrase_counter_[rule_][match_id] = 1
        else:
            if match_id not in rule_by_phrase_counter_[rule_]:
                rule_by_phrase_counter_[rule_][match_id] = 0
                rule_by_phrase_counter_[rule_][match_id] += 1


for category in [j for j in rules if j in cat_set]:
    string = f"| Learning {category} |"
    lines = "".join(["-" for i in range(len(string))])
    print(lines)
    print(string)
    print(lines)

    learn_rules = rules[category]
    for it in range(0, iterations):
        """
        =============================
            FIND PHRASES BY RULES
        =============================
        """
        patterns = list()
        lt = LoopTimer(update_after=500, avg_length=10000, target=db_size)
        for abstract_id, row in infoDF.iterrows():
            doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))
            patterns.extend(find_phrases_by_rule(doc, learn_rules, phrase_boundaries))
            n = lt.update(f"Find Phrases - {len(patterns)}")

        print()

        """
        =============================
                BUILD MATCHER
        =============================
        """
        matcher = Matcher(vocab)
        lt = LoopTimer(update_after=10000, avg_length=10000, target=len(patterns))
        for p_id, pattern in enumerate(patterns):
            matcher.add(f"pattern_{p_id}", None, pattern)
            lt.update("Build Matcher")
        print()
        """
        =============================
            FIND RULES BY PHRASES
        =============================
        """
        rule_by_phrase_counter = dict()
        phrase_counter = dict()
        lt = LoopTimer(update_after=100, avg_length=10000, target=db_size)
        for abstract_id, row in infoDF.iterrows():
            doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))
            find_rule_by_phrase(doc, matcher, rule_by_phrase_counter, phrase_counter, rules[category])
            n = lt.update(f"Find Rules - {len(rule_by_phrase_counter)}")

        rule_ranking = dict()
        for rule in rule_by_phrase_counter:
            rule_ranking[rule] = 0
            for p_id in rule_by_phrase_counter[rule]:
                rule_ranking[rule] += rule_by_phrase_counter[rule][p_id] / phrase_counter[p_id]

        sorted_rule_ranking = sorted(rule_ranking.items(), key=operator.itemgetter(1), reverse=True)

        print()
        for item in sorted_rule_ranking[:5]:
            t_word = item[0][0]
            dep = item[0][1]
            print(f"{str(item[1])[:5]} : {t_word} -> {dep}")

        top_rules = set()

        for item in sorted_rule_ranking[:select_top_n]:
            top_rules.add(item[0])

        if len(top_rules) == 0:
            break

        rules[category].update(top_rules)

        learn_rules = top_rules
print()
print()
for category in [j for j in rules if j in cat_set]:
    string = f"| {category} |"
    lines = "".join(["-" for i in range(len(string))])
    print(lines)
    print(string)
    print(lines)
    for rule in rules[category]:
        t_word = rule[0]
        dep = rule[1]
        print(f"{t_word} -> {dep}")


with open(os.path.join(path_to_learned_patterns, learned_rules_filename), 'w') as handle:
    for category in [j for j in rules if j in cat_set]:
        for rule in rules[category]:
            data = dict()

            data['category'] = category
            data['trigger_word'] = rule[0]
            data['dependency'] = rule[1]

            json_string = json.JSONEncoder().encode(data)

            handle.write(json_string + '\n')
