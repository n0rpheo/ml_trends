import operator
import os
import json
import pandas as pd
import spacy
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher

import src.modules.pattern_matching as pm
from src.utils.LoopTimer import LoopTimer


def findPhrasesByRule(doc_, rules_, phrase_boundaries_):
    min_len = phrase_boundaries_[0]
    max_len = phrase_boundaries_[1]
    phrases_ = set()

    for rule_ in rules_:
        trigger_word = rule_[0].lower()
        dependency = rule_[1]

        for token in doc_:
            if token.lower_ == trigger_word:
                for child in token.children:
                    if child.dep_ == dependency:
                        p_list = [subtoken.orth_ for subtoken in child.subtree]
                        if min_len <= len(p_list) < max_len-1:
                            phrase = " ".join(p_list)
                            phrases_.add(phrase)
    return phrases_


def findRuleByPhrase(doc_, matcher_, rule_by_phrase_counter_, phrase_counter_, ignore_rules_):
    matches = matcher_(doc_)

    for match_id, start, end in matches:

        if match_id not in phrase_counter_:
            phrase_counter_[match_id] = 0
        phrase_counter_[match_id] += 1

        phrase_tokens = [t for t in doc_[start:end]]
        head_set = set()
        lasthead_set = set()
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


cat_set = ['Background',
           #'Design',
           'Method',
           #'Result',
           'Objective']
mod_name = "dep_tree_big"
seed_rules_filename = "seed_rules.json"
learned_rules_filename = "aiml_leaned_rules.json"
iterations = 3
select_top_n = 2

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_annotations = os.path.join(path_to_db, "annotations")
rule_file_path = os.path.join(path_to_db, "pattern_matching", seed_rules_filename)

# Loading Rules
rule_state = "text"
rules = pm.load_rules(rule_file_path)
nlp = spacy.load('en_core_web_sm')
vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "spacy.vocab"))
infoDF = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'info_db.pandas'))
db_size = len(infoDF)
phrase_boundaries = (3, 10)

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
        phrases = set()
        lt = LoopTimer(update_after=100, avg_length=3000, target=db_size)
        for abstract_id, row in infoDF.iterrows():
            doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))
            phrases.update(findPhrasesByRule(doc, learn_rules, phrase_boundaries))
            n = lt.update(f"Find Phrases - {len(phrases)}")
            if n > 100000:
                break

        print()

        """
        =============================
                BUILD MATCHER
        =============================
        """
        matcher = PhraseMatcher(nlp.vocab, max_length=phrase_boundaries[1])
        lt = LoopTimer(update_after=100, avg_length=100, target=len(phrases))
        for p_id, phrase in enumerate(phrases):
            matcher.add(f"phrase_{p_id}", None, nlp(phrase))
            lt.update("Build Matcher")
        print()
        """
        =============================
            FIND RULES BY PHRASES
        =============================
        """
        rule_by_phrase_counter = dict()
        phrase_counter = dict()
        lt = LoopTimer(update_after=100, avg_length=3000, target=db_size)
        for abstract_id, row in infoDF.iterrows():
            doc = Doc(vocab).from_disk(os.path.join(path_to_annotations, f"{abstract_id}.spacy"))
            findRuleByPhrase(doc, matcher, rule_by_phrase_counter, phrase_counter, rules[category])
            n = lt.update(f"Find Rules - {len(rule_by_phrase_counter)}")
            if n > 100000:
                break

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

# Saving Rules
rule_file_path = os.path.join(path_to_db, "pattern_matching", learned_rules_filename)


with open(rule_file_path, 'w') as rule_file:
    for category in [j for j in rules if j in cat_set]:
        for rule in rules[category]:
            data = dict()

            data['category'] = category
            data['trigger_word'] = rule[0]
            data['dependency'] = rule[1]

            json_string = json.JSONEncoder().encode(data)

            rule_file.write(json_string + '\n')
