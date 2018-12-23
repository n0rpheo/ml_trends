import operator
import os
import json
import pickle
import gensim
import multiprocessing
from functools import partial
from time import time

from src.utils.LoopTimer import LoopTimer
from src.utils import sequence_matching
from scripts.ssorc.rhetorical_function.labeling.make_deptree import make_deptree
import src.modules.pattern_matching as pm


def compare_dict(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            print(f"Key not in Dict {key}")
        else:
            if dict1[key] != dict2[key]:
                print(f"Key {key}  in Dict but different Value. {dict1[key]} vs {dict2[key]}")

    for key in dict2:
        if key not in dict1:
            print(f"Key not in Dict {key}")
        else:
            if dict1[key] != dict2[key]:
                print(f"Key {key}  in Dict but different Value. {dict2[key]} vs {dict1[key]}")


def findphrasesbyrules(rf_rules):
    phrases_ = []
    lc = LoopTimer(update_after=5000, avg_length=10000, target=len(dep_tree_dict))
    for abstract_id, sentence_id in dep_tree_dict.keys():
        dep_tree = dep_tree_dict[(abstract_id, sentence_id)]

        for rf_rule in rf_rules:
            phrases_.extend(pm.get_phrases(dep_tree, rf_rule))
        lc.update("Find Phrases By Rules")
    return phrases_


def findrulesbyphrase(category_, search_phrases):
    new_rules_ = set()

    # Calculate Prefix-Table + Setup phrase_freqs
    phrase_freqs = dict()
    prefix_tables = dict()
    for phrase_id, search_phrase in enumerate(search_phrases):
        phrase_freqs[phrase_id] = 0
        prefix_tables[phrase_id] = sequence_matching.prefix(search_phrase)

    # rule_phrase_count[Rule][Pattern-ID] = Count
    rule_phrase_count = dict()

    lc = LoopTimer(update_after=1000, avg_length=5000, target=len(dep_tree_dict))
    for abstract_id, sentence_id in dep_tree_dict.keys():
        dep_tree = dep_tree_dict[(abstract_id, sentence_id)]

        token_idxs = list()
        tokens = list()

        for vertex in dep_tree.nodes.data():
            token_idxs.append(vertex[0])
            tokens.append(vertex[1]['token'])

        token_idxs.pop(0)  # remove Root
        tokens.pop(0)

        for phrase_id, search_phrase in enumerate(search_phrases):
            prefix_table = prefix_tables[phrase_id]
            results = sequence_matching.search_matching(search_phrase, prefix_table, tokens)

            if len(results) > 0:

                phrase_freqs[phrase_id] += len(results)

                for result in results:
                    result_ids = [i for i in range(result + 1, result + 1 + len(search_phrase))]

                    in_edge = dep_tree.in_edges(result+1)

                    for v in in_edge:
                        source = v[0]
                        target = v[1]

                    while source in result_ids:
                        in_edge = dep_tree.in_edges(source)
                        for v in in_edge:
                            source = v[0]
                            target = v[1]

                    trigger_vertex = dep_tree.nodes.data()[source]
                    trigger_token = trigger_vertex['token']

                    dependency_ = dep_tree.edges[source, target]['dependency']

                    rule_ = (trigger_token, dependency_)

                    if dependency_ != "ROOT" and rule_ not in rules[category_]:
                        rule_phrase = (rule_, phrase_id)
                        if rule_phrase not in rule_phrase_count:
                            rule_phrase_count[rule_phrase] = 0
                        rule_phrase_count[rule_phrase] += 1

                        new_rules_.add(rule_)
        #lc.update("Find Rules by Phrases")

    #print()
    #print(f"{len(new_rules_)} have been found.")
    return new_rules_, phrase_freqs, rule_phrase_count


def find_and_select_rules_by_phrases(category_, select_top_, phrases_):
    print("Find Rules by Phrases started.")
    t1 = time()
    chunk_phrases = [phrases_[i:i+chunk_size] for i in range(0, len(phrases_), chunk_size)]
    frbp_func = partial(findrulesbyphrase, category)

    with multiprocessing.Pool(num_procs) as p:
        results = p.map(frbp_func, chunk_phrases)

    new_rules_ = set()
    phrase_freqs = dict()
    rule_phrase_count = dict()

    for chunk_num, result in enumerate(results):
        part_new_rules = result[0]
        part_phrase_freqs = result[1]
        part_rule_phrase_count = result[2]

        new_rules_ = new_rules_ | part_new_rules

        for key in part_phrase_freqs:
            phrase_id = key + chunk_num*chunk_size
            phrase_freqs[phrase_id] = part_phrase_freqs[key]

        for key in part_rule_phrase_count:
            rule_key = key[0]
            phrase_id = key[1] + chunk_num*chunk_size
            key2 = (rule_key, phrase_id)
            if key2 not in rule_phrase_count:
                rule_phrase_count[key2] = 0
            rule_phrase_count[key2] += part_rule_phrase_count[key]

    elapsed_time = time() - t1
    print(f"Find Rules by Phrases finished. Took {elapsed_time} seconds")

    print(f"{len(new_rules_)} have been found.")

    """
    new_rules_2, phrase_freqs2, rule_phrase_count2 = findrulesbyphrase(category_, phrases_)

    if len(new_rules_ - new_rules_2) > 0 or len(new_rules_2 - new_rules_) > 2:
        print("Regelmenge nicht gleich")

    print("Compare Freqs")
    compare_dict(phrase_freqs, phrase_freqs2)
    print("Compare Count")
    compare_dict(rule_phrase_count, rule_phrase_count2)
    """

    rule_weight = dict()
    for rule_ in new_rules_:
        rule_weight[rule_] = 0

        for phrase_id in phrase_freqs:
            phrase_freq = phrase_freqs[phrase_id]

            if (rule_, phrase_id) in rule_phrase_count:
                rule_weight[rule_] += rule_phrase_count[(rule_, phrase_id)] / phrase_freq

    sorted_rule_weight = sorted(rule_weight.items(), key=operator.itemgetter(1), reverse=True)

    for item in sorted_rule_weight[:10]:
        word_id = item[0][0]
        word = dictionary[word_id]
        dep = item[0][1]
        print(str(item[1]) + ": " + word + " -> " + dep)

    result_rules = set()

    for item in sorted_rule_weight[:select_top_]:
        result_rules.add(item[0])

    print(result_rules)

    return result_rules


cat_set = ['Background']
mod_name = "dep_tree_big"
seed_rules_filename = "seed_rules.json"
learned_rules_filename = "leaned_rules_background.json"
iterations = 3
chunk_size = 100
select_top_n = 2
num_procs = 12
make_tree = True

path_to_db = "/media/norpheo/mySQL/db/ssorc"
rule_file_path = os.path.join(path_to_db, "pattern_matching", seed_rules_filename)
manager = multiprocessing.Manager()

if make_tree:
    dep_tree_dict = manager.dict()
    dictionary = gensim.corpora.Dictionary()
    make_deptree(mod_name, dep_tree_dict, dictionary, dep_type='basicDependencies', limit=30000)
else:
    dep_tree_path = os.path.join(path_to_db, 'pattern_matching', f"{mod_name}.pickle")
    dict_path = os.path.join(path_to_db, 'pattern_matching', f"{mod_name}.dict")

    # Loading Tree-Structure
    print("Loading Tree-File")
    with open(dep_tree_path, "rb") as dt_file:
        dep_tree_dict = manager.dict(pickle.load(dt_file))
    dictionary = gensim.corpora.Dictionary.load(dict_path)

# Loading Rules
rule_state = "text"
rules = pm.load_rules(rule_file_path)

# Learning Phase
rule_state, rules = pm.convert_rules(convert_to="ids",
                                     rule_state=rule_state,
                                     rules=rules,
                                     dictionary=dictionary)

for category in [j for j in rules if j in cat_set]:
    string = f"| Learning {category} |"
    lines = "".join(["-" for i in range(len(string))])
    print(lines)
    print(string)
    print(lines)

    learn_rules = rules[category]
    for it in range(0, iterations):
        print("Iteration " + str(it + 1) + ": " + str(len(rules[category])) + " Rules")
        phrases = findphrasesbyrules(learn_rules)
        print()
        print(str(len(phrases)) + " Phrases found.")

        new_rules = find_and_select_rules_by_phrases(category, select_top_n, phrases)
        print()

        if len(new_rules) == 0:
            break
        else:
            learn_rules = new_rules
            for new_rule in new_rules:
                rules[category].add(new_rule)


# Saving Rules
rule_file_path = os.path.join(path_to_db, "pattern_matching", learned_rules_filename)

rule_state, rules = pm.convert_rules(convert_to="text",
                                     rule_state=rule_state,
                                     rules=rules,
                                     dictionary=dictionary)

with open(rule_file_path, 'w') as rule_file:
    for category in rules:
        for rule in rules[category]:
            data = dict()

            data['category'] = category
            data['trigger_word'] = rule[0]
            data['dependency'] = rule[1]

            json_string = json.JSONEncoder().encode(data)

            rule_file.write(json_string + '\n')
