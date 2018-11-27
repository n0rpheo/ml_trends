import operator
import os
import json
import networkx as nx
import pickle
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

from src.utils.corpora import AnnotationStream
from src.utils.LoopTimer import LoopTimer
from src.utils import sequence_matching


class PatternMatching:
    def __init__(self, dep_type):
        self.dep_type = dep_type

        self.pattern_matching_dir = '/media/norpheo/mySQL/db/ssorc/pattern_matching'

        self.dep_tree_dict = dict()
        self.word_dictionary = Dictionary()

        self.rules = None
        self.rules_converted = None

    def build_dep_tree_dict(self, abstracts):

        size = len(abstracts)

        annotations = AnnotationStream(abstracts=abstracts, output='all')

        lc = LoopTimer(update_after=10, avg_length=200, target=size)
        for abstract_id, annotation in annotations:
            for sentence in annotation['sentences']:
                dep_tree = sentence2tree(sentence, dictionary=self.word_dictionary, dep_type_=self.dep_type)
                sentence_id = int(sentence['index'])
                if dep_tree is not None:
                    self.dep_tree_dict[(abstract_id, sentence_id)] = dep_tree
            lc.update("Build Dep Tree Dict")
        print()
        print(f"Size of Dictionary: {len(self.word_dictionary.id2token)}")

    def convert_rules(self, convert_to):
        if self.rules_converted == "text" and convert_to == "id":
            new_rules = dict()
            self.rules_converted = "ids"
            print("Convert from Word to ids")
            for category in self.rules:
                new_rules[category] = set()
                for rule in self.rules[category]:
                    trigger_word = rule[0]
                    if trigger_word in self.word_dictionary.token2id:
                        trigger_id = self.word_dictionary.token2id[trigger_word]
                        dependency = rule[1]
                        new_rule = (trigger_id, dependency)
                        new_rules[category].add(new_rule)
            self.rules = new_rules
        elif self.rules_converted == "ids" and convert_to == "text":
            new_rules = dict()
            self.rules_converted = "text"
            print("Convert from ids to Word")
            for category in self.rules:
                new_rules[category] = set()
                for rule in self.rules[category]:
                    trigger_word_id = rule[0]
                    if trigger_word_id in self.word_dictionary.id2token:
                        trigger_word = self.word_dictionary.id2token[trigger_word_id]
                        dependency = rule[1]
                        new_rule = (trigger_word, dependency)
                        new_rules[category].add(new_rule)
            self.rules = new_rules

    def learning(self, iterations=3):
        self.convert_rules("id")
        for category in self.rules:
            self.learn_category(category=category, iterations=iterations)

    def learn_category(self, category, iterations=3):
        print(f"Learning {category}")
        learn_rules = self.rules[category]
        for it in range(0, iterations):
            print("Iteration " + str(it+1) + ": " + str(len(self.rules[category])) + " Rules")
            phrases = self.findphrasesbyrules(learn_rules)
            print()
            print(str(len(phrases)) + " Phrases found.")
            new_rules = self.findrulesbyphrase(phrases, category)
            print()

            if len(new_rules) == 0:
                break
            else:
                learn_rules = new_rules
                for new_rule in new_rules:
                    self.rules[category].add(new_rule)

    def findphrasesbyrules(self, rf_rules):
        phrases = []
        lc = LoopTimer(update_after=5000, avg_length=10000, target=len(self.dep_tree_dict))
        for abstract_id, sentence_id in self.dep_tree_dict:
            dep_tree = self.dep_tree_dict[(abstract_id, sentence_id)]

            for rule in rf_rules:
                phrases.extend(get_phrases(dep_tree, rule))
            lc.update("Find Phrases By Rules")
        return phrases

    def findrulesbyphrase(self, search_phrases, category, select_top=2):
        rules = set()

        # Calculate Prefix-Table + Setup phrase_freqs
        phrase_freqs = dict()
        prefix_tables = dict()
        for phrase_id, search_phrase in enumerate(search_phrases):
            phrase_freqs[phrase_id] = 0
            prefix_tables[phrase_id] = sequence_matching.prefix(search_phrase)

        # rule_phrase_count[Rule][Patter-ID] = Count
        rule_phrase_count = dict()

        lc = LoopTimer(update_after=1000, avg_length=5000, target=len(self.dep_tree_dict))
        for abstract_id, sentence_id in self.dep_tree_dict:
            dep_tree = self.dep_tree_dict[(abstract_id, sentence_id)]

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

                        dependency = dep_tree.edges[source, target]['dependency']

                        rule = (trigger_token, dependency)

                        if dependency != "ROOT" and rule not in self.rules[category]:
                            rule_phrase = (rule, phrase_id)
                            if rule_phrase not in rule_phrase_count:
                                rule_phrase_count[rule_phrase] = 1
                            else:
                                rule_phrase_count[rule_phrase] += 1

                            rules.add(rule)
            lc.update("Find Rules by Phrases")

        print()
        print(str(len(rules)) + ' have been found. Selecting top ' + str(select_top))

        rule_weight = dict()
        for rule in rules:
            rule_weight[rule] = 0

            for phrase_id in phrase_freqs:
                phrase_freq = phrase_freqs[phrase_id]

                if (rule, phrase_id) in rule_phrase_count:
                    rule_weight[rule] += rule_phrase_count[(rule, phrase_id)] / phrase_freq

        sorted_rule_weight = sorted(rule_weight.items(), key=operator.itemgetter(1), reverse=True)

        for count, item in enumerate(sorted_rule_weight):
            word_id = item[0][0]
            word = self.word_dictionary.id2token[word_id]
            dep = item[0][1]
            print(str(item[1]) + ": " + word + " -> " + dep)
            if count == 10:
                break

        result_rules = set()

        for count, item in enumerate(sorted_rule_weight):
            if count == select_top:
                break
            result_rules.add(item[0])

        print(result_rules)

        return result_rules

    def predict(self, sentence):
        dep_tree = sentence2tree(sentence)

        if dep_tree is None:
            return None

        self.convert_rules("text")

        cat_count = dict()
        for category in self.rules:
            cat_count[category] = 0
            for rule in self.rules[category]:
                phrases = get_phrases(dep_tree, rule)
                cat_count[category] += len(phrases)

        sorted_cat_count = sorted(cat_count.items(), key=operator.itemgetter(1), reverse=True)

        if sorted_cat_count[0][1] == 0 or sorted_cat_count[0][1] == sorted_cat_count[1][1]:
            return None
        else:
            return sorted_cat_count[0][0]

    def load_rules(self, filename):
        self.rules_converted = "text"

        rule_file_path = os.path.join(self.pattern_matching_dir, filename)

        self.rules = dict()

        with open(rule_file_path) as rule_file:
            for line in rule_file:
                data = json.loads(line)

                category = data['category']
                trigger_word = data['trigger_word']
                dependency = data['dependency']

                rule = (trigger_word, dependency)

                if category not in self.rules:
                    self.rules[category] = set()

                self.rules[category].add(rule)

    def save_rules(self, filename):
        rule_file_path = os.path.join(self.pattern_matching_dir, filename)

        self.convert_rules("text")

        with open(rule_file_path, 'w') as rule_file:

            for category in self.rules:
                for rule in self.rules[category]:
                    data = dict()

                    data['category'] = category
                    data['trigger_word'] = rule[0]
                    data['dependency'] = rule[1]

                    json_string = json.JSONEncoder().encode(data)

                    rule_file.write(json_string + '\n')

    def save_dep_tree(self, filename):
        file_path = os.path.join(self.pattern_matching_dir, filename)

        with open(file_path, 'wb') as dt_file:
            pickle.dump(self.dep_tree_dict, dt_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dep_tree(self, filename):
        file_path = os.path.join(self.pattern_matching_dir, filename)

        with open(file_path, "rb") as dt_file:
            self.dep_tree_dict = pickle.load(dt_file)


class Dictionary:
    def __init__(self):
        self.token2id = dict()
        self.id2token = dict()

    def add(self, word):
        if word not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[word] = token_id
            self.id2token[token_id] = word
        else:
            token_id = self.token2id[word]
        return token_id


def sentence2tree(sentence_, dictionary=None, dep_type_='basicDependencies'):
    graph = nx.DiGraph()

    graph.add_node(0, token='ROOT')

    for idx, token in enumerate(sentence_['tokens']):
        if dictionary is None:
            token_content = token['word'].lower()
        else:
            token_content = dictionary.add(token['word'].lower())
        graph.add_node(idx+1, token=token_content)

    for dep in sentence_[dep_type_]:
        dependency = dep['dep']
        dep_idx = int(dep['dependent'])
        dep_word = dep['dependentGloss']
        gov_idx = int(dep['governor'])
        gov_word = dep['governorGloss']

        graph.add_edge(gov_idx, dep_idx, dependency=dependency)

    return graph


def get_phrases(graph, rule):
    phrases = []
    trigger_token = rule[0]
    dependency = rule[1]
    for vertex in graph.nodes.data():
        token_idx = vertex[0]
        token_word = vertex[1]['token']

        if token_word == trigger_token:
            out_edges = graph.out_edges(token_idx)

            for out_edge in out_edges:
                source_id = out_edge[0]
                target_id = out_edge[1]
                edge_dep = graph.get_edge_data(source_id, target_id)['dependency']
                if edge_dep == dependency:
                    phrase_indexes = nx.descendants(graph, target_id)
                    phrase_indexes.add(target_id)
                    phrase = [graph.nodes.data()[word_index]['token'] for word_index in phrase_indexes]
                    if len(phrase) > 3:
                        phrases.append(phrase)
    return phrases


def draw_graph(graph):
    pos = graphviz_layout(graph, prog='dot')
    nx.draw_networkx(graph, with_label=True, node_size=200, pos=pos, arrows=True)
    plt.show()
