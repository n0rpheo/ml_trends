import operator
import os
import json

import xml.etree.ElementTree as ET
import igraph

from src.utils.corpora import xml_para_stream
from src.utils.LoopTimer import LoopTimer
from src.utils import sequence_matching


class PatternMatching:
    def __init__(self, dtype):
        self.dtype = dtype

        dirname = os.path.dirname(__file__)
        self.pattern_matching_dir = os.path.join(dirname, '../../data/processed/' + self.dtype + '/pattern_matching')

        self.xml_corpus = xml_para_stream(dtype=dtype)

        self.dep_tree_dict = dict()
        self.word_dictionary = Dictionary()

        self.rules_converted = None

    def build_dep_tree_dict(self, size=10000):
        lc = LoopTimer()
        for para_count, (doc_id, para_id, xml_string) in enumerate(self.xml_corpus):
            root = ET.fromstring(xml_string)
            for sentence in root.iter('sentence'):
                s_id = sentence.attrib['id']
                dep_tree = sentence2tree(sentence, self.word_dictionary)
                if dep_tree is not None:
                    self.dep_tree_dict[(doc_id, para_id, s_id)] = dep_tree
            lc.update("Build Dep Tree Dict")
            if para_count == size:
                break
        print()
        print(len(self.word_dictionary.id2token))

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

    def learning(self, category, iterations=3):
        self.convert_rules("id")

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
        lc = LoopTimer(update_after=1000)
        for sent_count, (doc_id, para_id, sent_id) in enumerate(self.dep_tree_dict):
            dep_tree = self.dep_tree_dict[(doc_id, para_id, sent_id)]

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

        lc = LoopTimer(update_after=1000)
        for sent_count, (doc_id, para_id, sent_id) in enumerate(self.dep_tree_dict):
            dep_tree = self.dep_tree_dict[(doc_id, para_id, sent_id)]

            token_idxs = []

            for vertex in dep_tree.vs:
                token_idxs.append(vertex.index)

            token_idxs.pop(0)  # remove Root

            token_idxs = sorted(token_idxs)

            tokens = [dep_tree.vs[t_id]['token'] for t_id in token_idxs]

            for phrase_id, search_phrase in enumerate(search_phrases):

                prefix_table = prefix_tables[phrase_id]

                results = sequence_matching.search_matching(search_phrase, prefix_table, tokens)

                if len(results) > 0:

                    phrase_freqs[phrase_id] += len(results)

                    for result in results:
                        result_ids = [i for i in range(result + 1, result + 1 + len(search_phrase))]

                        in_edges = dep_tree.incident(result+1, mode='IN')

                        in_edge = dep_tree.es[in_edges[0]]

                        while in_edge.source in result_ids:
                            in_edges = dep_tree.incident(in_edge.source, mode='IN')
                            in_edge = dep_tree.es[in_edges[0]]

                        trigger_vertex = dep_tree.vs[in_edge.source]
                        trigger_token = trigger_vertex['token']

                        dependency = in_edge['dependency']

                        rule = (trigger_token, dependency)

                        if rule not in self.rules[category]:
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


def sentence2tree(sentence, dictionary=None, dep_type='collapsed-dependencies'):
    tree = igraph.Graph(directed=True)

    tree.add_vertex()
    tree.vs[0]['token'] = 'ROOT'

    for idx, token in enumerate(sentence.iter('token')):

        if dictionary is None:
            token_content = token.find('word').text.lower()
        else:
            token_content = dictionary.add(token.find('word').text.lower())
        tree.add_vertex(token=token_content)

    for dependencies in sentence.iter('dependencies'):
        if dependencies.attrib['type'] == dep_type:
            for edge_id, dep in enumerate(dependencies.iter('dep')):
                dependency = dep.attrib['type']

                dep_idx = int(dep.find('dependent').attrib['idx'])
                gov_idx = int(dep.find('governor').attrib['idx'])

                tree.add_edge(gov_idx, dep_idx, dependency=dependency)

    if graph_is_tree(tree):
        return tree
    else:
        return None


def remove_circle(graph):
    print(graph)


def get_phrases(graph, rule):
    phrases = []
    trigger_token = rule[0]
    dependency = rule[1]
    for vertex in graph.vs:
        idx = vertex.index
        if vertex['token'] == trigger_token:
            out_edges = graph.incident(idx, mode='OUT')
            for edge_id in out_edges:
                edge = graph.es[edge_id]
                if edge['dependency'] == dependency:
                    target_id = edge.target
                    phrase_indexes = sorted(get_subvertices_from_vertex(graph, target_id))

                    phrase = [graph.vs[word_index]['token'] for word_index in phrase_indexes]
                    if len(phrase) > 3:
                        phrases.append(phrase)
    return phrases


def get_subvertices_from_vertex(graph, vertex_id):
    sub_vertices = [vertex_id]

    vertex_list = [vertex_id]

    while len(vertex_list) > 0:
        cur_vertex_id = vertex_list.pop(0)
        out_edges = graph.incident(cur_vertex_id, mode='OUT')
        for edge_id in out_edges:
            edge = graph.es[edge_id]
            target_id = edge.target
            vertex_list.append(target_id)
            sub_vertices.append(target_id)

    return sub_vertices


def graph_is_tree(graph):
    if not graph.is_dag():
        return False

    for vertex in graph.vs[1:]:
        in_edges = graph.incident(vertex, mode='IN')
        if len(in_edges) != 1:
            return False

    return True


def print_graph(graph, vertex=(0, 0)):
    level = vertex[0]
    idx = vertex[1]

    print('{:10}'.format(graph.vs[idx]['token']), end=" ")

    out_edges = graph.incident(idx, mode='OUT')
    while len(out_edges) > 0:
        edge_id = out_edges.pop()
        edge = graph.es[edge_id]
        print('{:10}'.format(edge['dependency']), end=" ")
        target_id = edge.target
        print_graph(graph, (level + 1, target_id))
        print()
        for i in range(0, level+1):
            print("          ", end=" ")


def print_edges(graph):

    for edge in graph.es:
        source = graph.vs[edge.source]['token']
        target = graph.vs[edge.target]['token']
        dependency = edge['dependency']

        print(source + "(" + str(edge.source) + ")" + " -> " + dependency + " -> " + "(" + str(edge.target) + ")" + target)
