import json
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def load_rules(rule_file_path):
    loadrules = dict()

    with open(rule_file_path) as rule_file:
        for line in rule_file:
            data = json.loads(line)

            category = data['category']
            trigger_word = data['trigger_word']
            dependency = data['dependency']

            rule = (trigger_word, dependency)

            if category not in loadrules:
                loadrules[category] = set()

            loadrules[category].add(rule)
    return loadrules


def convert_rules(convert_to, rule_state, rules, dictionary):
    if rule_state == "text" and convert_to == "ids":
        new_rules = dict()
        rule_state = "ids"
        print("Convert from Word to ids")
        for category in rules:
            new_rules[category] = set()
            for rule in rules[category]:
                trigger_word = rule[0]
                if trigger_word in dictionary.token2id:
                    trigger_id = dictionary.token2id[trigger_word]
                    dependency = rule[1]
                    new_rule = (trigger_id, dependency)
                    new_rules[category].add(new_rule)
    elif rule_state == "ids" and convert_to == "text":
        new_rules = dict()
        rule_state = "text"
        print("Convert from ids to Word")
        for category in rules:
            new_rules[category] = set()
            for rule in rules[category]:
                trigger_word_id = rule[0]
                if trigger_word_id in dictionary:
                    trigger_word = dictionary[trigger_word_id]
                    dependency = rule[1]
                    new_rule = (trigger_word, dependency)
                    new_rules[category].add(new_rule)
    else:
        new_rules = rules

    return rule_state, new_rules


def sentence2tree(sentence_, dictionary=None, dep_type_='basicDependencies'):
    graph = nx.DiGraph()

    graph.add_node(0, token='ROOT')

    for idx, token in enumerate(sentence_['tokens']):
        word = token['originalText'].lower()
        if dictionary is None:
            token_content = word
        else:
            dictionary.doc2bow([word], allow_update=True)
            token_content = dictionary.token2id[word]
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
