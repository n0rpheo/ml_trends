import os
import mysql.connector
import operator

from src.utils.corpora import nlp_to_sent_token
import src.modules.pattern_matching as pm
from src.utils.corpora import AnnotationStream

path_to_db = "/media/norpheo/mySQL/db/ssorc"
target_path = os.path.join(path_to_db, 'features', 'rf_targets.pickle')  # output!!
rules_filename = "learned_rules_big.json"

rule_file_path = os.path.join(path_to_db, "pattern_matching", rules_filename)
dep_type = 'basicDependencies'
#dep_type = 'enhancedDependencies'
#dep_type = 'enhancedPlusPlusDependencies'

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="thesis",
)

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id FROM abstracts_ml WHERE entities LIKE '%machine learning%' AND annotated=1 LIMIT 100"
cursor.execute(sq1)

print("Collecting Abstracts")
abstracts = set()
for row in cursor:
    abstracts.add(row[0])
connection.close()

annotations = AnnotationStream(abstracts=abstracts, output='all')

rules = pm.load_rules(rule_file_path)

for abstract_id, annotation in annotations:
    sents_tokens = nlp_to_sent_token(annotation, token_type='originalText')
    for sentence in annotation['sentences']:
        sent_id = sentence['index']
        dep_tree = pm.sentence2tree(sentence)

        if dep_tree is None:
            pred = None
        else:
            cat_count = dict()
            for category in rules:
                cat_count[category] = 0
                for rule in rules[category]:
                    phrases = pm.get_phrases(dep_tree, rule)
                    cat_count[category] += len(phrases)

            sorted_cat_count = sorted(cat_count.items(), key=operator.itemgetter(1), reverse=True)

            if sorted_cat_count[0][1] == 0 or sorted_cat_count[0][1] == sorted_cat_count[1][1]:
                pred = None
            else:
                pred = sorted_cat_count[0][0]

            label_key = (abstract_id, sent_id)

        if pred is None:
            continue

        print(f"{pred}:")
        sent_tokens = " ".join(sents_tokens[sent_id])
        print(sent_tokens)
        print()
